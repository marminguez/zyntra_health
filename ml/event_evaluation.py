"""Event-based evaluation for Zyntra hypoglycemia prediction.

A 5-minute sample is not a clinical event. This module groups consecutive
positive samples into episodes and reports product/clinical metrics:
- event recall
- warning lead time
- false alert episodes per patient-day

Predictions are assumed to forecast hypoglycemia HORIZON_MINUTES ahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TIMESTEP_MINUTES = 5
HORIZON_MINUTES = 30


@dataclass
class Episode:
    start: pd.Timestamp
    end: pd.Timestamp


def _episodes(mask: Iterable[bool], timestamps: Iterable[pd.Timestamp], max_gap_minutes=5):
    """Group consecutive True samples into episodes."""
    episodes = []
    current_start = None
    current_end = None
    max_gap = pd.Timedelta(minutes=max_gap_minutes)

    for active, ts in zip(mask, timestamps):
        ts = pd.Timestamp(ts)
        if active:
            if current_start is None:
                current_start = ts
                current_end = ts
            elif ts - current_end <= max_gap:
                current_end = ts
            else:
                episodes.append(Episode(current_start, current_end))
                current_start = ts
                current_end = ts
        elif current_start is not None:
            episodes.append(Episode(current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None:
        episodes.append(Episode(current_start, current_end))
    return episodes


def evaluate_events(test_metadata, probabilities, threshold, horizon_minutes=HORIZON_MINUTES):
    """Evaluate alerts against real hypoglycemia episodes patient by patient.

    test_metadata must contain: p_id, timestamp, glucose.
    An alert emitted at time t forecasts risk at t+horizon. It detects an event
    when that forecasted time lands inside the real glucose<70 episode.
    Consecutive alert samples are grouped into a single alert episode.
    """
    meta = test_metadata.reset_index(drop=True).copy()
    if len(meta) != len(probabilities):
        raise ValueError("test_metadata and probabilities must have equal length")

    meta["probability"] = np.asarray(probabilities)
    meta["alert"] = meta["probability"] >= threshold
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])

    total_events = 0
    detected_events = 0
    false_alert_episodes = 0
    total_alert_episodes = 0
    lead_times = []
    patient_days = 0.0
    per_patient = []

    for pid, frame in meta.groupby("p_id"):
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        real_events = _episodes(frame["glucose"] < 70, frame["timestamp"])
        alert_episodes = _episodes(frame["alert"], frame["timestamp"])

        duration_days = max(
            (frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 86400.0,
            TIMESTEP_MINUTES / 1440.0,
        )
        patient_days += duration_days

        detected_for_patient = 0
        matched_alerts = set()

        for event in real_events:
            matching = frame[
                frame["alert"]
                & ((frame["timestamp"] + pd.Timedelta(minutes=horizon_minutes)) >= event.start)
                & ((frame["timestamp"] + pd.Timedelta(minutes=horizon_minutes)) <= event.end)
            ]
            if not matching.empty:
                detected_events += 1
                detected_for_patient += 1
                first_alert = matching["timestamp"].min()
                lead_times.append((event.start - first_alert).total_seconds() / 60.0)

                for idx, alert_episode in enumerate(alert_episodes):
                    if alert_episode.start <= first_alert <= alert_episode.end:
                        matched_alerts.add(idx)

        false_for_patient = len(alert_episodes) - len(matched_alerts)
        false_alert_episodes += false_for_patient
        total_alert_episodes += len(alert_episodes)
        total_events += len(real_events)

        per_patient.append({
            "p_id": int(pid) if isinstance(pid, (int, np.integer)) else str(pid),
            "hypoglycemia_events": len(real_events),
            "detected_events": detected_for_patient,
            "alert_episodes": len(alert_episodes),
            "false_alert_episodes": false_for_patient,
            "observed_days": round(duration_days, 3),
        })

    event_recall = detected_events / total_events if total_events else None
    false_alerts_per_patient_day = false_alert_episodes / patient_days if patient_days else None

    return {
        "hypoglycemia_events": total_events,
        "detected_events": detected_events,
        "missed_events": total_events - detected_events,
        "event_recall": event_recall,
        "median_warning_minutes": float(np.median(lead_times)) if lead_times else None,
        "mean_warning_minutes": float(np.mean(lead_times)) if lead_times else None,
        "min_warning_minutes": float(np.min(lead_times)) if lead_times else None,
        "max_warning_minutes": float(np.max(lead_times)) if lead_times else None,
        "alert_episodes": total_alert_episodes,
        "false_alert_episodes": false_alert_episodes,
        "false_alerts_per_patient_day": false_alerts_per_patient_day,
        "observed_patient_days": patient_days,
        "per_patient": per_patient,
        "definition": {
            "hypoglycemia": "glucose < 70 mg/dL",
            "prediction_horizon_minutes": horizon_minutes,
            "episode_gap_minutes": TIMESTEP_MINUTES,
            "note": "Consecutive positive samples are counted as one episode/alert, not multiple events.",
        },
    }
