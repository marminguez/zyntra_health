"""Product-oriented alert quality analysis for Zyntra hypoglycemia prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from event_evaluation import _episodes, evaluate_events


def threshold_sweep(metadata, probabilities, thresholds=None, horizon_minutes=30):
    """Compare event recall, warning time and false-alert burden across thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.91, 0.05)
    rows = []
    for threshold in thresholds:
        metrics = evaluate_events(metadata, probabilities, float(threshold), horizon_minutes)
        rows.append({
            "threshold": round(float(threshold), 3),
            "hypoglycemia_events": metrics["hypoglycemia_events"],
            "detected_events": metrics["detected_events"],
            "missed_events": metrics["missed_events"],
            "event_recall": metrics["event_recall"],
            "median_warning_minutes": metrics["median_warning_minutes"],
            "false_alert_episodes": metrics["false_alert_episodes"],
            "false_alerts_per_patient_day": metrics["false_alerts_per_patient_day"],
        })
    return pd.DataFrame(rows)


def build_alert_analysis(metadata, probabilities, threshold, horizon_minutes=30):
    """Return one row per real hypo episode and per unmatched alert episode.

    A TP must be emitted before event onset and no more than horizon_minutes before it.
    This makes lead time clinically interpretable and prevents post-onset alerts from
    being counted as successful predictions.
    """
    meta = metadata.reset_index(drop=True).copy()
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    meta["probability"] = np.asarray(probabilities)
    meta["alert"] = meta["probability"] >= threshold
    rows = []

    for pid, frame in meta.groupby("p_id"):
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        real_events = _episodes(frame["glucose"] < 70, frame["timestamp"])
        alert_events = _episodes(frame["alert"], frame["timestamp"])
        matched_alerts = set()

        for event_idx, event in enumerate(real_events, 1):
            candidates = frame[
                frame["alert"]
                & (frame["timestamp"] < event.start)
                & (frame["timestamp"] >= event.start - pd.Timedelta(minutes=horizon_minutes))
            ]
            if candidates.empty:
                rows.append({
                    "p_id": pid, "record_type": "hypo_event", "classification": "FN",
                    "event_id": event_idx, "event_start": event.start, "event_end": event.end,
                    "alert_time": pd.NaT, "lead_minutes": np.nan,
                    "probability": np.nan, "glucose_at_alert": np.nan,
                    "event_min_glucose": float(frame.loc[(frame.timestamp >= event.start) & (frame.timestamp <= event.end), "glucose"].min()),
                })
                continue

            first = candidates.iloc[0]
            lead = (event.start - first.timestamp).total_seconds() / 60.0
            for idx, alert in enumerate(alert_events):
                if alert.start <= first.timestamp <= alert.end:
                    matched_alerts.add(idx)
            rows.append({
                "p_id": pid, "record_type": "hypo_event", "classification": "TP",
                "event_id": event_idx, "event_start": event.start, "event_end": event.end,
                "alert_time": first.timestamp, "lead_minutes": lead,
                "probability": float(first.probability), "glucose_at_alert": float(first.glucose),
                "event_min_glucose": float(frame.loc[(frame.timestamp >= event.start) & (frame.timestamp <= event.end), "glucose"].min()),
            })

        for idx, alert in enumerate(alert_events):
            if idx in matched_alerts:
                continue
            segment = frame[(frame.timestamp >= alert.start) & (frame.timestamp <= alert.end)]
            first = segment.iloc[0]
            future_end = alert.start + pd.Timedelta(minutes=horizon_minutes)
            future = frame[(frame.timestamp > alert.start) & (frame.timestamp <= future_end)]
            rows.append({
                "p_id": pid, "record_type": "alert_episode", "classification": "FP",
                "event_id": np.nan, "event_start": pd.NaT, "event_end": pd.NaT,
                "alert_time": alert.start, "lead_minutes": np.nan,
                "probability": float(segment.probability.max()), "glucose_at_alert": float(first.glucose),
                "event_min_glucose": float(future.glucose.min()) if not future.empty else np.nan,
            })

    columns = ["p_id", "record_type", "classification", "event_id", "event_start", "event_end",
               "alert_time", "lead_minutes", "probability", "glucose_at_alert", "event_min_glucose"]
    return pd.DataFrame(rows, columns=columns)
