"""Product-oriented alert quality analysis for Zyntra hypoglycemia prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from event_evaluation import _episodes, evaluate_events

NEAR_MISS_GLUCOSE = 80.0


def _glucose_at_or_before(frame, timestamp, minutes_before):
    target = timestamp - pd.Timedelta(minutes=minutes_before)
    candidates = frame[frame["timestamp"] <= target]
    if candidates.empty:
        return np.nan
    return float(candidates.iloc[-1]["glucose"])


def _glucose_dynamics(frame, timestamp):
    """Describe recent CGM direction without changing the model yet."""
    now_rows = frame[frame["timestamp"] <= timestamp]
    if now_rows.empty:
        return {"delta_5m": np.nan, "delta_15m": np.nan, "delta_30m": np.nan, "acceleration_15m": np.nan}
    current = float(now_rows.iloc[-1]["glucose"])
    g5 = _glucose_at_or_before(frame, timestamp, 5)
    g15 = _glucose_at_or_before(frame, timestamp, 15)
    g30 = _glucose_at_or_before(frame, timestamp, 30)
    delta5 = current - g5 if np.isfinite(g5) else np.nan
    delta15 = current - g15 if np.isfinite(g15) else np.nan
    delta30 = current - g30 if np.isfinite(g30) else np.nan
    # Difference between the most recent 15-minute slope and the preceding 15-minute slope.
    previous15 = g15 - g30 if np.isfinite(g15) and np.isfinite(g30) else np.nan
    acceleration = delta15 - previous15 if np.isfinite(delta15) and np.isfinite(previous15) else np.nan
    return {"delta_5m": delta5, "delta_15m": delta15, "delta_30m": delta30, "acceleration_15m": acceleration}


def threshold_sweep(metadata, probabilities, thresholds=None, horizon_minutes=30):
    """Compare event recall, warning time and false-alert burden across thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.91, 0.05)
    rows = []
    for threshold in thresholds:
        metrics = evaluate_events(metadata, probabilities, float(threshold), horizon_minutes)
        analysis = build_alert_analysis(metadata, probabilities, float(threshold), horizon_minutes)
        fps = analysis[analysis["classification"] == "FP"]
        rows.append({
            "threshold": round(float(threshold), 3),
            "hypoglycemia_events": metrics["hypoglycemia_events"],
            "detected_events": metrics["detected_events"],
            "missed_events": metrics["missed_events"],
            "event_recall": metrics["event_recall"],
            "median_warning_minutes": metrics["median_warning_minutes"],
            "false_alert_episodes": metrics["false_alert_episodes"],
            "false_alerts_per_patient_day": metrics["false_alerts_per_patient_day"],
            "near_miss_alerts": int((fps["fp_subtype"] == "near_miss").sum()) if not fps.empty else 0,
            "clear_false_alerts": int((fps["fp_subtype"] == "clear_false_positive").sum()) if not fps.empty else 0,
        })
    return pd.DataFrame(rows)


def build_alert_analysis(metadata, probabilities, threshold, horizon_minutes=30):
    """Return one row per real hypo episode and per unmatched alert episode.

    False positives are split into near-misses (future glucose <80 mg/dL but not <70)
    and clear false positives. Recent CGM deltas are diagnostic features only; they
    are not yet fed into the LSTM, preventing an unmeasured architecture change.
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
            candidates = frame[frame["alert"] & (frame["timestamp"] < event.start) &
                               (frame["timestamp"] >= event.start - pd.Timedelta(minutes=horizon_minutes))]
            event_min = float(frame.loc[(frame.timestamp >= event.start) & (frame.timestamp <= event.end), "glucose"].min())
            if candidates.empty:
                rows.append({"p_id":pid,"record_type":"hypo_event","classification":"FN","fp_subtype":"",
                    "event_id":event_idx,"event_start":event.start,"event_end":event.end,"alert_time":pd.NaT,
                    "lead_minutes":np.nan,"probability":np.nan,"glucose_at_alert":np.nan,"event_min_glucose":event_min,
                    "future_min_glucose":np.nan,"delta_5m":np.nan,"delta_15m":np.nan,"delta_30m":np.nan,"acceleration_15m":np.nan})
                continue
            first = candidates.iloc[0]
            for idx, alert in enumerate(alert_events):
                if alert.start <= first.timestamp <= alert.end: matched_alerts.add(idx)
            dynamics = _glucose_dynamics(frame, first.timestamp)
            rows.append({"p_id":pid,"record_type":"hypo_event","classification":"TP","fp_subtype":"",
                "event_id":event_idx,"event_start":event.start,"event_end":event.end,"alert_time":first.timestamp,
                "lead_minutes":(event.start-first.timestamp).total_seconds()/60.0,"probability":float(first.probability),
                "glucose_at_alert":float(first.glucose),"event_min_glucose":event_min,"future_min_glucose":event_min,**dynamics})

        for idx, alert in enumerate(alert_events):
            if idx in matched_alerts: continue
            segment = frame[(frame.timestamp >= alert.start) & (frame.timestamp <= alert.end)]
            first = segment.iloc[0]
            future = frame[(frame.timestamp > alert.start) & (frame.timestamp <= alert.start + pd.Timedelta(minutes=horizon_minutes))]
            future_min = float(future.glucose.min()) if not future.empty else np.nan
            subtype = "near_miss" if np.isfinite(future_min) and 70 <= future_min < NEAR_MISS_GLUCOSE else "clear_false_positive"
            dynamics = _glucose_dynamics(frame, alert.start)
            rows.append({"p_id":pid,"record_type":"alert_episode","classification":"FP","fp_subtype":subtype,
                "event_id":np.nan,"event_start":pd.NaT,"event_end":pd.NaT,"alert_time":alert.start,"lead_minutes":np.nan,
                "probability":float(segment.probability.max()),"glucose_at_alert":float(first.glucose),"event_min_glucose":np.nan,
                "future_min_glucose":future_min,**dynamics})

    columns=["p_id","record_type","classification","fp_subtype","event_id","event_start","event_end","alert_time",
             "lead_minutes","probability","glucose_at_alert","event_min_glucose","future_min_glucose","delta_5m","delta_15m","delta_30m","acceleration_15m"]
    return pd.DataFrame(rows,columns=columns)
