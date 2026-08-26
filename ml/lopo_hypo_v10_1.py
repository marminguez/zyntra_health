"""Hypo V10.1: risk-aware alert state machine with notification metrics.

The contextual LSTM remains the risk model. V10.1 separates model alert state
from user notifications and rearms only after risk has genuinely cleared.
Policy selection uses validation only; the held-out patient is never used for
scaling, training, threshold/state selection, or tuning.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from event_evaluation import evaluate_events, _episodes
from lstm_hypoglycemia_classifier import build_hypoglycemia_classifier
from ohio_t1dm_loader import load_ohio_directory, dataset_summary

SEED = 42
LOOKBACK = 48
BASE = ["glucose", "glucose_delta_5m", "glucose_delta_15m", "glucose_delta_30m", "glucose_acceleration_15m"]
CONTEXT = ["iob_simple", "carbs_recent_30m", "carbs_recent_60m", "basal_rate", "exercise_intensity", "hour_sin", "hour_cos"]
FEATURES = BASE + CONTEXT
THRESHOLDS = np.arange(0.40, 0.91, 0.05)
PERSISTENCE = (1, 2, 3)
CLEAR_STEPS = (2, 3, 4, 6)
REARM_MARGIN = (0.05, 0.10, 0.15)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def add_context(df):
    out = []
    for pid, p in df.groupby("p_id"):
        p = p.sort_index().copy()
        p["carbs_recent_30m"] = p.carbs_g.rolling(6, min_periods=1).sum()
        p["carbs_recent_60m"] = p.carbs_g.rolling(12, min_periods=1).sum()
        h = p.index.hour + p.index.minute / 60.0
        p["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
        p["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
        out.append(p)
    return pd.concat(out).sort_index()


def sequences(frame, scaler):
    xs, ys, meta = [], [], []
    for pid, p in frame.groupby("p_id"):
        p = p.sort_index().copy()
        z = scaler.transform(p[FEATURES].fillna(0.0))
        for i in range(LOOKBACK, len(p)):
            r = p.iloc[i]
            xs.append(z[i-LOOKBACK:i])
            ys.append(int(r.target_hypo))
            meta.append({
                "p_id": int(pid),
                "timestamp": p.index[i],
                "glucose": float(r.glucose),
            })
    return np.asarray(xs, np.float32), np.asarray(ys, np.int32), pd.DataFrame(meta)


def validation_patient(ids, held_out):
    candidates = [x for x in sorted(ids) if x != held_out]
    return candidates[(SEED + int(held_out)) % len(candidates)]


def state_machine(meta, probs, threshold, persistence, clear_steps, rearm_margin):
    """Return model-state and user-notification signals.

    NORMAL -> WATCH after probability >= threshold.
    WATCH -> ALERTED once persistence is met; emit exactly one notification.
    ALERTED remains active while risk stays elevated.
    Rearm only after probability remains below (threshold - margin) for
    clear_steps consecutive samples, then a genuinely new episode may notify.
    """
    work = meta.reset_index(drop=True).copy()
    work["probability"] = np.asarray(probs)
    model_alert = np.zeros(len(work), dtype=float)
    notification = np.zeros(len(work), dtype=float)

    for _, idxs in work.groupby("p_id").groups.items():
        idxs = list(idxs)
        state = "NORMAL"
        high_streak = 0
        clear_streak = 0
        clear_threshold = max(0.0, float(threshold) - float(rearm_margin))

        for pos in idxs:
            p = float(work.at[pos, "probability"])

            if state == "NORMAL":
                if p >= threshold:
                    high_streak = 1
                    state = "WATCH"
                else:
                    high_streak = 0

            elif state == "WATCH":
                if p >= threshold:
                    high_streak += 1
                    if high_streak >= persistence:
                        state = "ALERTED"
                        model_alert[pos] = 1.0
                        notification[pos] = 1.0
                        clear_streak = 0
                else:
                    state = "NORMAL"
                    high_streak = 0

            elif state == "ALERTED":
                model_alert[pos] = 1.0
                if p < clear_threshold:
                    clear_streak += 1
                    if clear_streak >= clear_steps:
                        state = "NORMAL"
                        high_streak = 0
                        clear_streak = 0
                else:
                    clear_streak = 0

    return model_alert, notification


def notification_metrics(meta, notification_signal):
    meta = meta.reset_index(drop=True).copy()
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    meta["notify"] = np.asarray(notification_signal) >= 0.5
    notifications = int(meta["notify"].sum())
    patient_days = 0.0
    per_patient = []
    for pid, frame in meta.groupby("p_id"):
        frame = frame.sort_values("timestamp")
        days = max((frame.timestamp.max() - frame.timestamp.min()).total_seconds() / 86400.0, 5.0 / 1440.0)
        n = int(frame.notify.sum())
        patient_days += days
        per_patient.append({"p_id": int(pid), "notifications": n, "observed_days": days, "notifications_per_day": n / days})
    return {
        "notifications": notifications,
        "observed_patient_days": patient_days,
        "notifications_per_patient_day": notifications / patient_days if patient_days else None,
        "per_patient": per_patient,
    }


def evaluate_policy(meta, probs, params):
    model_alert, notification = state_machine(meta, probs, **params)
    event_metrics = evaluate_events(meta, model_alert, 0.5, 30)
    notify_metrics = notification_metrics(meta, notification)
    return event_metrics, notify_metrics


def policy_grid(meta, probs):
    rows = []
    for threshold in THRESHOLDS:
        for persistence in PERSISTENCE:
            for clear_steps in CLEAR_STEPS:
                for rearm_margin in REARM_MARGIN:
                    params = {
                        "threshold": float(threshold),
                        "persistence": int(persistence),
                        "clear_steps": int(clear_steps),
                        "rearm_margin": float(rearm_margin),
                    }
                    ev, nt = evaluate_policy(meta, probs, params)
                    rows.append({
                        **params,
                        "event_recall": ev["event_recall"],
                        "median_warning_minutes": ev["median_warning_minutes"],
                        "model_false_alerts_per_day": ev["false_alerts_per_patient_day"],
                        "model_false_alert_episodes": ev["false_alert_episodes"],
                        "notifications_per_day": nt["notifications_per_patient_day"],
                        "notifications": nt["notifications"],
                        "detected_events": ev["detected_events"],
                        "hypoglycemia_events": ev["hypoglycemia_events"],
                    })
    return pd.DataFrame(rows)


def choose_policy(grid):
    ok = grid[(grid.event_recall >= 0.90) & (grid.median_warning_minutes >= 15.0)]
    if len(ok):
        row = ok.sort_values(
            ["notifications_per_day", "model_false_alerts_per_day", "event_recall", "median_warning_minutes"],
            ascending=[True, True, False, False],
        ).iloc[0]
        tag = "constrained_min_notifications"
    else:
        row = grid.sort_values(
            ["event_recall", "notifications_per_day", "model_false_alerts_per_day"],
            ascending=[False, True, True],
        ).iloc[0]
        tag = "fallback_max_recall"
    params = {
        "threshold": float(row.threshold),
        "persistence": int(row.persistence),
        "clear_steps": int(row.clear_steps),
        "rearm_margin": float(row.rearm_margin),
    }
    return params, tag


def train_fold(data, test_id, outdir):
    seed_all()
    remaining = [int(x) for x in sorted(data.p_id.unique()) if int(x) != int(test_id)]
    val_id = validation_patient(remaining, test_id)
    train_ids = [x for x in remaining if x != val_id]
    train = data[data.p_id.isin(train_ids)]
    val = data[data.p_id == val_id]
    test = data[data.p_id == test_id]

    scaler = StandardScaler().fit(train[FEATURES].fillna(0.0))
    xtr, ytr, _ = sequences(train, scaler)
    xv, yv, mv = sequences(val, scaler)
    xt, yt, mt = sequences(test, scaler)

    classes = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_hypoglycemia_classifier(LOOKBACK, len(FEATURES))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(
        xtr, ytr,
        validation_data=(xv, yv),
        class_weight=class_weight,
        epochs=35,
        batch_size=128,
        shuffle=False,
        callbacks=[EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True, mode="max")],
        verbose=0,
    )

    p_val = model.predict(xv, verbose=0).ravel()
    p_test = model.predict(xt, verbose=0).ravel()

    grid = policy_grid(mv, p_val)
    params, policy = choose_policy(grid)
    fold_dir = outdir / f"patient_{test_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(fold_dir / "validation_risk_state_grid.csv", index=False)

    ev, nt = evaluate_policy(mt, p_test, params)
    auc = float(roc_auc_score(yt, p_test)) if len(np.unique(yt)) > 1 else None
    return {
        "test_patient": int(test_id),
        "validation_patient": int(val_id),
        "train_patients": train_ids,
        "selected_policy": params,
        "selection_policy": policy,
        "roc_auc": auc,
        "hypoglycemia_events": ev["hypoglycemia_events"],
        "detected_events": ev["detected_events"],
        "missed_events": ev["missed_events"],
        "event_recall": ev["event_recall"],
        "median_warning_minutes": ev["median_warning_minutes"],
        "model_false_alert_episodes": ev["false_alert_episodes"],
        "model_false_alerts_per_patient_day": ev["false_alerts_per_patient_day"],
        "notifications": nt["notifications"],
        "notifications_per_patient_day": nt["notifications_per_patient_day"],
        "observed_patient_days": ev["observed_patient_days"],
    }


def aggregate(folds):
    d = pd.DataFrame(folds)
    events = int(d.hypoglycemia_events.sum())
    detected = int(d.detected_events.sum())
    full = (
        (d.event_recall >= 0.90)
        & (d.notifications_per_patient_day <= 1.0)
        & (d.median_warning_minutes >= 15.0)
    )
    return {
        "patients": len(d),
        "pooled_events": events,
        "pooled_detected": detected,
        "pooled_event_recall": detected / events if events else None,
        "mean_patient_event_recall": float(d.event_recall.mean()),
        "median_patient_event_recall": float(d.event_recall.median()),
        "mean_model_false_alerts_per_patient_day": float(d.model_false_alerts_per_patient_day.mean()),
        "mean_notifications_per_patient_day": float(d.notifications_per_patient_day.mean()),
        "median_notifications_per_patient_day": float(d.notifications_per_patient_day.median()),
        "median_warning_minutes_across_patients": float(d.median_warning_minutes.dropna().median()),
        "patients_meeting_full_product_target": int(full.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("models/v10_1_risk_state_machine"), type=Path)
    args = ap.parse_args()

    data = add_context(load_ohio_directory(args.data_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary(data).to_csv(args.output_dir / "dataset_summary.csv", index=False)
    folds = []
    for pid in sorted(int(x) for x in data.p_id.unique()):
        print(f"V10.1 held-out patient {pid}", flush=True)
        folds.append(train_fold(data, pid, args.output_dir))
        pd.DataFrame(folds).to_csv(args.output_dir / "v10_1_per_patient.csv", index=False)

    report = {
        "model": "hypo_v10_1_risk_aware_state_machine",
        "seed": SEED,
        "features": FEATURES,
        "lookback_minutes": 240,
        "prediction_horizon_minutes": 30,
        "protocol": "LOPO; contextual LSTM plus validation-selected risk-aware state machine; user notifications measured separately from model alert episodes",
        "state_machine_grid": {
            "thresholds": [round(float(x), 2) for x in THRESHOLDS],
            "persistence_steps": list(PERSISTENCE),
            "clear_steps": list(CLEAR_STEPS),
            "rearm_margin": list(REARM_MARGIN),
        },
        "selection_constraints": {
            "event_recall_min": 0.90,
            "median_warning_minutes_min": 15.0,
        },
        "folds": folds,
        "aggregate": aggregate(folds),
        "research_target": {
            "event_recall_min": 0.90,
            "notifications_per_patient_day_max": 1.0,
            "median_warning_minutes_min": 15,
        },
        "clinical_status": "research only; not clinically validated",
    }
    (args.output_dir / "v10_1_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
