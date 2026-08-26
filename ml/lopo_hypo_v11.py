"""Hypo V11: population model plus 14-day patient-specific calibration.

For each held-out patient:
1) Train the contextual LSTM only on other patients.
2) Select a population alert policy on a separate external validation patient.
3) Use the held-out patient's first 14 days only to personalize the alert policy.
4) Evaluate both population and personalized policies only on later days.

No future patient data is used for calibration. The LSTM itself is not fine-tuned
in V11; this isolates the value of lightweight patient-specific calibration.
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

from event_evaluation import evaluate_events
from lstm_hypoglycemia_classifier import build_hypoglycemia_classifier
from ohio_t1dm_loader import load_ohio_directory, dataset_summary

SEED = 42
LOOKBACK = 48
CALIBRATION_DAYS = 14
MIN_CALIBRATION_EVENTS = 3
BASE = ["glucose", "glucose_delta_5m", "glucose_delta_15m", "glucose_delta_30m", "glucose_acceleration_15m"]
CONTEXT = ["iob_simple", "carbs_recent_30m", "carbs_recent_60m", "basal_rate", "exercise_intensity", "hour_sin", "hour_cos"]
FEATURES = BASE + CONTEXT
THRESHOLDS = np.arange(0.35, 0.91, 0.05)
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
            meta.append({"p_id": int(pid), "timestamp": p.index[i], "glucose": float(r.glucose)})
    return np.asarray(xs, np.float32), np.asarray(ys, np.int32), pd.DataFrame(meta)


def external_validation_patient(ids, held_out):
    candidates = [x for x in sorted(ids) if x != held_out]
    return candidates[(SEED + int(held_out)) % len(candidates)]


def state_machine(meta, probs, threshold, persistence, clear_steps, rearm_margin):
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
                    if persistence == 1:
                        state = "ALERTED"
                        model_alert[pos] = 1.0
                        notification[pos] = 1.0
                        clear_streak = 0
                    else:
                        state = "WATCH"
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


def notification_metrics(meta, signal):
    m = meta.reset_index(drop=True).copy()
    m["timestamp"] = pd.to_datetime(m.timestamp)
    m["notify"] = np.asarray(signal) >= 0.5
    n = int(m.notify.sum())
    days = 0.0
    for _, f in m.groupby("p_id"):
        days += max((f.timestamp.max() - f.timestamp.min()).total_seconds() / 86400.0, 5.0 / 1440.0)
    return {"notifications": n, "notifications_per_patient_day": n / days if days else None, "observed_patient_days": days}


def evaluate_policy(meta, probs, params):
    model_alert, notification = state_machine(meta, probs, **params)
    ev = evaluate_events(meta, model_alert, 0.5, 30)
    nt = notification_metrics(meta, notification)
    return ev, nt


def policy_grid(meta, probs):
    rows = []
    for t in THRESHOLDS:
        for k in PERSISTENCE:
            for c in CLEAR_STEPS:
                for r in REARM_MARGIN:
                    params = {"threshold": float(t), "persistence": int(k), "clear_steps": int(c), "rearm_margin": float(r)}
                    ev, nt = evaluate_policy(meta, probs, params)
                    rows.append({**params,
                                 "event_recall": ev["event_recall"],
                                 "median_warning_minutes": ev["median_warning_minutes"],
                                 "model_false_alerts_per_day": ev["false_alerts_per_patient_day"],
                                 "notifications_per_day": nt["notifications_per_patient_day"],
                                 "detected_events": ev["detected_events"],
                                 "hypoglycemia_events": ev["hypoglycemia_events"]})
    return pd.DataFrame(rows)


def choose_policy(grid):
    ok = grid[(grid.event_recall >= 0.90) & (grid.median_warning_minutes >= 15.0)]
    if len(ok):
        row = ok.sort_values(["notifications_per_day", "model_false_alerts_per_day", "event_recall"], ascending=[True, True, False]).iloc[0]
        tag = "constrained_min_notifications"
    else:
        row = grid.sort_values(["event_recall", "notifications_per_day", "model_false_alerts_per_day"], ascending=[False, True, True]).iloc[0]
        tag = "fallback_max_recall"
    params = {"threshold": float(row.threshold), "persistence": int(row.persistence), "clear_steps": int(row.clear_steps), "rearm_margin": float(row.rearm_margin)}
    return params, tag


def temporal_split_patient(frame):
    p = frame.sort_index().copy()
    start = p.index.min()
    cutoff = start + pd.Timedelta(days=CALIBRATION_DAYS)
    calibration = p[p.index < cutoff].copy()
    evaluation = p[p.index >= cutoff].copy()
    if len(calibration) <= LOOKBACK or len(evaluation) <= LOOKBACK:
        raise RuntimeError("Patient does not contain enough data for 14-day calibration plus evaluation")
    return calibration, evaluation, cutoff


def count_real_hypo_events(meta):
    if meta.empty:
        return 0
    temp = meta.sort_values("timestamp").copy()
    low = temp.glucose < 70
    starts = low & ~low.shift(1, fill_value=False)
    return int(starts.sum())


def train_fold(data, test_id, outdir):
    seed_all()
    all_other = [int(x) for x in sorted(data.p_id.unique()) if int(x) != int(test_id)]
    val_id = external_validation_patient(all_other, test_id)
    train_ids = [x for x in all_other if x != val_id]

    train = data[data.p_id.isin(train_ids)]
    external_val = data[data.p_id == val_id]
    target_full = data[data.p_id == test_id]
    target_cal, target_eval, cutoff = temporal_split_patient(target_full)

    scaler = StandardScaler().fit(train[FEATURES].fillna(0.0))
    xtr, ytr, _ = sequences(train, scaler)
    xv, yv, mv = sequences(external_val, scaler)
    xcal, ycal, mcal = sequences(target_cal, scaler)
    xev, yev, mev = sequences(target_eval, scaler)

    classes = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_hypoglycemia_classifier(LOOKBACK, len(FEATURES))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(xtr, ytr, validation_data=(xv, yv), class_weight=class_weight,
              epochs=35, batch_size=128, shuffle=False,
              callbacks=[EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True, mode="max")], verbose=0)

    p_val = model.predict(xv, verbose=0).ravel()
    p_cal = model.predict(xcal, verbose=0).ravel()
    p_eval = model.predict(xev, verbose=0).ravel()

    population_grid = policy_grid(mv, p_val)
    population_policy, population_selection = choose_policy(population_grid)

    calibration_grid = policy_grid(mcal, p_cal)
    calibration_events = count_real_hypo_events(mcal)
    if calibration_events >= MIN_CALIBRATION_EVENTS:
        personalized_policy, personalized_selection = choose_policy(calibration_grid)
        personalization_status = "personalized"
    else:
        personalized_policy = population_policy.copy()
        personalized_selection = "fallback_population_insufficient_events"
        personalization_status = "insufficient_calibration_events"

    ev_pop, nt_pop = evaluate_policy(mev, p_eval, population_policy)
    ev_per, nt_per = evaluate_policy(mev, p_eval, personalized_policy)
    auc = float(roc_auc_score(yev, p_eval)) if len(np.unique(yev)) > 1 else None

    fold_dir = outdir / f"patient_{test_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    population_grid.to_csv(fold_dir / "population_policy_grid.csv", index=False)
    calibration_grid.to_csv(fold_dir / "personal_calibration_grid_14d.csv", index=False)

    return {
        "test_patient": int(test_id),
        "population_validation_patient": int(val_id),
        "population_train_patients": train_ids,
        "calibration_days": CALIBRATION_DAYS,
        "calibration_cutoff": str(cutoff),
        "calibration_hypoglycemia_events": calibration_events,
        "personalization_status": personalization_status,
        "population_policy": population_policy,
        "population_selection": population_selection,
        "personalized_policy": personalized_policy,
        "personalized_selection": personalized_selection,
        "roc_auc_post_calibration": auc,
        "evaluation_hypoglycemia_events": ev_per["hypoglycemia_events"],
        "population_detected_events": ev_pop["detected_events"],
        "population_event_recall": ev_pop["event_recall"],
        "population_warning_minutes": ev_pop["median_warning_minutes"],
        "population_notifications_per_day": nt_pop["notifications_per_patient_day"],
        "personalized_detected_events": ev_per["detected_events"],
        "personalized_event_recall": ev_per["event_recall"],
        "personalized_warning_minutes": ev_per["median_warning_minutes"],
        "personalized_notifications_per_day": nt_per["notifications_per_patient_day"],
        "recall_delta_pp": (ev_per["event_recall"] - ev_pop["event_recall"]) * 100.0 if ev_per["event_recall"] is not None and ev_pop["event_recall"] is not None else None,
        "notifications_delta_per_day": nt_per["notifications_per_patient_day"] - nt_pop["notifications_per_patient_day"],
        "evaluation_observed_days": ev_per["observed_patient_days"],
    }


def aggregate(folds):
    d = pd.DataFrame(folds)
    events = int(d.evaluation_hypoglycemia_events.sum())
    pop_detected = int(d.population_detected_events.sum())
    per_detected = int(d.personalized_detected_events.sum())
    full = (d.personalized_event_recall >= 0.90) & (d.personalized_notifications_per_day <= 1.0) & (d.personalized_warning_minutes >= 15.0)
    return {
        "patients": len(d),
        "patients_personalized": int((d.personalization_status == "personalized").sum()),
        "patients_fallback_population": int((d.personalization_status != "personalized").sum()),
        "post_calibration_events": events,
        "population_pooled_detected": pop_detected,
        "population_pooled_event_recall": pop_detected / events if events else None,
        "personalized_pooled_detected": per_detected,
        "personalized_pooled_event_recall": per_detected / events if events else None,
        "mean_population_notifications_per_day": float(d.population_notifications_per_day.mean()),
        "mean_personalized_notifications_per_day": float(d.personalized_notifications_per_day.mean()),
        "median_population_notifications_per_day": float(d.population_notifications_per_day.median()),
        "median_personalized_notifications_per_day": float(d.personalized_notifications_per_day.median()),
        "mean_recall_delta_pp": float(d.recall_delta_pp.mean()),
        "mean_notifications_delta_per_day": float(d.notifications_delta_per_day.mean()),
        "median_personalized_warning_minutes": float(d.personalized_warning_minutes.dropna().median()),
        "patients_meeting_full_product_target_after_personalization": int(full.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("models/v11_personalized_calibration"), type=Path)
    args = ap.parse_args()

    data = add_context(load_ohio_directory(args.data_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary(data).to_csv(args.output_dir / "dataset_summary.csv", index=False)

    folds = []
    for pid in sorted(int(x) for x in data.p_id.unique()):
        print(f"V11 held-out patient {pid}: 14-day calibration -> future evaluation", flush=True)
        folds.append(train_fold(data, pid, args.output_dir))
        pd.DataFrame(folds).to_csv(args.output_dir / "v11_per_patient.csv", index=False)

    report = {
        "model": "hypo_v11_personalized_calibration",
        "seed": SEED,
        "features": FEATURES,
        "population_model": "contextual LSTM trained without held-out patient",
        "personalization": "14-day patient-specific alert-policy calibration; no LSTM fine-tuning",
        "calibration_days": CALIBRATION_DAYS,
        "minimum_calibration_hypoglycemia_events": MIN_CALIBRATION_EVENTS,
        "prediction_horizon_minutes": 30,
        "protocol": "patient-disjoint population model; first 14 chronological days of held-out patient used only for personal calibration; all later patient data reserved for evaluation",
        "research_target": {"event_recall_min": 0.90, "notifications_per_patient_day_max": 1.0, "median_warning_minutes_min": 15},
        "folds": folds,
        "aggregate": aggregate(folds),
        "clinical_status": "research only; not clinically validated",
    }
    (args.output_dir / "v11_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
