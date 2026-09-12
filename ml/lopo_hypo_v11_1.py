"""Hypo V11.1: patient-specific calibration window study.

Compares 7, 14, 21 and 30 days of personal calibration on the SAME future
period (day 30 onward) for each held-out patient. The contextual population
LSTM is trained without the held-out patient and is never fine-tuned here.
Only the alert policy is personalized. This isolates how much personal history
is needed before policy calibration becomes stable.
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
WINDOWS = (7, 14, 21, 30)
COMMON_EVAL_START_DAY = 30
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
        state = "NORMAL"
        high_streak = 0
        clear_streak = 0
        clear_threshold = max(0.0, float(threshold) - float(rearm_margin))
        for pos in list(idxs):
            p = float(work.at[pos, "probability"])
            if state == "NORMAL":
                if p >= threshold:
                    high_streak = 1
                    if persistence == 1:
                        state = "ALERTED"
                        model_alert[pos] = 1.0
                        notification[pos] = 1.0
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
    notifications = int(m.notify.sum())
    days = 0.0
    for _, f in m.groupby("p_id"):
        days += max((f.timestamp.max() - f.timestamp.min()).total_seconds() / 86400.0, 5.0 / 1440.0)
    return {"notifications": notifications, "notifications_per_patient_day": notifications / days if days else None, "observed_patient_days": days}


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
    return {
        "threshold": float(row.threshold),
        "persistence": int(row.persistence),
        "clear_steps": int(row.clear_steps),
        "rearm_margin": float(row.rearm_margin),
    }, tag


def count_hypo_events(meta):
    if meta.empty:
        return 0
    m = meta.sort_values("timestamp").copy()
    low = m.glucose < 70
    return int((low & ~low.shift(1, fill_value=False)).sum())


def subset_by_time(meta, probs, start=None, end=None):
    ts = pd.to_datetime(meta.timestamp)
    mask = pd.Series(True, index=meta.index)
    if start is not None:
        mask &= ts >= start
    if end is not None:
        mask &= ts < end
    idx = np.flatnonzero(mask.to_numpy())
    return meta.iloc[idx].reset_index(drop=True), np.asarray(probs)[idx]


def train_patient(data, test_id, outdir):
    seed_all()
    others = [int(x) for x in sorted(data.p_id.unique()) if int(x) != int(test_id)]
    val_id = external_validation_patient(others, test_id)
    train_ids = [x for x in others if x != val_id]
    train = data[data.p_id.isin(train_ids)]
    external_val = data[data.p_id == val_id]
    target = data[data.p_id == test_id].sort_index()

    scaler = StandardScaler().fit(train[FEATURES].fillna(0.0))
    xtr, ytr, _ = sequences(train, scaler)
    xv, yv, mv = sequences(external_val, scaler)
    xt, yt, mt = sequences(target, scaler)

    classes = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_hypoglycemia_classifier(LOOKBACK, len(FEATURES))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit(xtr, ytr, validation_data=(xv, yv), class_weight=class_weight,
              epochs=35, batch_size=128, shuffle=False,
              callbacks=[EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True, mode="max")], verbose=0)

    p_val = model.predict(xv, verbose=0).ravel()
    p_target = model.predict(xt, verbose=0).ravel()
    population_grid = policy_grid(mv, p_val)
    population_policy, population_selection = choose_policy(population_grid)

    target_start = pd.to_datetime(mt.timestamp).min()
    common_eval_start = target_start + pd.Timedelta(days=COMMON_EVAL_START_DAY)
    m_eval, p_eval = subset_by_time(mt, p_target, start=common_eval_start)
    y_eval_idx = pd.to_datetime(mt.timestamp) >= common_eval_start
    y_eval = yt[np.flatnonzero(y_eval_idx.to_numpy())]
    if len(m_eval) == 0:
        raise RuntimeError(f"Patient {test_id} has no common evaluation data after day {COMMON_EVAL_START_DAY}")

    ev_pop, nt_pop = evaluate_policy(m_eval, p_eval, population_policy)
    auc_eval = float(roc_auc_score(y_eval, p_eval)) if len(np.unique(y_eval)) > 1 else None

    patient_dir = outdir / f"patient_{test_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)
    population_grid.to_csv(patient_dir / "population_policy_grid.csv", index=False)

    rows = []
    for days in WINDOWS:
        cutoff = target_start + pd.Timedelta(days=days)
        m_cal, p_cal = subset_by_time(mt, p_target, end=cutoff)
        calibration_events = count_hypo_events(m_cal)
        if len(m_cal) == 0:
            raise RuntimeError(f"Patient {test_id} has no calibration samples for {days} days")
        grid = policy_grid(m_cal, p_cal)
        grid.to_csv(patient_dir / f"personal_calibration_grid_{days}d.csv", index=False)

        if calibration_events >= MIN_CALIBRATION_EVENTS:
            personal_policy, selection = choose_policy(grid)
            status = "personalized"
        else:
            personal_policy = population_policy.copy()
            selection = "fallback_population_insufficient_events"
            status = "insufficient_calibration_events"

        ev_per, nt_per = evaluate_policy(m_eval, p_eval, personal_policy)
        rows.append({
            "test_patient": int(test_id),
            "calibration_days": int(days),
            "population_validation_patient": int(val_id),
            "population_train_patients": train_ids,
            "calibration_hypoglycemia_events": int(calibration_events),
            "personalization_status": status,
            "personalized_selection": selection,
            "personalized_policy": personal_policy,
            "population_policy": population_policy,
            "population_selection": population_selection,
            "common_evaluation_start": str(common_eval_start),
            "evaluation_hypoglycemia_events": int(ev_per["hypoglycemia_events"]),
            "population_event_recall": ev_pop["event_recall"],
            "population_notifications_per_day": nt_pop["notifications_per_patient_day"],
            "population_warning_minutes": ev_pop["median_warning_minutes"],
            "personalized_event_recall": ev_per["event_recall"],
            "personalized_notifications_per_day": nt_per["notifications_per_patient_day"],
            "personalized_warning_minutes": ev_per["median_warning_minutes"],
            "personalized_detected_events": int(ev_per["detected_events"]),
            "population_detected_events": int(ev_pop["detected_events"]),
            "recall_delta_pp": (ev_per["event_recall"] - ev_pop["event_recall"]) * 100.0 if ev_per["event_recall"] is not None and ev_pop["event_recall"] is not None else None,
            "notifications_delta_per_day": nt_per["notifications_per_patient_day"] - nt_pop["notifications_per_patient_day"],
            "evaluation_observed_days": ev_per["observed_patient_days"],
            "roc_auc_common_evaluation": auc_eval,
        })
    return rows


def summarize(rows):
    d = pd.DataFrame(rows)
    summaries = []
    for days, g in d.groupby("calibration_days"):
        events = int(g.evaluation_hypoglycemia_events.sum())
        pop_detected = int(g.population_detected_events.sum())
        per_detected = int(g.personalized_detected_events.sum())
        full = (g.personalized_event_recall >= 0.90) & (g.personalized_notifications_per_day <= 1.0) & (g.personalized_warning_minutes >= 15.0)
        summaries.append({
            "calibration_days": int(days),
            "patients": int(len(g)),
            "patients_personalized": int((g.personalization_status == "personalized").sum()),
            "mean_calibration_hypoglycemia_events": float(g.calibration_hypoglycemia_events.mean()),
            "common_evaluation_events": events,
            "population_pooled_event_recall": pop_detected / events if events else None,
            "personalized_pooled_event_recall": per_detected / events if events else None,
            "mean_population_notifications_per_day": float(g.population_notifications_per_day.mean()),
            "mean_personalized_notifications_per_day": float(g.personalized_notifications_per_day.mean()),
            "median_personalized_notifications_per_day": float(g.personalized_notifications_per_day.median()),
            "mean_recall_delta_pp": float(g.recall_delta_pp.mean()),
            "mean_notifications_delta_per_day": float(g.notifications_delta_per_day.mean()),
            "median_personalized_warning_minutes": float(g.personalized_warning_minutes.dropna().median()),
            "patients_meeting_full_target": int(full.sum()),
        })
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("models/v11_1_calibration_window_study"), type=Path)
    args = ap.parse_args()

    data = add_context(load_ohio_directory(args.data_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary(data).to_csv(args.output_dir / "dataset_summary.csv", index=False)

    rows = []
    for pid in sorted(int(x) for x in data.p_id.unique()):
        print(f"V11.1 held-out patient {pid}: testing 7/14/21/30-day calibration", flush=True)
        rows.extend(train_patient(data, pid, args.output_dir))
        pd.DataFrame(rows).to_csv(args.output_dir / "v11_1_per_patient_window.csv", index=False)

    summary = summarize(rows)
    pd.DataFrame(summary).to_csv(args.output_dir / "v11_1_window_summary.csv", index=False)
    report = {
        "model": "hypo_v11_1_calibration_window_study",
        "seed": SEED,
        "features": FEATURES,
        "calibration_windows_days": list(WINDOWS),
        "common_evaluation_start_day": COMMON_EVAL_START_DAY,
        "minimum_calibration_hypoglycemia_events": MIN_CALIBRATION_EVENTS,
        "prediction_horizon_minutes": 30,
        "protocol": "patient-disjoint population LSTM; personalize alert policy with first 7/14/21/30 days; compare every window on the same held-out patient period beginning day 30",
        "research_target": {"event_recall_min": 0.90, "notifications_per_patient_day_max": 1.0, "median_warning_minutes_min": 15},
        "window_summary": summary,
        "rows": rows,
        "clinical_status": "research only; not clinically validated",
    }
    (args.output_dir / "v11_1_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
