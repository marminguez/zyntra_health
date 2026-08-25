"""Leakage-safe training/evaluation pipeline for Zyntra hypoglycemia LSTM.

P0 goals:
- split raw patient/time data BEFORE fitting StandardScaler
- explicit train / validation / test sets
- patient-level split whenever >= 3 patients are available
- chronological fallback for the current single-patient demo
- threshold selection ONLY on validation
- final metrics ONLY on untouched test

This file intentionally reuses data loading/model helpers from the original pipeline
while we validate the new methodology before replacing the production trainer.
"""
from pathlib import Path
import json
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from lstm_hypoglycemia_classifier import (
    build_hypoglycemia_classifier,
    create_sequences_for_classification,
    find_optimal_threshold,
    process_full_data,
)

ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR / "models"
MODEL_PATH = MODELS_DIR / "lstm_hypoglycemia_classifier_v2.h5"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold_v2.npy"
SCALER_PATH = MODELS_DIR / "lstm_feature_scaler_v2.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "lstm_feature_names_v2.json"
EVALUATION_PATH = MODELS_DIR / "evaluation_v2.json"
LOOKBACK = 48
BETA = 5.0

EXCLUDED_COLUMNS = ["glucose_future", "target_hypo", "p_id"]


def chronological_split(df, train_fraction=0.70, val_fraction=0.15):
    """Chronological fallback for a single patient; test remains the latest 15%."""
    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def split_dataset(df):
    """Prefer patient-disjoint train/val/test; fall back to chronological splitting."""
    patient_ids = sorted(df["p_id"].unique())

    if len(patient_ids) >= 3:
        rng = np.random.default_rng(42)
        ids = np.asarray(patient_ids)
        rng.shuffle(ids)

        n = len(ids)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1

        train_ids = ids[:n_train].tolist()
        val_ids = ids[n_train:n_train + n_val].tolist()
        test_ids = ids[n_train + n_val:].tolist()

        return (
            df[df["p_id"].isin(train_ids)].copy(),
            df[df["p_id"].isin(val_ids)].copy(),
            df[df["p_id"].isin(test_ids)].copy(),
            {"strategy": "patient_disjoint", "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids},
        )

    if len(patient_ids) == 2:
        # Two patients are still insufficient for a clean 3-way patient split.
        # Keep one patient entirely untouched for test and split the other chronologically.
        train_val = df[df["p_id"] == patient_ids[0]].copy()
        test_df = df[df["p_id"] == patient_ids[1]].copy()
        train_df, val_df, _ = chronological_split(train_val, train_fraction=0.80, val_fraction=0.20)
        return train_df, val_df, test_df, {
            "strategy": "hybrid_two_patient",
            "train_ids": [patient_ids[0]],
            "val_ids": [patient_ids[0]],
            "test_ids": [patient_ids[1]],
        }

    train_df, val_df, test_df = chronological_split(df)
    return train_df, val_df, test_df, {
        "strategy": "chronological_single_patient",
        "train_ids": patient_ids,
        "val_ids": patient_ids,
        "test_ids": patient_ids,
        "warning": "Single-patient evaluation does not demonstrate generalization to unseen patients.",
    }


def fit_and_apply_scaler(train_df, val_df, test_df, feature_names):
    """Fit scaler ONLY on training rows, then transform validation and test."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_names])

    scaled = []
    for frame in (train_df, val_df, test_df):
        out = frame.copy()
        out[feature_names] = scaler.transform(frame[feature_names])
        scaled.append(out)
    return (*scaled, scaler)


def frames_to_sequences(frame):
    """Create sequences independently per patient so windows never cross patient boundaries."""
    xs, ys = [], []
    for pid in sorted(frame["p_id"].unique()):
        x, y = create_sequences_for_classification(frame[frame["p_id"] == pid], lookback=LOOKBACK)
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        return np.empty((0, LOOKBACK, 0)), np.empty((0,), dtype=int)
    return np.concatenate(xs), np.concatenate(ys)


def calculate_metrics(model, x_test, y_test, threshold):
    probabilities = model.predict(x_test, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    auc = roc_auc_score(y_test, probabilities) if len(np.unique(y_test)) > 1 else None

    return {
        "roc_auc": auc,
        "recall_sensitivity": sensitivity,
        "precision_ppv": ppv,
        "specificity": specificity,
        "npv": npv,
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_samples": int(len(y_test)),
        "test_positive_samples": int(y_test.sum()),
    }


def main():
    print("\n=== ZYNTRA LSTM V2: LEAKAGE-SAFE EVALUATION ===")
    df = process_full_data()
    feature_names = [c for c in df.columns if c not in EXCLUDED_COLUMNS]

    train_df, val_df, test_df, split_info = split_dataset(df)
    print(f"Split strategy: {split_info['strategy']}")
    if split_info.get("warning"):
        print(f"WARNING: {split_info['warning']}")

    train_df, val_df, test_df, scaler = fit_and_apply_scaler(
        train_df, val_df, test_df, feature_names
    )

    x_train, y_train = frames_to_sequences(train_df)
    x_val, y_val = frames_to_sequences(val_df)
    x_test, y_test = frames_to_sequences(test_df)

    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise RuntimeError("Not enough data to create train/validation/test sequences.")

    present_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(present_classes, weights)}

    model = build_hypoglycemia_classifier(LOOKBACK, len(feature_names))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    early_stopping = EarlyStopping(
        monitor="val_auc", patience=15, restore_best_weights=True, mode="max"
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        class_weight=class_weights,
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Threshold is selected on validation only. Test remains untouched until now.
    threshold = float(find_optimal_threshold(model, x_val, y_val, beta=BETA))
    metrics = calculate_metrics(model, x_test, y_test, threshold)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    np.save(THRESHOLD_PATH, threshold)
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)

    report = {
        "model": "lstm_hypoglycemia_classifier_v2",
        "prediction_horizon_minutes": 30,
        "lookback_minutes": LOOKBACK * 5,
        "threshold": threshold,
        "threshold_metric": f"F{BETA}",
        "split": split_info,
        "patients_total": int(df["p_id"].nunique()),
        "rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "sequences": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "metrics": metrics,
        "limitations": [
            "Current demo contains too few patients for robust external generalization.",
            "Metrics are sample/window based; event-based hypoglycemia metrics are P0 next step.",
            "IOB is still a simple rolling bolus sum and should be replaced by an insulin-action model.",
        ],
    }
    with open(EVALUATION_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Evaluation saved to {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
