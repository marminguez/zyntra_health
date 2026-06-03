"""
Hypoglycemia LSTM classifier utilities for Zyntra.

The bundled ``lstm_hypoglycemia_classifier.h5`` model predicts the probability
of hypoglycemia (<70 mg/dL) 30 minutes ahead from a 48-step, 5-minute sequence.

Expected sequence shape: (48, 5)
Feature order: glucose, bolus, carbs_g, step_count, iob

This module intentionally lazy-loads optional scientific dependencies so the
Next.js app and the regular TypeScript test suite can run without TensorFlow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

FEATURES = ["glucose", "bolus", "carbs_g", "step_count", "iob"]
LOOKBACK_STEPS = 48
PREDICTION_HORIZON_MINUTES = 30
DEFAULT_THRESHOLD = 0.5
BASELINE_ACCURACY = 0.72
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "lstm_hypoglycemia_classifier.h5"
DEFAULT_METADATA_PATH = SCRIPT_DIR / "lstm_hypoglycemia_classifier_meta.json"
DEFAULT_SCALER_PATH = SCRIPT_DIR / "lstm_hypoglycemia_classifier_scaler.joblib"


def _load_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "numpy is required for LSTM prediction. Install the ML dependencies "
            "before running this script."
        ) from exc
    return np


def _load_tensorflow_model(model_path: Path):
    try:
        from tensorflow.keras.models import load_model  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "tensorflow is required to load the LSTM .h5 model. Install the ML "
            "dependencies before running LSTM inference."
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model not found at {model_path}")

    # compile=False avoids needing to deserialize training-only optimizer state.
    return load_model(model_path, compile=False)


def _load_joblib_scaler(scaler_path: Path | None):
    if scaler_path is None:
        return None

    try:
        import joblib  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("joblib is required when --scaler is provided.") from exc

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    return joblib.load(scaler_path)


def _as_sequence(payload: dict[str, Any]):
    """Return a numpy array with shape (1, 48, 5) from JSON payload."""
    np = _load_numpy()

    sequence = payload.get("sequence")
    if sequence is None:
        raise ValueError(
            "Missing 'sequence'. Provide a 48-row array in feature order: "
            f"{', '.join(FEATURES)}."
        )

    arr = np.asarray(sequence, dtype="float32")
    if arr.shape != (LOOKBACK_STEPS, len(FEATURES)):
        raise ValueError(
            f"Invalid sequence shape {arr.shape}; expected "
            f"({LOOKBACK_STEPS}, {len(FEATURES)})."
        )

    return arr.reshape(1, LOOKBACK_STEPS, len(FEATURES))


def _maybe_scale_sequence(sequence, scaler: Any | None):
    if scaler is None:
        return sequence

    original_shape = sequence.shape
    flat = sequence.reshape(-1, len(FEATURES))
    scaled = scaler.transform(flat)
    return scaled.reshape(original_shape)


def predict_probability(
    sequence: Iterable[Iterable[float]],
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path | None = None,
) -> float:
    """
    Predict hypoglycemia probability from one 48x5 sequence.

    By default the sequence is assumed to already be scaled exactly as the model
    was trained. Pass ``scaler_path`` to transform raw feature values first.
    """
    np = _load_numpy()
    payload = {"sequence": sequence}
    x = _as_sequence(payload)
    x = _maybe_scale_sequence(x, _load_joblib_scaler(scaler_path))
    model = _load_tensorflow_model(model_path)
    probability = model.predict(x, verbose=0).flatten()[0]
    return float(np.clip(probability, 0.0, 1.0))


def predict_from_payload(
    payload: dict[str, Any],
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Predict from an API-friendly JSON payload."""
    x = _as_sequence(payload)
    x = _maybe_scale_sequence(x, _load_joblib_scaler(scaler_path))
    model = _load_tensorflow_model(model_path)
    probability = float(model.predict(x, verbose=0).flatten()[0])
    probability = max(0.0, min(1.0, probability))

    return {
        "p_hypo_30m": round(probability, 6),
        "will_hypo_30m": probability >= threshold,
        "threshold": threshold,
        "horizon_minutes": PREDICTION_HORIZON_MINUTES,
        "lookback_steps": LOOKBACK_STEPS,
        "feature_order": FEATURES,
        "model_path": str(model_path),
        "fallback": False,
    }


def write_metadata(path: Path = DEFAULT_METADATA_PATH) -> None:
    """Write lightweight metadata next to the H5 model for app integration."""
    path.write_text(
        json.dumps(
            {
                "model_type": "bidirectional_lstm_binary_classifier",
                "task": "hypoglycemia_30m",
                "features": FEATURES,
                "lookback_steps": LOOKBACK_STEPS,
                "sampling_minutes": 5,
                "horizon_minutes": PREDICTION_HORIZON_MINUTES,
                "threshold": DEFAULT_THRESHOLD,
                "baseline_accuracy": BASELINE_ACCURACY,
                "baseline_accuracy_note": "Original holdout accuracy reported before moving the model asset; verify with the same held-out dataset before retraining or replacing the H5 file.",
                "model_file": DEFAULT_MODEL_PATH.name,
                "scaler_file": DEFAULT_SCALER_PATH.name,
                "dataset_required_for_inference": False,
                "dataset_required_to_reproduce_metrics": True,
                "validation_dataset_format": {
                    "sequence_key": "sequences",
                    "label_key": "labels",
                    "sequence_shape": [LOOKBACK_STEPS, len(FEATURES)],
                },
                "input_shape": [LOOKBACK_STEPS, len(FEATURES)],
            },
            indent=2,
        )
        + "\n"
    )


def run_predict_cli(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text() if args.input else os.sys.stdin.read())
    try:
        result = predict_from_payload(
            payload,
            model_path=Path(args.model),
            scaler_path=Path(args.scaler) if args.scaler else None,
            threshold=args.threshold,
        )
    except Exception as exc:  # CLI must return JSON so callers can fall back.
        result = {"fallback": True, "error": str(exc)}
        print(json.dumps(result))
        return 0

    print(json.dumps(result))
    return 0


def evaluate_json_dataset(
    dataset_path: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate the H5 model against a held-out JSON dataset.

    Expected JSON format::

        {"sequences": [[[... 5 values ...], ... 48 rows ...]], "labels": [0, 1]}

    Keep this dataset stable if the goal is to confirm the original ~72%
    accuracy after moving files, changing wrappers or replacing the model.
    """
    np = _load_numpy()
    payload = json.loads(dataset_path.read_text())
    sequences = np.asarray(payload.get("sequences"), dtype="float32")
    labels = np.asarray(payload.get("labels"), dtype="int32")

    if sequences.ndim != 3 or sequences.shape[1:] != (LOOKBACK_STEPS, len(FEATURES)):
        raise ValueError(
            f"Invalid sequences shape {sequences.shape}; expected "
            f"(n, {LOOKBACK_STEPS}, {len(FEATURES)})."
        )
    if labels.ndim != 1 or len(labels) != len(sequences):
        raise ValueError(
            f"Invalid labels shape {labels.shape}; expected one label per sequence."
        )

    x = _maybe_scale_sequence(sequences, _load_joblib_scaler(scaler_path))
    model = _load_tensorflow_model(model_path)
    probabilities = model.predict(x, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype("int32")
    accuracy = float((predictions == labels).mean())

    positives = labels == 1
    negatives = labels == 0
    true_positives = int(((predictions == 1) & positives).sum())
    false_positives = int(((predictions == 1) & negatives).sum())
    false_negatives = int(((predictions == 0) & positives).sum())
    true_negatives = int(((predictions == 0) & negatives).sum())

    return {
        "accuracy": round(accuracy, 6),
        "baseline_accuracy": BASELINE_ACCURACY,
        "within_baseline": accuracy >= BASELINE_ACCURACY,
        "threshold": threshold,
        "samples": int(len(labels)),
        "confusion_matrix": {
            "tn": true_negatives,
            "fp": false_positives,
            "fn": false_negatives,
            "tp": true_positives,
        },
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "fallback": False,
    }


def run_evaluate_cli(args: argparse.Namespace) -> int:
    try:
        result = evaluate_json_dataset(
            Path(args.dataset),
            model_path=Path(args.model),
            scaler_path=Path(args.scaler) if args.scaler else None,
            threshold=args.threshold,
        )
    except Exception as exc:  # CLI must return JSON so CI output is machine-readable.
        result = {"fallback": True, "error": str(exc)}
        print(json.dumps(result))
        return 0

    print(json.dumps(result))
    return 0


def build_hypoglycemia_classifier(lookback: int = LOOKBACK_STEPS, n_features: int = len(FEATURES)):
    """Build the LSTM architecture used by the bundled H5 classifier."""
    try:
        from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout  # type: ignore
        from tensorflow.keras.models import Sequential  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("tensorflow is required to build/train the LSTM model.") from exc

    return Sequential(
        [
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, n_features)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )


def train_from_uom_export(
    base_path: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
) -> None:
    """
    Train the classifier from a University of Manchester-style export folder.

    This preserves the original research workflow but keeps heavy dependencies
    local to training. It writes the .h5 model plus a metadata JSON file. A scaler
    should be persisted alongside the model before using raw-value inference.
    """
    try:
        import glob
        import pandas as pd  # type: ignore
        import tensorflow as tf  # type: ignore
        from sklearn.metrics import fbeta_score  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.utils.class_weight import compute_class_weight  # type: ignore
        import joblib  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Training requires pandas, scikit-learn and tensorflow. Install the "
            "ML dependencies before running train."
        ) from exc

    np = _load_numpy()
    id_pattern = __import__("re").compile(r"UoMGlucose(\d+)\.csv")
    all_patients_data = []

    glucose_files = glob.glob(str(base_path / "Glucose Data" / "UoMGlucose*.csv"))
    for file_path in glucose_files:
        match = id_pattern.search(Path(file_path).name)
        if not match:
            continue

        patient_id = int(match.group(1))
        try:
            df_bg = pd.read_csv(base_path / f"Glucose Data/UoMGlucose{patient_id}.csv")
            df_bolus = pd.read_csv(base_path / f"Insulin Data/Bolus Data/UoMBolus{patient_id}.csv")
            df_meals = pd.read_csv(base_path / f"Nutrition Data/UoMNutrition{patient_id}.csv")
            df_activity = pd.read_csv(base_path / f"Activity Data/UoMActivity{patient_id}.csv")

            def to_5min_grid(df, col):
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce").dt.tz_localize(None)
                df = df.dropna(subset=[col]).copy()
                df["ts_grid"] = df[col].dt.floor("5min")
                return df

            df_bg = to_5min_grid(df_bg, "bg_ts")
            df_bolus = to_5min_grid(df_bolus, "bolus_ts")
            df_meals = to_5min_grid(df_meals, "meal_ts")
            df_activity = to_5min_grid(df_activity, "activity_ts")

            master_index = pd.date_range(df_bg["ts_grid"].min(), df_bg["ts_grid"].max(), freq="5min")
            df_p = pd.DataFrame(index=master_index)
            df_bg_clean = df_bg.drop_duplicates("ts_grid").set_index("ts_grid")
            df_p["glucose"] = df_bg_clean["value"].reindex(df_p.index).interpolate(method="time") * 18
            df_p["bolus"] = df_bolus.groupby("ts_grid")["bolus_dose"].sum().reindex(df_p.index).fillna(0)
            df_p = df_p.join(df_meals.groupby("ts_grid")[["carbs_g"]].sum()).fillna(0)
            df_p = df_p.join(df_activity.groupby("ts_grid")[["step_count"]].sum()).fillna(0)
            df_p["iob"] = df_p["bolus"].rolling(window=48, min_periods=1).sum()
            df_p["p_id"] = patient_id
            df_p["glucose_future"] = df_p["glucose"].shift(-6)
            df_p["target_hypo"] = (df_p["glucose_future"] < 70).astype(int)
            all_patients_data.append(df_p.dropna())
        except Exception as exc:
            print(f"Skipping patient {patient_id}: {exc}")

    if not all_patients_data:
        raise RuntimeError(f"No compatible patient exports found under {base_path}")

    df_all = pd.concat(all_patients_data)
    scaler = StandardScaler()
    df_all[FEATURES] = scaler.fit_transform(df_all[FEATURES])

    def create_sequences(df):
        x_data = df[FEATURES].values
        y_data = df["target_hypo"].values
        x_seq, y_seq = [], []
        for i in range(LOOKBACK_STEPS, len(df)):
            x_seq.append(x_data[i - LOOKBACK_STEPS:i])
            y_seq.append(y_data[i])
        return np.array(x_seq), np.array(y_seq)

    patient_ids = df_all["p_id"].unique()
    train_ids = patient_ids[: int(len(patient_ids) * 0.8)]
    test_ids = patient_ids[int(len(patient_ids) * 0.8):]
    x_train = np.concatenate([create_sequences(df_all[df_all["p_id"] == pid])[0] for pid in train_ids])
    y_train = np.concatenate([create_sequences(df_all[df_all["p_id"] == pid])[1] for pid in train_ids])
    x_val = np.concatenate([create_sequences(df_all[df_all["p_id"] == pid])[0] for pid in test_ids])
    y_val = np.concatenate([create_sequences(df_all[df_all["p_id"] == pid])[1] for pid in test_ids])

    model = build_hypoglycemia_classifier(LOOKBACK_STEPS, len(FEATURES))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )
    class_weights_array = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
    model.fit(x_train, y_train, class_weight=class_weights, epochs=10, batch_size=64, validation_split=0.1)

    probabilities = model.predict(x_val, verbose=0).flatten()
    thresholds = np.arange(0.05, 0.95, 0.05)
    f_scores = [fbeta_score(y_val, (probabilities >= threshold).astype(int), beta=5.0) for threshold in thresholds]
    threshold = float(thresholds[int(np.argmax(f_scores))])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    write_metadata(model_path.with_name("lstm_hypoglycemia_classifier_meta.json"))
    print(json.dumps({"model_path": str(model_path), "scaler_path": str(scaler_path), "threshold": threshold}))


def main() -> int:
    parser = argparse.ArgumentParser(description="LSTM hypoglycemia classifier utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Read JSON and return p_hypo_30m")
    predict_parser.add_argument("--input", help="JSON file. Defaults to stdin.")
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to .h5 model")
    predict_parser.add_argument("--scaler", help="Optional joblib StandardScaler for raw input values")
    predict_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    metadata_parser = subparsers.add_parser("metadata", help="Write model metadata JSON")
    metadata_parser.add_argument("--output", default=str(DEFAULT_METADATA_PATH))

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a held-out JSON dataset")
    evaluate_parser.add_argument("--dataset", required=True, help="JSON file with sequences and labels")
    evaluate_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to .h5 model")
    evaluate_parser.add_argument("--scaler", help="Optional joblib StandardScaler for raw input values")
    evaluate_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    train_parser = subparsers.add_parser("train", help="Train from UoM export folder")
    train_parser.add_argument("--base-path", required=True)
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    train_parser.add_argument("--scaler", default=str(DEFAULT_SCALER_PATH))

    args = parser.parse_args()
    if args.command == "predict":
        return run_predict_cli(args)
    if args.command == "metadata":
        write_metadata(Path(args.output))
        return 0
    if args.command == "evaluate":
        return run_evaluate_cli(args)
    if args.command == "train":
        train_from_uom_export(Path(args.base_path), model_path=Path(args.model), scaler_path=Path(args.scaler))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
