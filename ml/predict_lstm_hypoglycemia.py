"""
Inference entrypoint for the LSTM hypoglycemia classifier.

Reads a JSON payload from stdin with a `sequence` array of timesteps:
{
  "sequence": [
    {"glucose": 142, "bolus": 0, "carbs_g": 0, "step_count": 14, "iob": 1.2},
    ... 48 total timesteps ...
  ],
  "threshold": 0.35  # optional
}

Writes JSON to stdout:
{"probability": 0.82, "alert": true, "threshold": 0.35, ...}
"""
import json
import os
import pickle
import sys
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

LOOKBACK = 48
FEATURES = ["glucose", "bolus", "carbs_g", "step_count", "iob"]
DEFAULT_THRESHOLD = 0.5
MODEL_PATH = Path(os.environ.get("HYPO_LSTM_MODEL_PATH", "ml/models/lstm_hypoglycemia_classifier.h5"))
THRESHOLD_PATH = Path(os.environ.get("HYPO_LSTM_THRESHOLD_PATH", "ml/models/optimal_threshold.npy"))
SCALER_PATH = Path(os.environ.get("HYPO_LSTM_SCALER_PATH", "ml/models/lstm_feature_scaler.pkl"))


def _error(message, fallback=True):
    print(json.dumps({"error": message, "fallback": fallback}))
    sys.exit(0)


def _load_threshold(payload):
    if isinstance(payload.get("threshold"), (int, float)):
        return float(payload["threshold"])

    if THRESHOLD_PATH.exists() and np is not None:
        try:
            return float(np.load(THRESHOLD_PATH))
        except Exception:
            pass

    return DEFAULT_THRESHOLD


def _load_scaler():
    if not SCALER_PATH.exists():
        return None
    try:
        with open(SCALER_PATH, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _build_sequence(raw_sequence):
    if not isinstance(raw_sequence, list):
        _error("`sequence` must be an array")
    if len(raw_sequence) < LOOKBACK:
        _error(f"At least {LOOKBACK} timesteps are required")

    if np is None:
        _error("NumPy is not available")

    # Use the latest 48 samples and fill missing optional values with 0.0.
    latest = raw_sequence[-LOOKBACK:]
    matrix = []
    for row in latest:
        if not isinstance(row, dict):
            _error("Each timestep must be an object")
        matrix.append([float(row.get(feature, 0.0) or 0.0) for feature in FEATURES])

    return np.asarray(matrix, dtype=np.float32)


def main():
    if not MODEL_PATH.exists():
        _error(f"LSTM hypoglycemia model not found at {MODEL_PATH}")

    try:
        # Import TensorFlow lazily so API startup does not fail in environments
        # where the runtime dependency has not been installed yet.
        if np is None:
            _error("NumPy is not available")
        from tensorflow.keras.models import load_model
    except Exception as exc:
        _error(f"TensorFlow is not available: {exc}")

    try:
        payload = json.loads(sys.stdin.read() or "{}")
        threshold = _load_threshold(payload)
        sequence = _build_sequence(payload.get("sequence"))
        scaler = _load_scaler()
        scaler_used = scaler is not None
        if scaler_used:
            sequence = scaler.transform(sequence)
        x = np.expand_dims(sequence, axis=0)

        model = load_model(MODEL_PATH)
        probability = float(model.predict(x, verbose=0).flatten()[0])
        probability = max(0.0, min(1.0, probability))

        print(json.dumps({
            "probability": round(probability, 4),
            "threshold": round(threshold, 4),
            "alert": probability >= threshold,
            "lookback": LOOKBACK,
            "features": FEATURES,
            "model_version": "lstm_hypoglycemia_classifier_v1",
            "scaler_used": scaler_used,
            "fallback": False,
        }))
    except Exception as exc:
        _error(f"LSTM hypoglycemia inference failed: {exc}")


if __name__ == "__main__":
    main()
