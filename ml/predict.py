"""
Inference script called by the Zyntra API.
Reads a JSON object from stdin: {"z_glucose": x, "z_hrv": x, ...}
Writes a JSON object to stdout: {"score": x, "confidence": x, "model_version": "..."}
Missing features are filled with 0.0 (neutral z-score).
"""
import json
import sys
import joblib
import numpy as np
from pathlib import Path

FEATURES = ["z_glucose", "z_hrv", "z_sleep_score", "z_steps", "z_adherence"]
MODEL_PATH = Path("ml/model.pkl")

def main():
    if not MODEL_PATH.exists():
        result = {
            "score": None,
            "error": "Model not trained. Run: npm run ml:gen && npm run ml:train",
            "fallback": True,
        }
        print(json.dumps(result))
        sys.exit(0)

    model = joblib.load(MODEL_PATH)
    input_data = json.loads(sys.stdin.read())

    # Build feature vector — missing = 0.0 (neutral)
    available = [k for k in FEATURES if input_data.get(k) is not None]
    confidence = len(available) / len(FEATURES)

    x = np.array([[input_data.get(f, 0.0) for f in FEATURES]])
    score = float(model.predict_proba(x)[0, 1])

    # Load metadata
    meta_path = Path("ml/model_meta.json")
    model_version = "unknown"
    if meta_path.exists():
        with open(meta_path) as mf:
            model_version = json.load(mf).get("version", "unknown")

    print(json.dumps({
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "model_version": model_version,
        "features_used": available,
        "fallback": False,
    }))

if __name__ == "__main__":
    main()
