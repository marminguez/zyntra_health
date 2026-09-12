"""Matched clinical comparison: frozen V14.1 vs V14.4 residual forecaster.
No fitting or optimization is performed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

HORIZONS = (30, 60, 90, 120)
SLICES = (
    "all", "hypoglycemia_<70", "target_70_180", "hyperglycemia_>180",
    "severe_hyper_>250", "rapid_drop_>=30mgdl", "rapid_rise_>=30mgdl",
)

def load_validation(root: Path):
    xs, ys = [], []
    for p in sorted((root / "validation").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No validation shards found")
    return np.concatenate(xs), np.concatenate(ys)

def dclass(d):
    return np.where(d > 5, 1, np.where(d < -5, -1, 0))

def calc(yt, yp, cur, mask):
    yt, yp, cur = yt[mask], yp[mask], cur[mask]
    err = yp - yt
    return {
        "n": int(len(yt)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mard": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
        "direction_accuracy": float(np.mean(dclass(yt-cur) == dclass(yp-cur))),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v14_1")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--v14-4-dir", default="ml/results/v14_4")
    ap.add_argument("--outdir", default="ml/results/v14_4_clinical")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    data, d1, d4, out = map(Path, [a.data_dir, a.v14_1_dir, a.v14_4_dir, a.outdir])
    out.mkdir(parents=True, exist_ok=True)
    x, y = load_validation(data)
    cur = x[:, -1, 0].astype(np.float64)
    with np.load(d1 / "normalization.npz") as z:
        mean, std = z["mean"], z["std"]
    xn = ((x - mean) / std).astype(np.float32)
    m1 = tf.keras.models.load_model(d1 / "best_model.keras")
    raw1 = m1.predict(xn, batch_size=a.batch_size, verbose=1)
    p1 = np.column_stack([v.reshape(-1) for v in raw1]).astype(np.float64)
    with np.load(d4 / "validation_predictions.npz") as z:
        p4 = z["y_pred"].astype(np.float64)
        y4 = z["y_true"].astype(np.float64)
    if p4.shape != p1.shape or y4.shape != y.shape or not np.allclose(y4, y, atol=1e-5):
        raise ValueError("V14.4 predictions do not align with validation windows")
    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y[:, i].astype(np.float64)
        delta = yt - cur
        masks = {
            "all": np.ones(len(yt), dtype=bool),
            "hypoglycemia_<70": yt < 70,
            "target_70_180": (yt >= 70) & (yt <= 180),
            "hyperglycemia_>180": yt > 180,
            "severe_hyper_>250": yt > 250,
            "rapid_drop_>=30mgdl": delta <= -30,
            "rapid_rise_>=30mgdl": delta >= 30,
        }
        for model_name, pred in (("v14_1", p1[:, i]), ("v14_4", p4[:, i])):
            for s in SLICES:
                rows.append({"model": model_name, "horizon_minutes": h, "slice": s, **calc(yt, pred, cur, masks[s])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "v14_4_vs_v14_1_clinical_metrics.csv", index=False)
    comp = []
    for h in HORIZONS:
        for s in SLICES:
            r1 = df[(df.model == "v14_1") & (df.horizon_minutes == h) & (df.slice == s)].iloc[0]
            r4 = df[(df.model == "v14_4") & (df.horizon_minutes == h) & (df.slice == s)].iloc[0]
            comp.append({
                "horizon_minutes": h, "slice": s, "n": int(r1.n),
                "v14_1_rmse": r1.rmse, "v14_4_rmse": r4.rmse,
                "rmse_change_pct_v14_4_vs_v14_1": (r4.rmse-r1.rmse)/r1.rmse*100,
                "v14_1_mard": r1.mard, "v14_4_mard": r4.mard,
                "mard_change_pct_v14_4_vs_v14_1": (r4.mard-r1.mard)/r1.mard*100,
                "v14_1_direction": r1.direction_accuracy, "v14_4_direction": r4.direction_accuracy,
                "direction_change_pp": (r4.direction_accuracy-r1.direction_accuracy)*100,
                "v14_4_better_rmse": bool(r4.rmse < r1.rmse),
            })
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "v14_4_vs_v14_1_clinical_comparison.csv", index=False)
    (out / "report.json").write_text(json.dumps({
        "version": "v14.4_matched_clinical_comparison", "validation_windows": int(len(x)),
        "models": ["v14_1", "v14_4"], "matched": True, "test_parquet_used": False,
        "optimization_performed": False,
    }, indent=2), encoding="utf-8")
    print("\nV14.4 vs V14.1 — CLINICAL COMPARISON")
    print(comp.to_string(index=False))
    print(f"\nArtifacts written to {out}")

if __name__ == "__main__":
    main()
