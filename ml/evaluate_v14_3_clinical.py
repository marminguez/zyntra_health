"""Matched clinical comparison of frozen V14.1 vs V14.3.

Runs both frozen models on the same validation windows and reports global and
clinical/dynamic slices. No fitting or optimization is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

HORIZONS = (30, 60, 90, 120)


def load_validation(root: Path):
    xs, ys = [], []
    for path in sorted((root / "validation").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No validation shards found")
    return np.concatenate(xs), np.concatenate(ys)


def dclass(delta):
    return np.where(delta > 5, 1, np.where(delta < -5, -1, 0))


def metrics(yt, yp, cur, mask):
    yt, yp, cur = yt[mask], yp[mask], cur[mask]
    if len(yt) == 0:
        return None
    e = yp - yt
    return {
        "n": int(len(yt)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mard": float(np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-6)) * 100),
        "direction_accuracy": float(np.mean(dclass(yt-cur) == dclass(yp-cur))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v14_1")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--v14-3-dir", default="ml/results/v14_3")
    ap.add_argument("--outdir", default="ml/results/v14_3_clinical")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    data, d1, d3, out = map(Path, [args.data_dir, args.v14_1_dir, args.v14_3_dir, args.outdir])
    out.mkdir(parents=True, exist_ok=True)
    x, y = load_validation(data)
    cur = x[:, -1, 0].astype(np.float64)

    with np.load(d1 / "normalization.npz") as z:
        mean, std = z["mean"], z["std"]
    xn = ((x - mean) / std).astype(np.float32)

    m1 = tf.keras.models.load_model(d1 / "best_model.keras")
    p1raw = m1.predict(xn, batch_size=args.batch_size, verbose=1)
    p1 = np.column_stack([a.reshape(-1) for a in p1raw])

    with np.load(d3 / "validation_predictions.npz") as z:
        p3 = z["y_pred"].astype(np.float64)
        y3 = z["y_true"].astype(np.float64)
    if p3.shape != p1.shape or y3.shape != y.shape or not np.allclose(y3, y, atol=1e-5):
        raise ValueError("V14.3 saved predictions do not align with current validation windows")

    rows = []
    slices_order = ["all", "hypoglycemia_<70", "target_70_180", "hyperglycemia_>180", "severe_hyper_>250", "rapid_drop_>=30mgdl", "rapid_rise_>=30mgdl"]
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
        for model_name, pred in (("v14_1", p1[:, i]), ("v14_3", p3[:, i])):
            for s in slices_order:
                r = metrics(yt, pred, cur, masks[s])
                if r:
                    rows.append({"model": model_name, "horizon_minutes": h, "slice": s, **r})

    df = pd.DataFrame(rows)
    df.to_csv(out / "v14_3_vs_v14_1_clinical_metrics.csv", index=False)

    comp = []
    for h in HORIZONS:
        for s in slices_order:
            a = df[(df.model == "v14_1") & (df.horizon_minutes == h) & (df.slice == s)].iloc[0]
            b = df[(df.model == "v14_3") & (df.horizon_minutes == h) & (df.slice == s)].iloc[0]
            comp.append({
                "horizon_minutes": h,
                "slice": s,
                "n": int(a.n),
                "v14_1_rmse": a.rmse,
                "v14_3_rmse": b.rmse,
                "rmse_change_pct_v14_3_vs_v14_1": (b.rmse-a.rmse)/a.rmse*100,
                "v14_1_mard": a.mard,
                "v14_3_mard": b.mard,
                "mard_change_pct_v14_3_vs_v14_1": (b.mard-a.mard)/a.mard*100,
                "v14_1_direction": a.direction_accuracy,
                "v14_3_direction": b.direction_accuracy,
                "direction_change_pp": (b.direction_accuracy-a.direction_accuracy)*100,
                "v14_3_better_rmse": bool(b.rmse < a.rmse),
            })
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "v14_3_vs_v14_1_clinical_comparison.csv", index=False)

    report = {
        "version": "v14.3_matched_clinical_comparison",
        "validation_windows": int(len(x)),
        "models": ["v14_1", "v14_3"],
        "matched": True,
        "test_parquet_used": False,
        "optimization_performed": False,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nV14.3 vs V14.1 — CLINICAL COMPARISON")
    print(comp.to_string(index=False))
    print(f"\nArtifacts written to {out}")

if __name__ == "__main__":
    main()
