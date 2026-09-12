"""Zyntra V14.2: matched validation and clinical stress test.

Evaluates the frozen V14.1 model, Persistence and Linear Trend on exactly the
same validation windows. No fitting or threshold optimization occurs here.
Also reports clinically relevant target ranges and rapid glucose dynamics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from forecasting.model_v14 import HORIZONS


def load_validation(root: Path):
    xs, ys, subjects = [], [], []
    for path in sorted((root / "validation").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            x = z["x"].astype(np.float32)
            y = z["y"].astype(np.float32)
        xs.append(x); ys.append(y)
        subjects.extend([path.stem] * len(x))
    if not xs:
        raise ValueError("No validation shards found")
    return np.concatenate(xs), np.concatenate(ys), np.asarray(subjects)


def direction(delta):
    return np.where(delta > 5, 1, np.where(delta < -5, -1, 0))


def metric_row(model, horizon, yt, yp, current, mask, slice_name):
    yt, yp, current = yt[mask], yp[mask], current[mask]
    if len(yt) == 0:
        return None
    err = yp - yt
    return {
        "model": model, "horizon_minutes": horizon, "slice": slice_name, "n": int(len(yt)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mard": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
        "direction_accuracy": float(np.mean(direction(yt-current) == direction(yp-current))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="ml/data/v14_1")
    p.add_argument("--model-dir", default="ml/results/v14_1")
    p.add_argument("--outdir", default="ml/results/v14_2")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    data_dir, model_dir, outdir = Path(args.data_dir), Path(args.model_dir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x, y, subjects = load_validation(data_dir)
    with np.load(model_dir / "normalization.npz") as z:
        mean, std = z["mean"], z["std"]
    xn = ((x - mean) / std).astype(np.float32)
    model = tf.keras.models.load_model(model_dir / "best_model.keras")
    raw = model.predict(xn, batch_size=args.batch_size, verbose=1)
    zyntra = np.column_stack([a.reshape(-1) for a in raw])

    current = x[:, -1, 0].astype(np.float64)
    # delta_15m at the forecast origin is feature index 2.
    delta15 = x[:, -1, 2].astype(np.float64)
    persistence = np.repeat(current[:, None], len(HORIZONS), axis=1)
    linear = np.column_stack([
        np.clip(current + (delta15 / 15.0) * h, 40.0, 400.0) for h in HORIZONS
    ])

    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y[:, i].astype(np.float64)
        true_delta = yt - current
        masks = {
            "all": np.ones(len(yt), dtype=bool),
            "hypoglycemia_<70": yt < 70,
            "target_70_180": (yt >= 70) & (yt <= 180),
            "hyperglycemia_>180": yt > 180,
            "severe_hyper_>250": yt > 250,
            "rapid_drop_>=30mgdl": true_delta <= -30,
            "rapid_rise_>=30mgdl": true_delta >= 30,
        }
        for name, pred in (("zyntra_v14_1", zyntra[:, i]), ("persistence", persistence[:, i]), ("linear_trend", linear[:, i])):
            for slice_name, mask in masks.items():
                row = metric_row(name, h, yt, pred.astype(np.float64), current, mask, slice_name)
                if row: rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "v14_2_matched_metrics.csv", index=False)

    all_rows = metrics[metrics.slice == "all"].copy()
    pivot = all_rows.pivot(index="horizon_minutes", columns="model", values=["rmse", "mard", "direction_accuracy"])
    comparison_rows = []
    for h in HORIZONS:
        z = all_rows[(all_rows.horizon_minutes == h) & (all_rows.model == "zyntra_v14_1")].iloc[0]
        pe = all_rows[(all_rows.horizon_minutes == h) & (all_rows.model == "persistence")].iloc[0]
        li = all_rows[(all_rows.horizon_minutes == h) & (all_rows.model == "linear_trend")].iloc[0]
        comparison_rows.append({
            "horizon_minutes": h,
            "zyntra_rmse": z.rmse,
            "persistence_rmse": pe.rmse,
            "rmse_improvement_vs_persistence_pct": (pe.rmse-z.rmse)/pe.rmse*100,
            "zyntra_mard": z.mard,
            "persistence_mard": pe.mard,
            "mard_improvement_vs_persistence_pct": (pe.mard-z.mard)/pe.mard*100,
            "zyntra_direction_accuracy": z.direction_accuracy,
            "linear_direction_accuracy": li.direction_accuracy,
            "direction_gain_vs_linear_pp": (z.direction_accuracy-li.direction_accuracy)*100,
            "beats_persistence_rmse": bool(z.rmse < pe.rmse),
            "beats_persistence_mard": bool(z.mard < pe.mard),
            "beats_linear_direction": bool(z.direction_accuracy > li.direction_accuracy),
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(outdir / "v14_2_matched_comparison.csv", index=False)

    report = {
        "version": "v14.2",
        "purpose": "matched validation and clinical stress test of frozen V14.1",
        "validation_windows": int(len(x)),
        "validation_subject_shards": int(len(np.unique(subjects))),
        "models": ["zyntra_v14_1", "persistence", "linear_trend"],
        "matched_evaluation": True,
        "slices": ["all", "hypoglycemia_<70", "target_70_180", "hyperglycemia_>180", "severe_hyper_>250", "rapid_drop_>=30mgdl", "rapid_rise_>=30mgdl"],
        "test_parquet_used": False,
        "optimization_performed": False,
        "clinical_status": "retrospective research evaluation only; not for clinical decision-making",
    }
    (outdir / "v14_2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nV14.2 MATCHED COMPARISON")
    print(comparison.to_string(index=False))
    print("\nV14.2 CLINICAL STRESS TEST — ZYNTRA")
    print(metrics[metrics.model == "zyntra_v14_1"].to_string(index=False))
    print(f"\nArtifacts written to {outdir}")

if __name__ == "__main__":
    main()
