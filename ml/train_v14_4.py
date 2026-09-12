"""Zyntra V14.4: residual/delta multi-horizon glucose forecasting.

Everything from V14.1 is held fixed except the prediction target. Instead of
predicting absolute future glucose, the model predicts future glucose change
relative to the current CGM value. Absolute glucose is reconstructed only for
evaluation:

    pred_glucose(t+h) = glucose(t) + predicted_delta(h)

This tests the V14.2/V14.3 diagnosis that direction is learned better than the
magnitude of excursions, especially for rapid drops/rises.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from forecasting.model_v14 import build_v14_forecaster, HORIZONS

SEED = 42


def shard_paths(root: Path, split: str):
    return sorted((root / split).glob("*.npz"))


def load_shards(paths):
    xs, ys = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No V14 shards found")
    return np.concatenate(xs), np.concatenate(ys)


def direction(delta):
    return np.where(delta > 5, 1, np.where(delta < -5, -1, 0))


def evaluate(y_true_abs, y_pred_abs, current):
    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y_true_abs[:, i].astype(np.float64)
        yp = y_pred_abs[:, i].astype(np.float64)
        cur = current.astype(np.float64)
        err = yp - yt
        rows.append({
            "horizon_minutes": h,
            "n": int(len(yt)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mard": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100.0),
            "direction_accuracy": float(np.mean(direction(yt-cur) == direction(yp-cur))),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="ml/data/v14_1")
    p.add_argument("--v14-1-dir", default="ml/results/v14_1")
    p.add_argument("--outdir", default="ml/results/v14_4")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data_dir = Path(args.data_dir)
    v14_1_dir = Path(args.v14_1_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_paths = shard_paths(data_dir, "train")
    val_paths = shard_paths(data_dir, "validation")
    x_train_raw, y_train_abs = load_shards(train_paths)
    x_val_raw, y_val_abs = load_shards(val_paths)

    current_train = x_train_raw[:, -1, 0].astype(np.float32)
    current_val = x_val_raw[:, -1, 0].astype(np.float32)
    y_train_delta = y_train_abs - current_train[:, None]
    y_val_delta = y_val_abs - current_val[:, None]

    with np.load(v14_1_dir / "normalization.npz") as z:
        mean, std = z["mean"], z["std"]
    x_train = ((x_train_raw - mean) / std).astype(np.float32)
    x_val = ((x_val_raw - mean) / std).astype(np.float32)

    print("V14.4 Residual Forecasting")
    print(f"train windows: {len(x_train):,}; validation windows: {len(x_val):,}")
    print("Targets are glucose deltas relative to current CGM")
    for i, h in enumerate(HORIZONS):
        d = y_train_delta[:, i]
        print(f"  +{h}: mean delta={d.mean():.2f}, std={d.std():.2f}, min={d.min():.1f}, max={d.max():.1f}")

    model = build_v14_forecaster(x_train.shape[1], x_train.shape[2])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(str(outdir / "best_model.keras"), monitor="val_loss", save_best_only=True),
    ]

    hist = model.fit(
        x_train,
        [y_train_delta[:, i] for i in range(len(HORIZONS))],
        validation_data=(x_val, [y_val_delta[:, i] for i in range(len(HORIZONS))]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    raw = model.predict(x_val, batch_size=args.batch_size, verbose=1)
    pred_delta = np.column_stack([a.reshape(-1) for a in raw]).astype(np.float64)
    pred_abs = current_val[:, None].astype(np.float64) + pred_delta

    metrics = evaluate(y_val_abs, pred_abs, current_val)
    metrics.to_csv(outdir / "v14_4_validation_metrics.csv", index=False)
    pd.DataFrame(hist.history).to_csv(outdir / "v14_4_training_history.csv", index=False)
    np.savez_compressed(
        outdir / "validation_predictions.npz",
        y_true=y_val_abs,
        y_pred=pred_abs,
        predicted_delta=pred_delta,
        true_delta=y_val_delta,
        current_glucose=current_val,
    )

    report = {
        "version": "v14.4",
        "hypothesis": "forecasting glucose change rather than absolute glucose improves magnitude prediction in rapid and extreme trajectories",
        "architecture": "unchanged from V14.1",
        "target": "future_glucose_minus_current_glucose",
        "reconstruction": "absolute_prediction = current_glucose + predicted_delta",
        "sample_weighting": False,
        "normalization": "reused V14.1 train-only statistics",
        "train_subjects": len(train_paths),
        "validation_subjects": len(val_paths),
        "train_windows": int(len(x_train)),
        "validation_windows": int(len(x_val)),
        "test_parquet_used": False,
        "clinical_status": "retrospective research only; not for clinical decision-making",
    }
    (outdir / "v14_4_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nV14.4 VALIDATION METRICS")
    print(metrics.to_string(index=False))
    print(f"\nArtifacts written to {outdir}")

if __name__ == "__main__":
    main()
