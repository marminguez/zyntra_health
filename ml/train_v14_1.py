"""Train and evaluate the first Zyntra V14.1 multi-horizon forecaster.

Consumes the compact per-subject NPZ shards produced by
prepare_v14_training_data.py. Normalization statistics are learned from TRAIN
only. Validation subjects are never used for fitting or normalization.
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


def shard_paths(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*.npz"))


def load_shards(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No V14.1 shards found")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def normalization_from_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Every timestep from every TRAIN window contributes; validation does not.
    mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, current_glucose: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y_true[:, i].astype(np.float64)
        yp = y_pred[:, i].astype(np.float64)
        cur = current_glucose.astype(np.float64)
        err = yp - yt
        true_delta = yt - cur
        pred_delta = yp - cur
        true_dir = np.where(true_delta > 5, 1, np.where(true_delta < -5, -1, 0))
        pred_dir = np.where(pred_delta > 5, 1, np.where(pred_delta < -5, -1, 0))
        rows.append({
            "horizon_minutes": h,
            "n": int(len(yt)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mard": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100.0),
            "direction_accuracy": float(np.mean(true_dir == pred_dir)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="ml/data/v14_1")
    p.add_argument("--outdir", default="ml/results/v14_1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data_dir, outdir = Path(args.data_dir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

    train_paths = shard_paths(data_dir, "train")
    val_paths = shard_paths(data_dir, "validation")
    print(f"Loading {len(train_paths)} train subjects and {len(val_paths)} validation subjects...")
    x_train, y_train = load_shards(train_paths)
    x_val, y_val = load_shards(val_paths)
    print(f"train windows: {len(x_train):,}; validation windows: {len(x_val):,}")
    print(f"input shape: {x_train.shape}; targets: {y_train.shape}")

    mean, std = normalization_from_train(x_train)
    np.savez(outdir / "normalization.npz", mean=mean, std=std)
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_val_raw = x_val.copy()
    x_val = ((x_val - mean) / std).astype(np.float32)

    model = build_v14_forecaster(x_train.shape[1], x_train.shape[2])
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(str(outdir / "best_model.keras"), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        x_train,
        {f"pred_{h}": y_train[:, i] for i, h in enumerate(HORIZONS)},
        validation_data=(x_val, {f"pred_{h}": y_val[:, i] for i, h in enumerate(HORIZONS)}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    raw_pred = model.predict(x_val, batch_size=args.batch_size, verbose=1)
    y_pred = np.column_stack([p.reshape(-1) for p in raw_pred])
    current_glucose = x_val_raw[:, -1, 0]
    metrics = evaluate(y_val, y_pred, current_glucose)
    metrics.to_csv(outdir / "v14_1_validation_metrics.csv", index=False)
    pd.DataFrame(history.history).to_csv(outdir / "v14_1_training_history.csv", index=False)

    baseline_rmse = {30: 27.281960, 60: 43.183844, 90: 53.261696, 120: 60.046349}
    baseline_direction = {30: 0.522874, 60: 0.491473, 90: 0.481378, 120: 0.466234}
    comparison = metrics.copy()
    comparison["persistence_rmse_reference"] = comparison.horizon_minutes.map(baseline_rmse)
    comparison["beats_persistence_rmse"] = comparison.rmse < comparison.persistence_rmse_reference
    comparison["linear_direction_reference"] = comparison.horizon_minutes.map(baseline_direction)
    comparison["beats_linear_direction"] = comparison.direction_accuracy > comparison.linear_direction_reference
    comparison.to_csv(outdir / "v14_1_baseline_comparison.csv", index=False)

    report = {
        "version": "v14.1",
        "purpose": "first learned multi-horizon continuous glucose forecaster",
        "seed": SEED,
        "train_subjects": len(train_paths),
        "validation_subjects": len(val_paths),
        "train_windows": int(len(x_train)),
        "validation_windows": int(len(x_val)),
        "features": metadata["features"],
        "horizons_minutes": list(HORIZONS),
        "normalization": "fit on train only",
        "validation": "patient-disjoint internal validation; MetaboNet train/test-overlap subjects excluded",
        "test_parquet_used": False,
        "clinical_status": "retrospective research only; not for clinical decision-making",
    }
    (outdir / "v14_1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nV14.1 VALIDATION METRICS")
    print(metrics.to_string(index=False))
    print("\nV14.1 vs V14.0 reference baselines")
    print(comparison.to_string(index=False))
    print(f"\nArtifacts written to {outdir}")

if __name__ == "__main__":
    main()
