"""Fast Scale-10 screen for the causal trajectory forecaster.

Dataset, frozen validation, normalization, targets, future-known channels,
50/50 absolute/delta fusion and seed remain aligned with V15.4 Scale-10.
Scientific change: replace independent horizon-prefix encoders with one causal
24-step future trajectory decoder initialized from metabolic history.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.forecasting.model_trajectory import HORIZONS, READ_STEPS, build_trajectory_forecaster
from ml.train_v15_4_scale10 import (
    SEED,
    ShardSequence,
    evaluate,
    load_validation,
    normalize_future,
    train_manifest,
)

CHANNELS = (2, 3, 4, 5, 6, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v15_4_scale10")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--outdir", default="ml/results/trajectory_scale10")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = Path(args.data_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = train_manifest(data)
    train_windows = sum(n for _, n in manifest)
    xv_raw, fv_raw, yv = load_validation(data)
    current = xv_raw[:, -1, 0].astype(float)
    delta_v = yv - current[:, None]

    with np.load(Path(args.v14_1_dir) / "normalization.npz") as z:
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)

    xv = ((xv_raw - mean) / std).astype(np.float32)
    fv = normalize_future(fv_raw, mean, std)
    targets = tuple(yv[:, i] for i in range(4)) + tuple(delta_v[:, i] for i in range(4))

    seq = ShardSequence(manifest, mean, std, batch_size=args.batch_size, seed=SEED)
    model = build_trajectory_forecaster(xv.shape[1], xv.shape[2])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(out / "training_history.csv")),
    ]

    print("=== TRAJECTORY SCALE-10 FAST SCREEN ===")
    print(f"train shards:       {len(manifest):,}")
    print(f"train windows:      {train_windows:,}")
    print(f"validation windows: {len(yv):,}")
    print("decoder steps:      24 x 5 min")
    print("read steps:         6 / 12 / 18 / 24")
    print("future CGM input:   False")
    print(f"epochs max:         {args.epochs}")
    print(f"batch size:         {args.batch_size}")
    print(f"steps/epoch:        {len(seq):,}")

    model.fit(
        x=seq,
        validation_data=([xv, fv], targets),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best = out / "best.weights.h5"
    if best.exists():
        model.load_weights(best)

    raw = model.predict([xv, fv], batch_size=args.batch_size, verbose=1)
    absolute = np.column_stack([raw[i].ravel() for i in range(4)])
    delta_head = np.column_stack([raw[i + 4].ravel() for i in range(4)])
    reconstructed = current[:, None] + delta_head
    final = 0.5 * absolute + 0.5 * reconstructed

    abs_metrics = evaluate(yv, absolute, current)
    delta_metrics = evaluate(yv, reconstructed, current)
    final_metrics = evaluate(yv, final, current)

    abs_metrics.to_csv(out / "absolute_head_metrics.csv", index=False)
    delta_metrics.to_csv(out / "delta_reconstructed_metrics.csv", index=False)
    final_metrics.to_csv(out / "trajectory_scale10_validation_metrics.csv", index=False)
    np.savez_compressed(
        out / "validation_predictions.npz",
        y_true=yv,
        current_glucose=current,
        absolute_head=absolute,
        delta_head=delta_head,
        delta_reconstructed=reconstructed,
        y_pred=final,
    )

    means = {
        "mean_mard_4h": float(final_metrics.mard.mean()),
        "mean_rmse_4h": float(final_metrics.rmse.mean()),
        "mean_mae_4h": float(final_metrics.mae.mean()),
        "mean_direction_accuracy_4h": float(final_metrics.direction_accuracy.mean()),
    }
    report = {
        "experiment": "trajectory-scale10-fast-screen",
        "baseline": "V15.4 Scale-10",
        "scientific_change": "single causal recurrent future trajectory decoder with reads at 30/60/90/120 min",
        "intermediate_states": "latent; no unavailable intermediate CGM labels synthesized",
        "future_features": ["basal", "basal_missing", "bolus", "bolus_missing", "carbs", "carbs_missing"],
        "read_steps": READ_STEPS,
        "future_cgm_input": False,
        "normalization": "V14.1 frozen train-only statistics",
        "fusion": "fixed 0.5 absolute + 0.5 reconstructed delta",
        "seed": SEED,
        "train_windows": int(train_windows),
        "validation_windows": int(len(yv)),
        "epochs_requested": args.epochs,
        "test_parquet_used": False,
        "live_targets_used": False,
        **means,
    }
    (out / "trajectory_scale10_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTRAJECTORY SCALE-10 VALIDATION METRICS")
    print(final_metrics.to_string(index=False))
    print("\n4H MEANS")
    print(f"MARD: {means['mean_mard_4h']:.6f}%")
    print(f"RMSE: {means['mean_rmse_4h']:.6f}")
    print(f"Direction: {means['mean_direction_accuracy_4h']*100:.4f}%")
    print("\nComponent RMSEs")
    print(pd.DataFrame({
        "horizon": HORIZONS,
        "absolute_rmse": abs_metrics.rmse,
        "delta_reconstructed_rmse": delta_metrics.rmse,
        "hybrid_rmse": final_metrics.rmse,
    }).to_string(index=False))
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
