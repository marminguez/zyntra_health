"""Train FIR-1 on the frozen V15.4 Scale-50 dataset.

Controlled change vs Scale-50 baseline: add deterministic per-horizon future
intervention summaries. Dataset, validation, history/raw-future inputs, targets,
normalization, loss, 50/50 fusion, seed and callbacks remain aligned.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.forecasting.future_intervention_features import (
    SUMMARY_FEATURE_NAMES,
    build_future_intervention_summary,
)
from ml.forecasting.model_fir1 import HORIZONS, PREFIX_STEPS, build_fir1_forecaster

SEED = 42
CHANNELS = (2, 3, 4, 5, 6, 7)


def load_validation(root):
    xs, fs, ys = [], [], []
    for p in sorted((root / "validation").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            fs.append(z["future_known"].astype(np.float32)[:, :, CHANNELS])
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No validation shards")
    return np.concatenate(xs), np.concatenate(fs), np.concatenate(ys)


def manifest(root):
    items = []
    for p in sorted((root / "train").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            n = len(z["y"])
        if n:
            items.append((p, n))
    if not items:
        raise ValueError("No train shards")
    return items


def norm_future(f, mean, std):
    f = f.copy()
    for vi, mi, hi in ((0, 1, 5), (2, 3, 7), (4, 5, 11)):
        present = f[:, :, mi] < 0.5
        f[:, :, vi] = np.where(
            present,
            (f[:, :, vi] - float(mean[hi])) / float(std[hi]),
            0.0,
        )
    return f.astype(np.float32)


class ShardSequence(tf.keras.utils.Sequence):
    def __init__(self, items, mean, std, batch_size=64, seed=SEED):
        super().__init__()
        self.items = items
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.bs = batch_size
        self.seed = seed
        self.epoch = 0
        self._rebuild()

    def _rebuild(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for mi in rng.permutation(len(self.items)):
            _, n = self.items[int(mi)]
            idx = rng.permutation(n)
            for start in range(0, n, self.bs):
                batches.append((int(mi), idx[start : start + self.bs]))
        rng.shuffle(batches)
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        mi, rows = self.batches[i]
        p, _ = self.items[mi]
        with np.load(p, allow_pickle=False) as z:
            x = z["x"][rows].astype(np.float32)
            future_raw = z["future_known"][rows].astype(np.float32)[:, :, CHANNELS]
            y = z["y"][rows].astype(np.float32)

        current = x[:, -1, 0].copy()
        delta = y - current[:, None]
        summary = build_future_intervention_summary(
            future_raw,
            basal_mean=float(self.mean[5]),
            basal_std=float(self.std[5]),
        )
        x = ((x - self.mean) / self.std).astype(np.float32)
        future = norm_future(future_raw, self.mean, self.std)
        targets = tuple(y[:, j] for j in range(4)) + tuple(delta[:, j] for j in range(4))
        return (x, future, summary), targets

    def on_epoch_end(self):
        self.epoch += 1
        self._rebuild()


def direction_class(delta):
    return np.where(delta > 5, 1, np.where(delta < -5, -1, 0))


def evaluate(y, pred, current):
    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y[:, i].astype(float)
        yp = pred[:, i].astype(float)
        error = yp - yt
        rows.append(
            {
                "horizon_minutes": h,
                "n": len(yt),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mard": float(np.mean(np.abs(error) / np.maximum(np.abs(yt), 1e-6)) * 100),
                "direction_accuracy": float(
                    np.mean(
                        direction_class(yt - current)
                        == direction_class(yp - current)
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v15_4_scale50")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--outdir", default="ml/results/fir1_scale50")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = Path(args.data_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    backup = out / "training_backup"

    items = manifest(data)
    n_train = sum(n for _, n in items)
    xv_raw, fv_raw, yv = load_validation(data)
    current = xv_raw[:, -1, 0].astype(float)
    delta_v = yv - current[:, None]

    with np.load(Path(args.v14_1_dir) / "normalization.npz") as z:
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)

    sv = build_future_intervention_summary(
        fv_raw,
        basal_mean=float(mean[5]),
        basal_std=float(std[5]),
    )
    xv = ((xv_raw - mean) / std).astype(np.float32)
    fv = norm_future(fv_raw, mean, std)
    validation_targets = tuple(yv[:, i] for i in range(4)) + tuple(delta_v[:, i] for i in range(4))

    seq = ShardSequence(items, mean, std, args.batch_size)
    model = build_fir1_forecaster(xv.shape[1], xv.shape[2])

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup), save_freq="epoch", delete_checkpoint=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(out / "training_history.csv"), append=True),
    ]

    print("=== FIR-1 SCALE-50 TRAINING ===")
    print(f"train shards:       {len(items):,}")
    print(f"train windows:      {n_train:,}")
    print(f"validation windows: {len(yv):,}")
    print(f"summary features:   {len(SUMMARY_FEATURE_NAMES)} per horizon")
    print("future CGM input:   False")
    print(f"batch size:         {args.batch_size}")
    print(f"steps/epoch:        {len(seq):,}")
    print(f"backup dir:         {backup.resolve()}")
    if backup.exists() and any(backup.iterdir()):
        print("Resume checkpoint detected: Keras will restore training state automatically.")

    model.fit(
        x=seq,
        validation_data=([xv, fv, sv], validation_targets),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best = out / "best.weights.h5"
    if best.exists():
        model.load_weights(best)

    raw = model.predict([xv, fv, sv], batch_size=args.batch_size, verbose=1)
    absolute = np.column_stack([raw[i].ravel() for i in range(4)])
    delta_head = np.column_stack([raw[i + 4].ravel() for i in range(4)])
    reconstructed = current[:, None] + delta_head
    final = 0.5 * absolute + 0.5 * reconstructed

    abs_metrics = evaluate(yv, absolute, current)
    delta_metrics = evaluate(yv, reconstructed, current)
    final_metrics = evaluate(yv, final, current)
    abs_metrics.to_csv(out / "absolute_head_metrics.csv", index=False)
    delta_metrics.to_csv(out / "delta_reconstructed_metrics.csv", index=False)
    final_metrics.to_csv(out / "fir1_scale50_validation_metrics.csv", index=False)
    np.savez_compressed(
        out / "validation_predictions.npz",
        y_true=yv,
        current_glucose=current,
        absolute_head=absolute,
        delta_head=delta_head,
        delta_reconstructed=reconstructed,
        y_pred=final,
    )

    report = {
        "experiment": "FIR-1 Scale-50",
        "baseline": "V15.4 Scale-50",
        "controlled_change": "add 13 deterministic horizon-specific future-intervention summaries",
        "architecture": "V15.4 plus shared Dense(16) intervention-summary encoder",
        "summary_features": list(SUMMARY_FEATURE_NAMES),
        "future_features": [
            "basal", "basal_missing", "bolus", "bolus_missing", "carbs", "carbs_missing"
        ],
        "future_prefix_steps": PREFIX_STEPS,
        "future_cgm_input": False,
        "fusion": "fixed 0.5 absolute + 0.5 reconstructed delta",
        "normalization": "V14.1 frozen train-only statistics; FIR basal z summaries clipped to +/-5",
        "seed": SEED,
        "train_windows": int(n_train),
        "validation_windows": int(len(yv)),
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "resumable": True,
        "test_parquet_used": False,
        "live_targets_used": False,
    }
    (out / "fir1_scale50_report.json").write_text(json.dumps(report, indent=2))

    print("\nFIR-1 SCALE-50 VALIDATION METRICS")
    print(final_metrics.to_string(index=False))
    print("\nComponent RMSEs")
    print(
        pd.DataFrame(
            {
                "horizon": HORIZONS,
                "absolute_rmse": abs_metrics.rmse,
                "delta_reconstructed_rmse": delta_metrics.rmse,
                "hybrid_rmse": final_metrics.rmse,
            }
        ).to_string(index=False)
    )
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
