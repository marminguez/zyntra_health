"""Train Top-3 multi-scale history to convergence on frozen Scale-10 data.

Scientific change vs V15.4 Scale-10:
- same train/validation anchors
- same normalization, targets, future-known features, loss and 50/50 fusion
- replaces single full-history LSTM stack with a dual-scale history encoder:
  recent 2h high-resolution + full 24h pooled long-context trend

No future CGM is used.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.forecasting.model_multiscale_history import (
    HORIZONS,
    PREFIX_STEPS,
    build_multiscale_forecaster,
)

SEED = 42
CHANNELS = (2, 3, 4, 5, 6, 7)


def load_validation(root: Path):
    xs, fs, ys = [], [], []
    for p in sorted((root / "validation").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            fs.append(z["future_known"].astype(np.float32)[:, :, CHANNELS])
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No validation shards")
    return np.concatenate(xs), np.concatenate(fs), np.concatenate(ys)


def train_manifest(root: Path):
    items = []
    for p in sorted((root / "train").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            n = len(z["y"])
        if n:
            items.append((p, n))
    if not items:
        raise ValueError("No train shards")
    return items


def normalize_future(f: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    f = f.copy()
    for vi, mi, hist_idx in ((0, 1, 5), (2, 3, 7), (4, 5, 11)):
        present = f[:, :, mi] < 0.5
        f[:, :, vi] = np.where(
            present,
            (f[:, :, vi] - float(mean[hist_idx])) / float(std[hist_idx]),
            0.0,
        )
    return f.astype(np.float32)


class ShardSequence(tf.keras.utils.Sequence):
    def __init__(self, manifest, mean, std, batch_size=64, seed=SEED):
        super().__init__()
        self.manifest = manifest
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self._batches = []
        self._rebuild()

    def _rebuild(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.manifest))
        batches = []
        for mi in order:
            _, n = self.manifest[int(mi)]
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batches.append((int(mi), idx[start:start + self.batch_size]))
        rng.shuffle(batches)
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, index):
        mi, rows = self._batches[index]
        path, _ = self.manifest[mi]
        with np.load(path, allow_pickle=False) as z:
            x = z["x"][rows].astype(np.float32)
            f = z["future_known"][rows].astype(np.float32)[:, :, CHANNELS]
            y = z["y"][rows].astype(np.float32)
        current = x[:, -1, 0].copy()
        delta = y - current[:, None]
        x = ((x - self.mean) / self.std).astype(np.float32)
        f = normalize_future(f, self.mean, self.std)
        targets = tuple(y[:, i] for i in range(4)) + tuple(delta[:, i] for i in range(4))
        return (x, f), targets

    def on_epoch_end(self):
        self.epoch += 1
        self._rebuild()


def direction_class(d):
    return np.where(d > 5, 1, np.where(d < -5, -1, 0))


def evaluate(y, pred, current):
    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y[:, i].astype(float)
        yp = pred[:, i].astype(float)
        err = yp - yt
        rows.append({
            "horizon_minutes": h,
            "n": len(yt),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mard": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
            "direction_accuracy": float(
                np.mean(direction_class(yt - current) == direction_class(yp - current))
            ),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v15_4_scale10")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--outdir", default="ml/results/multiscale_scale10_40ep")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = Path(args.data_dir)
    v14 = Path(args.v14_1_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    backup = out / "training_backup"

    manifest = train_manifest(data)
    train_windows = sum(n for _, n in manifest)
    xv, fv, yv = load_validation(data)
    current_val = xv[:, -1, 0].astype(float)
    delta_val = yv - current_val[:, None]

    with np.load(v14 / "normalization.npz") as z:
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)

    xv = ((xv - mean) / std).astype(np.float32)
    fv = normalize_future(fv, mean, std)
    val_targets = tuple(yv[:, i] for i in range(4)) + tuple(delta_val[:, i] for i in range(4))

    train_seq = ShardSequence(manifest, mean, std, batch_size=args.batch_size, seed=SEED)
    model = build_multiscale_forecaster(xv.shape[1], xv.shape[2])

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup),
            save_freq="epoch",
            delete_checkpoint=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
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

    print("=== MULTISCALE HISTORY SCALE-10 — CONVERGENCE / RESUMABLE ===")
    print(f"train shards: {len(manifest):,}")
    print(f"train windows: {train_windows:,}")
    print(f"validation windows: {len(yv):,}")
    print(f"epochs ceiling: {args.epochs}")
    print("history encoder: recent 2h @5m + full 24h pooled @30m")
    print("future CGM input: False")

    hist = model.fit(
        x=train_seq,
        validation_data=([xv, fv], val_targets),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_weights = out / "best.weights.h5"
    if best_weights.exists():
        model.load_weights(best_weights)

    raw = model.predict([xv, fv], batch_size=args.batch_size, verbose=1)
    absolute = np.column_stack([raw[i].ravel() for i in range(4)])
    delta_head = np.column_stack([raw[i + 4].ravel() for i in range(4)])
    reconstructed = current_val[:, None] + delta_head
    final = 0.5 * absolute + 0.5 * reconstructed

    ma = evaluate(yv, absolute, current_val)
    mr = evaluate(yv, reconstructed, current_val)
    mf = evaluate(yv, final, current_val)
    ma.to_csv(out / "absolute_head_metrics.csv", index=False)
    mr.to_csv(out / "delta_reconstructed_metrics.csv", index=False)
    mf.to_csv(out / "multiscale_scale10_validation_metrics.csv", index=False)
    np.savez_compressed(
        out / "validation_predictions.npz",
        y_true=yv,
        current_glucose=current_val,
        absolute_head=absolute,
        delta_head=delta_head,
        delta_reconstructed=reconstructed,
        y_pred=final,
    )

    report = {
        "experiment": "multiscale-history-scale10-convergence",
        "scientific_change_vs_scale10": "dual-scale history encoder: recent 2h high-resolution + pooled 24h long context",
        "future_features": [
            "basal",
            "basal_missing",
            "bolus",
            "bolus_missing",
            "carbs",
            "carbs_missing",
        ],
        "future_prefix_steps": PREFIX_STEPS,
        "future_cgm_input": False,
        "fusion": "fixed 0.5 absolute + 0.5 reconstructed delta",
        "normalization": "V14.1 frozen train-only statistics",
        "seed": SEED,
        "train_windows": int(train_windows),
        "validation_windows": int(len(yv)),
        "epochs_ceiling": args.epochs,
        "training_protocol": "40ep ceiling; ES5; ReduceLR2; resumable",
        "best_val_loss_epoch_this_process": int(np.argmin(hist.history["val_loss"]) + 1) if hist.history.get("val_loss") else None,
        "promotion_gate": "graduate only if mean DTS-A improves about >=0.5 pp OR mean MARD improves >=2% relative, without material RMSE regression",
    }
    (out / "multiscale_scale10_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nMULTISCALE HISTORY SCALE-10 VALIDATION METRICS")
    print(mf.to_string(index=False))
    print("\n4H MEANS")
    print(f"MARD: {mf.mard.mean():.6f}%")
    print(f"RMSE: {mf.rmse.mean():.6f}")
    print(f"Direction: {100.0 * mf.direction_accuracy.mean():.4f}%")
    bestep = int(np.argmin(hist.history["val_loss"]) + 1) if hist.history.get("val_loss") else None
    print(f"Best val_loss epoch in this process: {bestep} | epochs run in this process: {len(hist.history.get('loss', []))}")
    print("\nComponent RMSEs")
    print(
        pd.DataFrame(
            {
                "horizon": HORIZONS,
                "absolute_rmse": ma.rmse,
                "delta_reconstructed_rmse": mr.rmse,
                "hybrid_rmse": mf.rmse,
            }
        ).to_string(index=False)
    )
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
