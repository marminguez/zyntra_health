"""Fast Scale-10 ablation for independent horizon-specific dense towers.

Everything is frozen to the V15.4 Scale-10 baseline except replacing the shared
Dense(32)+Dropout representation with an independent tower per horizon.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from forecasting.model_v15_4_horizon_towers import (
    HORIZONS,
    build_v15_4_horizon_towers_forecaster,
)

SEED = 42
CHANNELS = (2, 3, 4, 5, 6, 7)
BASELINE_MARD = 14.5988565
BASELINE_RMSE = 30.88723775
BASELINE_DIRECTION = 0.6916525


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


def normalize_future(f, mean, std):
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
    def __init__(self, manifest, mean, std, batch_size=64, seed=SEED):
        super().__init__()
        self.manifest = manifest
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self._rebuild()

    def _rebuild(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for mi in rng.permutation(len(self.manifest)):
            _, n = self.manifest[int(mi)]
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batches.append((int(mi), idx[start:start + self.batch_size]))
        rng.shuffle(batches)
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        mi, rows = self.batches[index]
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
            "direction_accuracy": float(np.mean(direction_class(yt-current) == direction_class(yp-current))),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v15_4_scale10")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--outdir", default="ml/results/horizon_towers_scale10")
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
    n_train = sum(n for _, n in manifest)
    xv_raw, fv_raw, yv = load_validation(data)
    current = xv_raw[:, -1, 0].astype(float)
    delta_v = yv - current[:, None]

    with np.load(Path(args.v14_1_dir) / "normalization.npz") as z:
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)

    xv = ((xv_raw - mean) / std).astype(np.float32)
    fv = normalize_future(fv_raw, mean, std)
    val_targets = tuple(yv[:, i] for i in range(4)) + tuple(delta_v[:, i] for i in range(4))

    seq = ShardSequence(manifest, mean, std, args.batch_size)
    model = build_v15_4_horizon_towers_forecaster(xv.shape[1], xv.shape[2])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best.weights.h5"), monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(out / "training_history.csv")),
    ]

    print("=== HORIZON TOWERS SCALE-10 FAST SCREEN ===")
    print(f"train shards: {len(manifest):,}")
    print(f"train windows: {n_train:,}")
    print(f"validation windows: {len(yv):,}")
    print(f"epochs requested: {args.epochs}")
    print("history encoder: exact V15.4")
    print("future encoder: exact V15.4")
    print("loss: original Huber")
    print("experimental change: independent Dense(32)+Dropout tower per horizon")
    print("future CGM input: False")

    model.fit(
        seq,
        validation_data=([xv, fv], val_targets),
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

    ma = evaluate(yv, absolute, current)
    md = evaluate(yv, reconstructed, current)
    mf = evaluate(yv, final, current)
    mf.to_csv(out / "horizon_towers_scale10_validation_metrics.csv", index=False)
    np.savez_compressed(
        out / "validation_predictions.npz",
        y_true=yv,
        current_glucose=current,
        absolute_head=absolute,
        delta_head=delta_head,
        delta_reconstructed=reconstructed,
        y_pred=final,
    )

    mean_mard = float(mf.mard.mean())
    mean_rmse = float(mf.rmse.mean())
    mean_direction = float(mf.direction_accuracy.mean())
    rel_mard = 100.0 * (mean_mard / BASELINE_MARD - 1.0)
    rel_rmse = 100.0 * (mean_rmse / BASELINE_RMSE - 1.0)
    dir_pp = 100.0 * (mean_direction - BASELINE_DIRECTION)
    go = ((rel_mard <= -2.0) or (dir_pp >= 0.5)) and rel_rmse <= 1.0

    report = {
        "experiment": "horizon-towers-scale10-fast-screen",
        "baseline": "V15.4 Scale-10",
        "controlled_change": "independent Dense(32)+Dropout(0.10) representation per horizon",
        "history_encoder": "exact V15.4",
        "future_encoder": "exact V15.4",
        "loss": "original Huber for all 8 outputs",
        "fusion": "fixed 0.5 absolute + 0.5 reconstructed delta",
        "future_cgm_input": False,
        "normalization": "V14.1 frozen train-only statistics",
        "train_windows": int(n_train),
        "validation_windows": int(len(yv)),
        "epochs_requested": args.epochs,
        "result": {
            "mard": mean_mard,
            "rmse": mean_rmse,
            "direction": mean_direction,
            "mard_relative_change_pct": rel_mard,
            "rmse_relative_change_pct": rel_rmse,
            "direction_change_pp": dir_pp,
        },
        "promotion_gate": "MARD improves >=2% relative OR direction >=+0.5pp, with RMSE regression <=1%",
        "go_scale50": bool(go),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nHORIZON TOWERS SCALE-10 VALIDATION METRICS")
    print(mf.to_string(index=False))
    print("\n4H MEANS")
    print(f"MARD: {mean_mard:.6f}%")
    print(f"RMSE: {mean_rmse:.6f}")
    print(f"Direction: {100*mean_direction:.4f}%")
    print("\nVS SCALE-10")
    print(f"MARD relative change: {rel_mard:+.3f}%")
    print(f"RMSE relative change: {rel_rmse:+.3f}%")
    print(f"Direction change: {dir_pp:+.3f} pp")
    print(f"PROMOTION TO SCALE-50: {'GO' if go else 'NO-GO'}")
    print("\nComponent RMSEs")
    print(pd.DataFrame({
        "horizon": HORIZONS,
        "absolute_rmse": ma.rmse,
        "delta_reconstructed_rmse": md.rmse,
        "hybrid_rmse": mf.rmse,
    }).to_string(index=False))


if __name__ == "__main__":
    main()
