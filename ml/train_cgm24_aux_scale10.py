"""Fast-screen Scale-10 control with a dedicated 24h CGM-only auxiliary encoder.

Scientific change vs V15.4 Scale-10:
- same train/validation anchors
- same normalization, targets, loss, batch size, future-known features and fusion
- adds one separately encoded CGM-only 24h branch from ml/data/glucofm_scale10

This is NOT GlucoFM and uses no external/pretrained weights.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.forecasting.model_cgm24_aux import HORIZONS, PREFIX_STEPS, build_cgm24_aux_forecaster

SEED = 42
CHANNELS = (2, 3, 4, 5, 6, 7)


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


def load_validation(data_root: Path, sidecar_root: Path):
    xs, fs, ys, cs = [], [], [], []
    for p in sorted((data_root / "validation").glob("*.npz")):
        side = sidecar_root / "validation" / p.name
        if not side.exists():
            raise FileNotFoundError(side)
        with np.load(p, allow_pickle=False) as z, np.load(side, allow_pickle=False) as s:
            if not np.array_equal(z["timestamp"], s["timestamp"]):
                raise AssertionError(f"Timestamp mismatch: {p.name}")
            xs.append(z["x"].astype(np.float32))
            fs.append(z["future_known"].astype(np.float32)[:, :, CHANNELS])
            ys.append(z["y"].astype(np.float32))
            cs.append(s["cgm_24h"].astype(np.float32)[..., None])
    if not xs:
        raise ValueError("No validation shards")
    return np.concatenate(xs), np.concatenate(fs), np.concatenate(ys), np.concatenate(cs)


def train_manifest(data_root: Path, sidecar_root: Path):
    items = []
    for p in sorted((data_root / "train").glob("*.npz")):
        side = sidecar_root / "train" / p.name
        if not side.exists():
            raise FileNotFoundError(side)
        with np.load(p, allow_pickle=False) as z, np.load(side, allow_pickle=False) as s:
            n = len(z["y"])
            if n != len(s["cgm_24h"]):
                raise AssertionError(f"Row count mismatch: {p.name}")
            if not np.array_equal(z["timestamp"], s["timestamp"]):
                raise AssertionError(f"Timestamp mismatch: {p.name}")
        if n:
            items.append((p, side, n))
    if not items:
        raise ValueError("No train shards")
    return items


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
            _, _, n = self.manifest[int(mi)]
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batches.append((int(mi), idx[start:start + self.batch_size]))
        rng.shuffle(batches)
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, index):
        mi, rows = self._batches[index]
        path, side, _ = self.manifest[mi]
        with np.load(path, allow_pickle=False) as z, np.load(side, allow_pickle=False) as s:
            x = z["x"][rows].astype(np.float32)
            f = z["future_known"][rows].astype(np.float32)[:, :, CHANNELS]
            y = z["y"][rows].astype(np.float32)
            c = s["cgm_24h"][rows].astype(np.float32)[..., None]

        current = x[:, -1, 0].copy()
        delta = y - current[:, None]
        x = ((x - self.mean) / self.std).astype(np.float32)
        f = normalize_future(f, self.mean, self.std)
        c = ((c - float(self.mean[0])) / float(self.std[0])).astype(np.float32)
        targets = tuple(y[:, i] for i in range(4)) + tuple(delta[:, i] for i in range(4))
        return (x, f, c), targets

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
    ap.add_argument("--sidecar-dir", default="ml/data/glucofm_scale10")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--outdir", default="ml/results/cgm24_aux_scale10")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = Path(args.data_dir)
    sidecar = Path(args.sidecar_dir)
    v14 = Path(args.v14_1_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    backup = out / "training_backup"

    manifest = train_manifest(data, sidecar)
    train_windows = sum(n for _, _, n in manifest)
    xv, fv, yv, cv = load_validation(data, sidecar)
    current_val = xv[:, -1, 0].astype(float)
    delta_val = yv - current_val[:, None]

    with np.load(v14 / "normalization.npz") as z:
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)

    xv = ((xv - mean) / std).astype(np.float32)
    fv = normalize_future(fv, mean, std)
    cv = ((cv - float(mean[0])) / float(std[0])).astype(np.float32)
    val_targets = tuple(yv[:, i] for i in range(4)) + tuple(delta_val[:, i] for i in range(4))

    train_seq = ShardSequence(manifest, mean, std, batch_size=args.batch_size, seed=SEED)
    model = build_cgm24_aux_forecaster(xv.shape[1], xv.shape[2])

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup), save_freq="epoch", delete_checkpoint=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best.weights.h5"), monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(out / "training_history.csv"), append=True),
    ]

    print("=== CGM24 AUX SCALE-10 FAST SCREEN ===")
    print("NOTE: Zyntra baseline already uses 24h multivariable history.")
    print("This experiment only tests a dedicated CGM-only auxiliary encoder.")
    print(f"train shards: {len(manifest):,}")
    print(f"train windows: {train_windows:,}")
    print(f"validation windows: {len(yv):,}")
    print(f"epochs requested: {args.epochs}")

    model.fit(
        x=train_seq,
        validation_data=([xv, fv, cv], val_targets),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_weights = out / "best.weights.h5"
    if best_weights.exists():
        model.load_weights(best_weights)

    raw = model.predict([xv, fv, cv], batch_size=args.batch_size, verbose=1)
    absolute = np.column_stack([raw[i].ravel() for i in range(4)])
    delta_head = np.column_stack([raw[i+4].ravel() for i in range(4)])
    reconstructed = current_val[:, None] + delta_head
    final = 0.5 * absolute + 0.5 * reconstructed

    ma = evaluate(yv, absolute, current_val)
    mr = evaluate(yv, reconstructed, current_val)
    mf = evaluate(yv, final, current_val)
    ma.to_csv(out / "absolute_head_metrics.csv", index=False)
    mr.to_csv(out / "delta_reconstructed_metrics.csv", index=False)
    mf.to_csv(out / "cgm24_aux_scale10_validation_metrics.csv", index=False)
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
        "experiment": "cgm24-aux-scale10",
        "scientific_change_vs_scale10": "dedicated learnable 24h CGM-only auxiliary LSTM branch",
        "new_information_vs_scale10": False,
        "purpose": "fusion-seam control before official frozen GlucoFM embeddings are available",
        "glucofm_used": False,
        "external_weights_used": False,
        "future_cgm_input": False,
        "history_context": "24h multivariable baseline + duplicate CGM-only 24h encoder",
        "fusion": "fixed 0.5 absolute + 0.5 reconstructed delta",
        "normalization": "V14.1 frozen train-only statistics; CGM uses glucose channel mean/std",
        "seed": SEED,
        "train_windows": int(train_windows),
        "validation_windows": int(len(yv)),
        "epochs_requested": args.epochs,
        "promotion_gate": "graduate only if mean DTS-A improves about >=0.5 pp OR mean MARD improves >=2% relative, without material RMSE regression",
    }
    (out / "cgm24_aux_scale10_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nCGM24 AUX SCALE-10 VALIDATION METRICS")
    print(mf.to_string(index=False))
    print("\n4H MEANS")
    print(f"MARD: {mf.mard.mean():.6f}%")
    print(f"RMSE: {mf.rmse.mean():.6f}")
    print(f"Direction: {100.0 * mf.direction_accuracy.mean():.4f}%")
    print("\nComponent RMSEs")
    print(pd.DataFrame({
        "horizon": HORIZONS,
        "absolute_rmse": ma.rmse,
        "delta_reconstructed_rmse": mr.rmse,
        "hybrid_rmse": mf.rmse,
    }).to_string(index=False))
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
