"""Train Zyntra V14.6 drop-aware multitask forecaster.

Fixed from V14.5:
- same V14.1 dataset/split/normalization
- same LSTM backbone and absolute/delta forecast heads
- same fixed 50/50 absolute + reconstructed-delta fusion
- no sample weighting

Intentional change:
- add one auxiliary rapid-drop classifier per horizon with label
  G(t+h)-G(t) <= -30 mg/dL
- auxiliary heads shape the shared representation only; they do not enter the
  final glucose prediction
- auxiliary BCE loss weight is fixed a priori at 5.0 and is not validation tuned
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.forecasting.model_v14_6 import (
    AUX_LOSS_WEIGHT,
    HORIZONS,
    build_v14_6_forecaster,
)

SEED = 42
DROP_THRESHOLD_MGDL = -30.0


def load(root: Path, split: str):
    xs, ys = [], []
    for p in sorted((root / split).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32))
            ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError(f"No shards found for {split}")
    return np.concatenate(xs), np.concatenate(ys)


def dclass(d):
    return np.where(d > 5, 1, np.where(d < -5, -1, 0))


def evaluate(y, p, cur):
    rows = []
    for i, h in enumerate(HORIZONS):
        yt = y[:, i].astype(float)
        yp = p[:, i].astype(float)
        e = yp - yt
        rows.append({
            "horizon_minutes": h,
            "n": len(yt),
            "mae": np.mean(np.abs(e)),
            "rmse": np.sqrt(np.mean(e ** 2)),
            "mard": np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-6)) * 100,
            "direction_accuracy": np.mean(dclass(yt - cur) == dclass(yp - cur)),
        })
    return pd.DataFrame(rows)


def binary_metrics(y_true, p):
    y_true = y_true.astype(np.int32)
    pred = (p >= 0.5).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "prevalence": float(np.mean(y_true)),
        "accuracy_at_0_5": float(np.mean(pred == y_true)),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "specificity_at_0_5": specificity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v14_1")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--outdir", default="ml/results/v14_6")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = Path(a.data_dir)
    d1 = Path(a.v14_1_dir)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    xt, yt = load(data, "train")
    xv, yv = load(data, "validation")
    ct = xt[:, -1, 0].astype(np.float32)
    cv = xv[:, -1, 0].astype(np.float64)

    dt = yt - ct[:, None]
    dv = yv - cv[:, None]
    drop_t = (dt <= DROP_THRESHOLD_MGDL).astype(np.float32)
    drop_v = (dv <= DROP_THRESHOLD_MGDL).astype(np.float32)

    with np.load(d1 / "normalization.npz") as z:
        mean, std = z["mean"], z["std"]
    xtn = ((xt - mean) / std).astype(np.float32)
    xvn = ((xv - mean) / std).astype(np.float32)

    model = build_v14_6_forecaster(xtn.shape[1], xtn.shape[2])
    train_targets = (
        [yt[:, i] for i in range(4)]
        + [dt[:, i] for i in range(4)]
        + [drop_t[:, i] for i in range(4)]
    )
    val_targets = (
        [yv[:, i] for i in range(4)]
        + [dv[:, i] for i in range(4)]
        + [drop_v[:, i] for i in range(4)]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best_model.keras"), monitor="val_loss", save_best_only=True
        ),
    ]

    hist = model.fit(
        xtn,
        train_targets,
        validation_data=(xvn, val_targets),
        epochs=a.epochs,
        batch_size=a.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    raw = model.predict(xvn, batch_size=a.batch_size, verbose=1)
    pa = np.column_stack([raw[i].reshape(-1) for i in range(4)]).astype(float)
    pdlt = np.column_stack([raw[i + 4].reshape(-1) for i in range(4)]).astype(float)
    pdrop = np.column_stack([raw[i + 8].reshape(-1) for i in range(4)]).astype(float)
    pr = cv[:, None] + pdlt
    pf = 0.5 * pa + 0.5 * pr

    ma = evaluate(yv, pa, cv)
    mr = evaluate(yv, pr, cv)
    mf = evaluate(yv, pf, cv)
    ma.to_csv(out / "absolute_head_metrics.csv", index=False)
    mr.to_csv(out / "delta_reconstructed_metrics.csv", index=False)
    mf.to_csv(out / "v14_6_validation_metrics.csv", index=False)
    pd.DataFrame(hist.history).to_csv(out / "training_history.csv", index=False)

    drop_rows = []
    for i, h in enumerate(HORIZONS):
        m = binary_metrics(drop_v[:, i], pdrop[:, i])
        auc = tf.keras.metrics.AUC()
        auc.update_state(drop_v[:, i], pdrop[:, i])
        m.update({"horizon_minutes": h, "auc": float(auc.result().numpy())})
        drop_rows.append(m)
    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_csv(out / "rapid_drop_aux_metrics.csv", index=False)

    np.savez_compressed(
        out / "validation_predictions.npz",
        y_true=yv,
        current_glucose=cv,
        absolute_head=pa,
        delta_head=pdlt,
        delta_reconstructed=pr,
        rapid_drop_probability=pdrop,
        y_pred=pf,
    )

    report = {
        "version": "v14.6",
        "hypothesis": "an explicit rapid-drop auxiliary task preserves descending-dynamics representations while retaining V14.5 hybrid forecasting gains",
        "architecture": "V14.5 absolute+delta forecast heads plus four auxiliary rapid-drop sigmoid heads",
        "rapid_drop_label": "G(t+h)-G(t) <= -30 mg/dL",
        "auxiliary_loss": "binary crossentropy",
        "auxiliary_loss_weight": AUX_LOSS_WEIGHT,
        "auxiliary_weight_tuned_on_validation": False,
        "auxiliary_heads_used_in_final_forecast": False,
        "fusion": "fixed 0.5 absolute + 0.5 reconstructed delta",
        "sample_weighting": False,
        "normalization": "V14.1 train-only statistics",
        "train_windows": int(len(xtn)),
        "validation_windows": int(len(xvn)),
        "test_parquet_used": False,
        "clinical_status": "retrospective research only; not for clinical decision-making",
    }
    (out / "v14_6_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nV14.6 VALIDATION METRICS — DROP-AWARE HYBRID")
    print(mf.to_string(index=False))
    print("\nComponent RMSEs")
    print(pd.DataFrame({
        "horizon": HORIZONS,
        "absolute_rmse": ma.rmse,
        "delta_reconstructed_rmse": mr.rmse,
        "hybrid_rmse": mf.rmse,
    }).to_string(index=False))
    print("\nRapid-drop auxiliary metrics")
    print(drop_df.to_string(index=False))
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
