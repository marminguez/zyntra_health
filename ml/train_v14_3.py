"""Zyntra V14.3: extreme-aware multi-horizon forecasting experiment.

Keeps the V14.1 architecture and patient split fixed. The only intentional
training change is sample weighting: windows with future hypo/hyperglycemia or
large glucose excursions receive more weight. This tests whether the failure
identified in V14.2 is primarily an objective/imbalance problem rather than an
architecture problem.

No validation-derived threshold or weight optimization is performed.
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


def paths(root: Path, split: str):
    return sorted((root / split).glob("*.npz"))


def load(paths_):
    xs, ys = [], []
    for p in paths_:
        with np.load(p, allow_pickle=False) as z:
            xs.append(z["x"].astype(np.float32)); ys.append(z["y"].astype(np.float32))
    if not xs:
        raise ValueError("No shards found")
    return np.concatenate(xs), np.concatenate(ys)


def weights_for_horizon(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pre-registered coarse weights driven only by clinical/dynamic regime."""
    delta = target - current
    w = np.ones(len(target), dtype=np.float32)
    moderate = (target > 180) | (np.abs(delta) >= 30)
    severe = (target < 70) | (target > 250)
    w[moderate] = 2.0
    w[severe] = 3.0
    return w


def evaluate(y_true, y_pred, current):
    rows = []
    for i, h in enumerate(HORIZONS):
        yt, yp = y_true[:, i].astype(float), y_pred[:, i].astype(float)
        err = yp - yt
        td, pd_ = yt-current, yp-current
        tdir = np.where(td > 5, 1, np.where(td < -5, -1, 0))
        pdir = np.where(pd_ > 5, 1, np.where(pd_ < -5, -1, 0))
        rows.append({"horizon_minutes": h, "n": len(yt),
                     "mae": np.mean(np.abs(err)), "rmse": np.sqrt(np.mean(err**2)),
                     "mard": np.mean(np.abs(err)/np.maximum(np.abs(yt),1e-6))*100,
                     "direction_accuracy": np.mean(tdir == pdir)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v14_1")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    ap.add_argument("--outdir", default="ml/results/v14_3")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    data, olddir, out = Path(args.data_dir), Path(args.v14_1_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    xp, yp = load(paths(data,"train")); xv_raw, yv = load(paths(data,"validation"))
    with np.load(olddir/"normalization.npz") as z: mean, std = z["mean"], z["std"]
    xcurrent = xp[:,-1,0].copy(); vcurrent = xv_raw[:,-1,0].astype(float)
    xp = ((xp-mean)/std).astype(np.float32); xv=((xv_raw-mean)/std).astype(np.float32)

    # Keras 3 is strict about matching nested structures for multi-output
    # targets and sample_weight. Use lists in the exact model-output order.
    y_train_list = [yp[:, i] for i, _ in enumerate(HORIZONS)]
    y_val_list = [yv[:, i] for i, _ in enumerate(HORIZONS)]
    sw_list = [weights_for_horizon(xcurrent, yp[:, i]) for i, _ in enumerate(HORIZONS)]

    print("V14.3 fixed weighting prevalence (TRAIN only):")
    for i,h in enumerate(HORIZONS):
        w = sw_list[i]
        print(f"  +{h}: 1x={(w==1).mean():.1%}, 2x={(w==2).mean():.1%}, 3x={(w==3).mean():.1%}")

    model=build_v14_forecaster(xp.shape[1],xp.shape[2])
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=4,restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=2,min_lr=1e-5),
               tf.keras.callbacks.ModelCheckpoint(str(out/"best_model.keras"),monitor="val_loss",save_best_only=True)]
    hist=model.fit(
        xp,
        y_train_list,
        sample_weight=sw_list,
        validation_data=(xv, y_val_list),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    raw=model.predict(xv,batch_size=args.batch_size,verbose=1)
    pred=np.column_stack([a.reshape(-1) for a in raw])
    metrics=evaluate(yv,pred,vcurrent); metrics.to_csv(out/"v14_3_validation_metrics.csv",index=False)
    pd.DataFrame(hist.history).to_csv(out/"v14_3_training_history.csv",index=False)
    np.savez_compressed(out/"validation_predictions.npz",y_true=yv,y_pred=pred,current_glucose=vcurrent)
    report={"version":"v14.3","hypothesis":"extreme-aware weighting reduces V14.2 extreme-regime error without sacrificing general forecasting",
            "architecture":"unchanged from V14.1","weights":{"normal":1.0,"moderate_hyper_or_abs_delta_ge_30":2.0,"hypo_or_severe_hyper":3.0},
            "weight_selection":"fixed a priori; not tuned on validation","normalization":"reused V14.1 train-only statistics",
            "test_parquet_used":False,"clinical_status":"retrospective research only; not for clinical decision-making"}
    (out/"v14_3_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\nV14.3 VALIDATION METRICS"); print(metrics.to_string(index=False)); print(f"\nArtifacts written to {out}")

if __name__=="__main__": main()
