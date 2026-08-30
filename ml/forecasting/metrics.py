"""Evaluation metrics for Zyntra V14."""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd

from .dataset import HORIZONS_MINUTES


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if not len(y_true):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mard": np.nan}
    error = y_pred - y_true
    denom = np.maximum(np.abs(y_true), 1e-6)
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mard": float(np.mean(np.abs(error) / denom) * 100.0),
    }


def direction_accuracy(current: np.ndarray, future: np.ndarray, predicted: np.ndarray, stable_band: float = 5.0) -> float:
    """Three-class accuracy for falling/stable/rising trajectories."""
    current = np.asarray(current, dtype=float)
    true_delta = np.asarray(future, dtype=float) - current
    pred_delta = np.asarray(predicted, dtype=float) - current
    true_dir = np.where(true_delta > stable_band, 1, np.where(true_delta < -stable_band, -1, 0))
    pred_dir = np.where(pred_delta > stable_band, 1, np.where(pred_delta < -stable_band, -1, 0))
    mask = np.isfinite(true_delta) & np.isfinite(pred_delta)
    return float(np.mean(true_dir[mask] == pred_dir[mask])) if mask.any() else np.nan


def evaluate_forecasts(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS_MINUTES,
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        target_col, pred_col = f"target_{int(horizon)}", f"pred_{int(horizon)}"
        if target_col not in frame or pred_col not in predictions:
            raise ValueError(f"Missing {target_col} or {pred_col}")
        values = _metrics(frame[target_col].to_numpy(), predictions[pred_col].to_numpy())
        values.update({
            "horizon_minutes": int(horizon),
            "direction_accuracy": direction_accuracy(
                frame.glucose.to_numpy(), frame[target_col].to_numpy(), predictions[pred_col].to_numpy()
            ),
        })
        rows.append(values)
    return pd.DataFrame(rows)[["horizon_minutes", "n", "mae", "rmse", "mard", "direction_accuracy"]]


def clinical_slice_metrics(frame: pd.DataFrame, predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Error anatomy by future glycemic range for a single horizon."""
    target_col, pred_col = f"target_{horizon}", f"pred_{horizon}"
    y = frame[target_col]
    slices = {
        "hypoglycemia_<70": y < 70,
        "target_70_180": (y >= 70) & (y <= 180),
        "hyperglycemia_>180": y > 180,
        "severe_hyper_>250": y > 250,
    }
    rows = []
    for name, mask in slices.items():
        vals = _metrics(y[mask].to_numpy(), predictions.loc[mask, pred_col].to_numpy())
        rows.append({"slice": name, "horizon_minutes": horizon, **vals})
    return pd.DataFrame(rows)
