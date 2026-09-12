"""Mandatory V14 forecasting baselines."""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd

from .dataset import HORIZONS_MINUTES


def persistence_predictions(
    frame: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS_MINUTES,
    glucose_col: str = "glucose",
) -> pd.DataFrame:
    """Predict that future glucose equals current glucose."""
    return pd.DataFrame(
        {f"pred_{int(h)}": frame[glucose_col].to_numpy(dtype=float) for h in horizons},
        index=frame.index,
    )


def linear_trend_predictions(
    frame: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS_MINUTES,
    glucose_col: str = "glucose",
    slope_col: str = "glucose_delta_15m",
    clip: tuple[float, float] = (40.0, 400.0),
) -> pd.DataFrame:
    """Project the most recent 15-minute glucose slope into the future.

    This is intentionally simple and causal. It establishes whether a learned
    model adds value beyond extrapolating the current CGM direction.
    """
    if slope_col not in frame.columns:
        raise ValueError(f"Missing {slope_col}; call add_glucose_dynamics first")
    current = frame[glucose_col].to_numpy(dtype=float)
    slope_per_minute = frame[slope_col].fillna(0.0).to_numpy(dtype=float) / 15.0
    low, high = clip
    return pd.DataFrame(
        {
            f"pred_{int(h)}": np.clip(current + slope_per_minute * int(h), low, high)
            for h in horizons
        },
        index=frame.index,
    )
