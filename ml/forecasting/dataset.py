"""Dataset utilities for Zyntra V14 multi-horizon forecasting.

Targets are created independently inside each patient/source sequence so future
values can never cross patient or dataset boundaries.
"""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd

HORIZONS_MINUTES = (30, 60, 90, 120)
SAMPLE_MINUTES = 5


def _group_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in ("p_id", "patient_id", "source_file", "source_split") if c in df.columns]
    # Prefer one patient identifier, while retaining source boundaries when present.
    if "p_id" in cols and "patient_id" in cols:
        cols.remove("patient_id")
    return cols


def add_forecast_targets(
    df: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS_MINUTES,
    glucose_col: str = "glucose",
    sample_minutes: int = SAMPLE_MINUTES,
) -> pd.DataFrame:
    """Return a copy with glucose targets at each requested future horizon.

    Assumes regular ``sample_minutes`` sampling inside each patient/source
    sequence. Rows without all requested future targets are removed.
    """
    if glucose_col not in df.columns:
        raise ValueError(f"Missing required column: {glucose_col}")
    horizons = tuple(int(h) for h in horizons)
    if any(h <= 0 or h % sample_minutes for h in horizons):
        raise ValueError("Every horizon must be a positive multiple of sample_minutes")

    work = df.copy()
    groups = _group_columns(work)
    sort_cols = groups.copy()
    if "timestamp" in work.columns:
        sort_cols.append("timestamp")
    elif isinstance(work.index, pd.DatetimeIndex):
        work = work.assign(_timestamp=work.index)
        sort_cols.append("_timestamp")
    if sort_cols:
        work = work.sort_values(sort_cols)

    grouped = work.groupby(groups, sort=False, dropna=False)[glucose_col] if groups else None
    target_cols = []
    for horizon in horizons:
        col = f"target_{horizon}"
        steps = horizon // sample_minutes
        work[col] = grouped.shift(-steps) if grouped is not None else work[glucose_col].shift(-steps)
        target_cols.append(col)

    work = work.dropna(subset=[glucose_col, *target_cols]).copy()
    if "_timestamp" in work.columns:
        work = work.drop(columns="_timestamp")
    return work


def add_glucose_dynamics(df: pd.DataFrame, glucose_col: str = "glucose") -> pd.DataFrame:
    """Add causal glucose dynamics used by Zyntra without future leakage."""
    work = df.copy()
    groups = _group_columns(work)
    g = work.groupby(groups, sort=False, dropna=False)[glucose_col] if groups else None

    def diff(periods: int) -> pd.Series:
        return g.diff(periods) if g is not None else work[glucose_col].diff(periods)

    work["glucose_delta_5m"] = diff(1)
    work["glucose_delta_15m"] = diff(3)
    work["glucose_delta_30m"] = diff(6)
    previous_delta15 = work["glucose_delta_15m"].groupby(
        [work[c] for c in groups], sort=False, dropna=False
    ).shift(3) if groups else work["glucose_delta_15m"].shift(3)
    work["glucose_acceleration_15m"] = work["glucose_delta_15m"] - previous_delta15
    return work


def describe_forecast_dataset(df: pd.DataFrame) -> dict:
    patient_col = "p_id" if "p_id" in df.columns else "patient_id" if "patient_id" in df.columns else None
    return {
        "rows": int(len(df)),
        "patients": int(df[patient_col].nunique()) if patient_col else None,
        "glucose_min": float(df.glucose.min()) if len(df) else None,
        "glucose_max": float(df.glucose.max()) if len(df) else None,
        "hypoglycemia_rows": int((df.glucose < 70).sum()) if len(df) else 0,
        "hyperglycemia_rows": int((df.glucose > 180).sum()) if len(df) else 0,
    }
