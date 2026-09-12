"""FIR-1: deterministic future-intervention summaries for Top-3 experiments.

Input channels must match the six future-known channels selected by V15.4:
    basal, basal_missing, bolus, bolus_missing, carbs, carbs_missing

Every derived feature is computed only from the future-known prefix available to
its target horizon (+30/+60/+90/+120). No CGM values are read or derived here.
"""
from __future__ import annotations

import numpy as np

HORIZONS = (30, 60, 90, 120)
PREFIX_STEPS = (6, 12, 18, 24)
STEP_MINUTES = 5

SUMMARY_FEATURE_NAMES = (
    "log1p_bolus_sum",
    "log1p_carbs_sum",
    "bolus_event_rate",
    "carb_event_rate",
    "time_to_first_bolus_frac",
    "time_to_first_carb_frac",
    "log1p_first_bolus",
    "log1p_first_carbs",
    "basal_mean_z",
    "basal_delta_z",
    "basal_coverage",
    "basal_change_rate",
    "log1p_basal_total_variation",
)


def _first_event(values: np.ndarray, event: np.ndarray, horizon_minutes: int):
    any_event = event.any(axis=1)
    first_idx = np.argmax(event, axis=1)
    first_value = values[np.arange(len(values)), first_idx]
    time_frac = np.where(
        any_event,
        ((first_idx + 1) * STEP_MINUTES) / float(horizon_minutes),
        1.0,
    )
    magnitude = np.where(any_event, np.maximum(first_value, 0.0), 0.0)
    return time_frac.astype(np.float32), np.log1p(magnitude).astype(np.float32)


def _basal_stats(
    basal: np.ndarray,
    present: np.ndarray,
    basal_mean: float,
    basal_std: float,
):
    n, steps = basal.shape
    count = present.sum(axis=1)
    safe_count = np.maximum(count, 1)
    mean_raw = np.sum(np.where(present, basal, 0.0), axis=1) / safe_count
    any_present = count > 0

    first_idx = np.argmax(present, axis=1)
    last_idx = steps - 1 - np.argmax(present[:, ::-1], axis=1)
    first = basal[np.arange(n), first_idx]
    last = basal[np.arange(n), last_idx]
    delta = np.where(any_present, last - first, 0.0)

    std = float(basal_std) if float(basal_std) > 1e-6 else 1.0
    mean_z = np.where(any_present, (mean_raw - float(basal_mean)) / std, 0.0)
    delta_z = delta / std
    coverage = count / float(steps)

    pair_present = present[:, 1:] & present[:, :-1]
    diffs = np.where(pair_present, np.abs(np.diff(basal, axis=1)), 0.0)
    changed = pair_present & (np.abs(np.diff(basal, axis=1)) > 1e-6)
    change_rate = changed.sum(axis=1) / float(max(steps - 1, 1))
    total_variation = diffs.sum(axis=1)

    return (
        mean_z.astype(np.float32),
        delta_z.astype(np.float32),
        coverage.astype(np.float32),
        change_rate.astype(np.float32),
        np.log1p(np.maximum(total_variation, 0.0)).astype(np.float32),
    )


def build_future_intervention_summary(
    future_raw: np.ndarray,
    basal_mean: float,
    basal_std: float,
) -> np.ndarray:
    """Return FIR-1 summaries with shape ``(N, 4, 13)``.

    Parameters
    ----------
    future_raw:
        Unnormalized array shaped ``(N, 24, 6)`` with channels
        basal, basal_missing, bolus, bolus_missing, carbs, carbs_missing.
    basal_mean / basal_std:
        Frozen train-only V14.1 basal normalization statistics.
    """
    f = np.asarray(future_raw, dtype=np.float32)
    if f.ndim != 3 or f.shape[1:] != (24, 6):
        raise ValueError(f"Expected future_raw shape (N, 24, 6), got {f.shape}")

    result = np.empty(
        (len(f), len(HORIZONS), len(SUMMARY_FEATURE_NAMES)), dtype=np.float32
    )

    for hi, (horizon, steps) in enumerate(zip(HORIZONS, PREFIX_STEPS)):
        p = f[:, :steps, :]
        basal, basal_missing = p[:, :, 0], p[:, :, 1]
        bolus, bolus_missing = p[:, :, 2], p[:, :, 3]
        carbs, carbs_missing = p[:, :, 4], p[:, :, 5]

        basal_present = basal_missing < 0.5
        bolus_event = (bolus_missing < 0.5) & (bolus > 0.0)
        carb_event = (carbs_missing < 0.5) & (carbs > 0.0)

        bolus_positive = np.where(bolus_event, bolus, 0.0)
        carbs_positive = np.where(carb_event, carbs, 0.0)

        bolus_sum = np.log1p(np.maximum(bolus_positive.sum(axis=1), 0.0))
        carbs_sum = np.log1p(np.maximum(carbs_positive.sum(axis=1), 0.0))
        bolus_rate = bolus_event.sum(axis=1) / float(steps)
        carb_rate = carb_event.sum(axis=1) / float(steps)
        t_bolus, first_bolus = _first_event(bolus, bolus_event, horizon)
        t_carb, first_carb = _first_event(carbs, carb_event, horizon)
        basal_stats = _basal_stats(basal, basal_present, basal_mean, basal_std)

        result[:, hi, :] = np.column_stack(
            [
                bolus_sum,
                carbs_sum,
                bolus_rate,
                carb_rate,
                t_bolus,
                t_carb,
                first_bolus,
                first_carb,
                *basal_stats,
            ]
        ).astype(np.float32)

    if not np.isfinite(result).all():
        raise ValueError("Non-finite values generated in FIR-1 summary")
    return result
