"""FIR-2: richer deterministic future-intervention summaries for fast Top-3 screening.

FIR-2 extends FIR-1 without reading future CGM or targets. Every feature is
computed only from the intervention prefix available to its target horizon.
The added action features use fixed, target-independent temporal kernels to
represent when insulin/carbohydrate events are likely to matter for the target.
"""
from __future__ import annotations

import numpy as np

from ml.forecasting.future_intervention_features import (
    HORIZONS,
    PREFIX_STEPS,
    STEP_MINUTES,
    SUMMARY_FEATURE_NAMES as FIR1_FEATURE_NAMES,
    build_future_intervention_summary,
)

BOLUS_PEAK_MIN = 60.0
BOLUS_WIDTH_MIN = 35.0
CARB_PEAK_MIN = 30.0
CARB_WIDTH_MIN = 20.0
RECENT_WINDOW_MIN = 30

FIR2_EXTRA_FEATURE_NAMES = (
    "log1p_bolus_action",
    "log1p_carb_action",
    "action_log_balance",
    "last_bolus_to_target_frac",
    "last_carb_to_target_frac",
    "log1p_bolus_last30",
    "log1p_carbs_last30",
    "bolus_carb_coevent_rate",
)

SUMMARY_FEATURE_NAMES = FIR1_FEATURE_NAMES + FIR2_EXTRA_FEATURE_NAMES


def _action_kernel(event_minutes: np.ndarray, horizon: int, peak: float, width: float):
    lag_to_target = float(horizon) - event_minutes
    active = lag_to_target >= 0.0
    weight = np.exp(-0.5 * ((lag_to_target - peak) / width) ** 2)
    return np.where(active, weight, 0.0).astype(np.float32)


def _last_event_to_target_frac(event: np.ndarray, horizon: int) -> np.ndarray:
    n, steps = event.shape
    any_event = event.any(axis=1)
    rev_idx = np.argmax(event[:, ::-1], axis=1)
    last_idx = steps - 1 - rev_idx
    last_minute = (last_idx + 1) * STEP_MINUTES
    frac = np.where(any_event, (float(horizon) - last_minute) / float(horizon), 1.0)
    return np.clip(frac, 0.0, 1.0).astype(np.float32)


def build_future_intervention_summary_v2(
    future_raw: np.ndarray,
    basal_mean: float,
    basal_std: float,
) -> np.ndarray:
    """Return FIR-2 summaries with shape ``(N, 4, 21)``."""
    f = np.asarray(future_raw, dtype=np.float32)
    if f.ndim != 3 or f.shape[1:] != (24, 6):
        raise ValueError(f"Expected future_raw shape (N, 24, 6), got {f.shape}")

    fir1 = build_future_intervention_summary(f, basal_mean=basal_mean, basal_std=basal_std)
    extra = np.empty((len(f), len(HORIZONS), len(FIR2_EXTRA_FEATURE_NAMES)), dtype=np.float32)

    for hi, (horizon, steps) in enumerate(zip(HORIZONS, PREFIX_STEPS)):
        p = f[:, :steps, :]
        bolus, bolus_missing = p[:, :, 2], p[:, :, 3]
        carbs, carbs_missing = p[:, :, 4], p[:, :, 5]

        bolus_event = (bolus_missing < 0.5) & (bolus > 0.0)
        carb_event = (carbs_missing < 0.5) & (carbs > 0.0)
        bolus_positive = np.where(bolus_event, bolus, 0.0)
        carbs_positive = np.where(carb_event, carbs, 0.0)

        event_minutes = (np.arange(steps, dtype=np.float32) + 1.0) * STEP_MINUTES
        bolus_w = _action_kernel(event_minutes, horizon, BOLUS_PEAK_MIN, BOLUS_WIDTH_MIN)
        carb_w = _action_kernel(event_minutes, horizon, CARB_PEAK_MIN, CARB_WIDTH_MIN)

        bolus_action_raw = np.sum(bolus_positive * bolus_w[None, :], axis=1)
        carb_action_raw = np.sum(carbs_positive * carb_w[None, :], axis=1)
        bolus_action = np.log1p(np.maximum(bolus_action_raw, 0.0))
        carb_action = np.log1p(np.maximum(carb_action_raw, 0.0))
        action_balance = carb_action - bolus_action

        last_bolus = _last_event_to_target_frac(bolus_event, horizon)
        last_carb = _last_event_to_target_frac(carb_event, horizon)

        recent_steps = min(steps, RECENT_WINDOW_MIN // STEP_MINUTES)
        bolus_last30 = np.log1p(
            np.maximum(bolus_positive[:, -recent_steps:].sum(axis=1), 0.0)
        )
        carbs_last30 = np.log1p(
            np.maximum(carbs_positive[:, -recent_steps:].sum(axis=1), 0.0)
        )
        coevent_rate = np.mean(bolus_event & carb_event, axis=1)

        extra[:, hi, :] = np.column_stack(
            [
                bolus_action,
                carb_action,
                action_balance,
                last_bolus,
                last_carb,
                bolus_last30,
                carbs_last30,
                coevent_rate,
            ]
        ).astype(np.float32)

    result = np.concatenate([fir1, extra], axis=2).astype(np.float32)
    if result.shape[2] != len(SUMMARY_FEATURE_NAMES):
        raise ValueError(f"Unexpected FIR-2 feature count: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite values generated in FIR-2 summary")
    return result
