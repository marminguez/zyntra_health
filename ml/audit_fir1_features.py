"""Sanity/leakage audit for FIR-1 future intervention summaries.

Runs only on frozen Scale-50 validation shards and frozen V14.1 normalization.
No model training or target fitting is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ml.forecasting.future_intervention_features import (
    HORIZONS,
    PREFIX_STEPS,
    SUMMARY_FEATURE_NAMES,
    build_future_intervention_summary,
)

CHANNELS = (2, 3, 4, 5, 6, 7)


def load_validation(root: Path):
    fs, ys = [], []
    for p in sorted((root / "validation").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            fs.append(z["future_known"].astype(np.float32)[:, :, CHANNELS])
            ys.append(z["y"].astype(np.float32))
    if not fs:
        raise ValueError("No validation shards")
    return np.concatenate(fs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="ml/data/v15_4_scale50")
    ap.add_argument("--v14-1-dir", default="ml/results/v14_1")
    a = ap.parse_args()

    future_raw, y = load_validation(Path(a.data_dir))
    with np.load(Path(a.v14_1_dir) / "normalization.npz") as z:
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)

    s = build_future_intervention_summary(
        future_raw,
        basal_mean=float(mean[5]),
        basal_std=float(std[5]),
    )

    assert s.shape == (len(y), 4, len(SUMMARY_FEATURE_NAMES))
    assert np.isfinite(s).all()

    # Prefix-causality audit: perturb only steps after each horizon and prove the
    # summary for that horizon is exactly unchanged.
    rng = np.random.default_rng(12345)
    for hi, steps in enumerate(PREFIX_STEPS):
        perturbed = future_raw.copy()
        if steps < future_raw.shape[1]:
            tail = perturbed[:, steps:, :]
            # Only therapy/carb values and masks are modified; still no CGM exists.
            tail[:] = rng.normal(3.0, 2.0, size=tail.shape).astype(np.float32)
            perturbed[:, steps:, 1::2] = (rng.random(perturbed[:, steps:, 1::2].shape) > 0.5).astype(np.float32)
        sp = build_future_intervention_summary(
            perturbed,
            basal_mean=float(mean[5]),
            basal_std=float(std[5]),
        )
        if not np.array_equal(s[:, hi, :], sp[:, hi, :]):
            raise AssertionError(f"Horizon +{HORIZONS[hi]} summary changed from post-target perturbation")

    print("=== FIR-1 FEATURE AUDIT ===")
    print(f"validation windows: {len(y):,}")
    print(f"summary shape:      {s.shape}")
    print(f"features/horizon:   {len(SUMMARY_FEATURE_NAMES)}")
    print("future CGM input:   False by construction")
    print("post-target prefix leakage: NONE")
    print("all values finite: True")
    print("\nFeature names:")
    for name in SUMMARY_FEATURE_NAMES:
        print(" -", name)

    print("\nPer-horizon feature means/stds:")
    for hi, h in enumerate(HORIZONS):
        print(f"\n+{h} min")
        for fi, name in enumerate(SUMMARY_FEATURE_NAMES):
            v = s[:, hi, fi].astype(float)
            print(f"  {name:32s} mean={v.mean():10.5f} std={v.std():10.5f} min={v.min():10.5f} max={v.max():10.5f}")

    print("\nPASS — FIR-1 summaries are finite, horizon-prefix causal and CGM-free.")


if __name__ == "__main__":
    main()
