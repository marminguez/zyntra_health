"""Zyntra V14.0: multi-horizon dataset + mandatory forecasting baselines.

Example:
  python -m ml.run_v14_baselines --data-dir /path/to/OhioT1DM --outdir ml/results/v14_0

No neural model is trained here. This establishes the performance floor V14
must beat before model development.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ohio_t1dm_loader import load_ohio_directory
from forecasting.baselines import persistence_predictions, linear_trend_predictions
from forecasting.dataset import add_forecast_targets, add_glucose_dynamics, describe_forecast_dataset
from forecasting.metrics import evaluate_forecasts, clinical_slice_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing OhioT1DM *-ws-*.xml files")
    parser.add_argument("--outdir", default="ml/results/v14_0")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_ohio_directory(args.data_dir)
    # Loader already contains causal dynamics, but recomputing here makes V14
    # independent from the legacy classification target implementation.
    data = add_glucose_dynamics(raw)
    data = add_forecast_targets(data)

    persistence = persistence_predictions(data)
    linear = linear_trend_predictions(data)
    persistence_metrics = evaluate_forecasts(data, persistence)
    linear_metrics = evaluate_forecasts(data, linear)
    persistence_metrics.insert(0, "model", "persistence")
    linear_metrics.insert(0, "model", "linear_trend")
    metrics = pd.concat([persistence_metrics, linear_metrics], ignore_index=True)
    metrics.to_csv(outdir / "v14_baseline_metrics.csv", index=False)

    slices = []
    for model_name, pred in (("persistence", persistence), ("linear_trend", linear)):
        for horizon in (30, 60, 90, 120):
            part = clinical_slice_metrics(data, pred, horizon)
            part.insert(0, "model", model_name)
            slices.append(part)
    pd.concat(slices, ignore_index=True).to_csv(outdir / "v14_clinical_slices.csv", index=False)

    summary = {
        "model": "v14_0_multi_horizon_baselines",
        "purpose": "establish leakage-safe persistence and linear-trend baselines before learned forecasting",
        "horizons_minutes": [30, 60, 90, 120],
        "dataset": describe_forecast_dataset(data),
        "clinical_status": "retrospective research only; not for clinical decision-making",
    }
    (outdir / "v14_baseline_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nBaseline metrics:")
    print(metrics.to_string(index=False))
    print(f"\nArtifacts written to {outdir}")


if __name__ == "__main__":
    main()
