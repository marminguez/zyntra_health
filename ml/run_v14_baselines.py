"""Zyntra V14.0: MetaboNet multi-horizon forecasting baselines.

Example on Windows:
  python -m ml.run_v14_baselines --data-dir "C:\\zyntra\\dataset\\Metabonet\\train" --outdir ml/results/v14_0

Reads all parquet files in the MetaboNet train directory. No neural model is
trained here: this establishes the performance floor V14 must beat.
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

from forecasting.baselines import persistence_predictions, linear_trend_predictions
from forecasting.dataset import add_forecast_targets, add_glucose_dynamics, describe_forecast_dataset
from forecasting.metabonet_loader import load_metabonet_train, metabonet_summary
from forecasting.metrics import evaluate_forecasts, clinical_slice_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing MetaboNet train .parquet files")
    parser.add_argument("--outdir", default="ml/results/v14_0")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw, files = load_metabonet_train(args.data_dir)
    print("Loaded MetaboNet train files:")
    for name in files:
        print(f"  - {name}")
    print(json.dumps(metabonet_summary(raw, files), indent=2))

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
        "model": "v14_0_metabonet_multi_horizon_baselines",
        "purpose": "establish leakage-safe MetaboNet persistence and linear-trend baselines before learned forecasting",
        "horizons_minutes": [30, 60, 90, 120],
        "source": metabonet_summary(raw, files),
        "forecast_dataset": describe_forecast_dataset(data),
        "clinical_status": "retrospective research only; not for clinical decision-making",
    }
    (outdir / "v14_baseline_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nForecast dataset:")
    print(json.dumps(summary["forecast_dataset"], indent=2))
    print("\nBaseline metrics:")
    print(metrics.to_string(index=False))
    print(f"\nArtifacts written to {outdir}")


if __name__ == "__main__":
    main()
