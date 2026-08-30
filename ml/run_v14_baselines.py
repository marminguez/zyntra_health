"""Zyntra V14.0: bounded-memory MetaboNet forecasting baselines.

Uses PyArrow streaming over train.parquet only. test.parquet remains untouched
as a future holdout. No neural model is trained in V14.0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from forecasting.streaming_baselines import evaluate_metabonet_train_streaming


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing MetaboNet train.parquet")
    parser.add_argument("--outdir", default="ml/results/v14_0")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Rows per bounded PyArrow batch")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics, slices, source = evaluate_metabonet_train_streaming(
        args.data_dir,
        batch_size=args.batch_size,
    )
    metrics.to_csv(outdir / "v14_baseline_metrics.csv", index=False)
    slices.to_csv(outdir / "v14_clinical_slices.csv", index=False)

    summary = {
        "model": "v14_0_metabonet_multi_horizon_baselines",
        "purpose": "bounded-memory persistence and linear-trend baselines before learned forecasting",
        "horizons_minutes": [30, 60, 90, 120],
        "source": source,
        "clinical_status": "retrospective research only; not for clinical decision-making",
    }
    (outdir / "v14_baseline_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nV14.0 source summary:")
    print(json.dumps(source, indent=2))
    print("\nBaseline metrics:")
    print(metrics.to_string(index=False))
    print(f"\nArtifacts written to {outdir}")


if __name__ == "__main__":
    main()
