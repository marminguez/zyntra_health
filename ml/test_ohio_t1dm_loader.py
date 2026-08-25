"""Lightweight validation for the OhioT1DM loader.

Run locally after placing the XML files in a directory:
python ml/test_ohio_t1dm_loader.py /path/to/ohio
"""
from pathlib import Path
import sys
import numpy as np

from ohio_t1dm_loader import load_ohio_directory, dataset_summary


def main(data_dir):
    df = load_ohio_directory(data_dir)
    summary = dataset_summary(df)
    assert df.p_id.nunique() >= 3, "Need multiple patients for patient-disjoint evaluation"
    assert set(["glucose","target_hypo","glucose_delta_5m","glucose_delta_15m","glucose_delta_30m"]).issubset(df.columns)
    assert df.target_hypo.sum() > 0, "Dataset contains no 30-minute hypo targets"
    assert (df.glucose < 70).sum() > 0, "Dataset contains no observed hypoglycemia"
    assert not df.duplicated(subset=["p_id", df.index.name]).any() if df.index.name in df.columns else True
    print(summary.to_string(index=False))
    print(f"Patients: {df.p_id.nunique()} | rows: {len(df)} | hypo samples: {(df.glucose < 70).sum()} | 30m positive targets: {df.target_hypo.sum()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python test_ohio_t1dm_loader.py DATA_DIR")
    main(Path(sys.argv[1]))
