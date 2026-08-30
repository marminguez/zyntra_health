"""Loader for the public MetaboNet training parquet files used by Zyntra V14.

MetaboNet uses a standardized 5-minute schema. This loader reads every parquet
in the provided train directory, concatenates them, preserves source/patient
boundaries, and maps the official CGM column to Zyntra's internal ``glucose``
name without discarding the original CGM column.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = ("id", "source_file", "date", "CGM")


def load_metabonet_train(data_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"MetaboNet directory not found: {root}")

    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {root}")

    frames = []
    for path in files:
        df = pd.read_parquet(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required MetaboNet columns: {missing}")
        df = df.copy()
        df["_metabonet_file"] = path.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["CGM"] = pd.to_numeric(data["CGM"], errors="coerce")
    data = data.dropna(subset=["id", "source_file", "date", "CGM"]).copy()

    # Keep source_file in the sequence key because an id alone is not assumed to
    # be globally unique across source datasets.
    data["patient_id"] = data["id"].astype(str)
    data["glucose"] = data["CGM"].astype(float)
    data = data.sort_values(["source_file", "patient_id", "date"]).reset_index(drop=True)

    # Guard against duplicate rows before target shifting. Duplicate timestamps
    # would otherwise break the 5-minute horizon interpretation.
    dup = data.duplicated(["source_file", "patient_id", "date"], keep=False)
    if dup.any():
        n = int(dup.sum())
        raise ValueError(
            f"Found {n:,} duplicate (source_file, id, date) rows across the parquet files. "
            "Resolve duplicate train partitions before forecasting."
        )

    return data, [p.name for p in files]


def metabonet_summary(df: pd.DataFrame, files: list[str]) -> dict:
    return {
        "files": files,
        "rows": int(len(df)),
        "subjects": int(df[["source_file", "patient_id"]].drop_duplicates().shape[0]),
        "datasets": int(df["source_file"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "cgm_missing_after_load": int(df["glucose"].isna().sum()),
    }
