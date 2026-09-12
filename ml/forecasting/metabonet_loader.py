"""Memory-conscious loader for the public MetaboNet V14 training parquet.

V14 development uses ``train.parquet`` only. ``test.parquet`` is deliberately
left untouched as a future holdout. For V14.0 baselines we read only the four
columns required to construct patient-safe glucose trajectories.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = ("id", "source_file", "date", "CGM")


def load_metabonet_train(data_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"MetaboNet directory not found: {root}")

    path = root / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected MetaboNet training file not found: {path}")

    # Critical for local laptops/WSL: do not materialize the ~1 GB parquet with
    # every feature when V14.0 only needs identity, time and CGM.
    df = pd.read_parquet(path, columns=list(REQUIRED))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required MetaboNet columns: {missing}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["CGM"] = pd.to_numeric(data["CGM"], errors="coerce", downcast="float")
    data = data.dropna(subset=["id", "source_file", "date", "CGM"]).copy()

    # id is not assumed to be globally unique across source datasets.
    data["patient_id"] = data["id"].astype(str)
    data["glucose"] = data["CGM"].astype("float32")
    data = data[["source_file", "patient_id", "date", "glucose"]]
    data = data.sort_values(["source_file", "patient_id", "date"]).reset_index(drop=True)

    dup = data.duplicated(["source_file", "patient_id", "date"], keep=False)
    if dup.any():
        n = int(dup.sum())
        raise ValueError(
            f"Found {n:,} duplicate (source_file, id, date) rows in train.parquet. "
            "Resolve duplicates before forecasting."
        )

    return data, [path.name]


def metabonet_summary(df: pd.DataFrame, files: list[str]) -> dict:
    return {
        "files": files,
        "holdout_policy": "test.parquet excluded from V14 development",
        "rows": int(len(df)),
        "subjects": int(df[["source_file", "patient_id"]].drop_duplicates().shape[0]),
        "datasets": int(df["source_file"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "cgm_missing_after_load": int(df["glucose"].isna().sum()),
    }
