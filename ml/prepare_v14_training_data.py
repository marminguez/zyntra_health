"""Build a bounded-memory V14.1 training dataset from MetaboNet train.parquet.

Design:
- 24 h history at 5-minute cadence (288 steps)
- exact +30/+60/+90/+120 minute CGM targets
- patient-level deterministic train/validation split
- subjects marked subject_split_across_traintest are excluded
- capped windows per patient to prevent Loop / long-record subjects dominating
- writes one compressed NPZ shard per subject, so RAM stays bounded

V14.1 pilot defaults deliberately sample a modest number of windows per subject.
They can be increased later after the first end-to-end training result.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from forecasting.splits_v14 import development_split

SEED = 42
HISTORY_MINUTES = 24 * 60
STEP_MINUTES = 5
SEQ_LEN = HISTORY_MINUTES // STEP_MINUTES  # 288
HORIZONS = (30, 60, 90, 120)

# First V14.1 metabolic feature set. Missingness indicators are explicit.
BASE_FEATURES = ("CGM", "basal", "bolus", "insulin", "carbs")
FEATURE_NAMES = (
    "glucose",
    "glucose_delta_5m",
    "glucose_delta_15m",
    "glucose_delta_30m",
    "glucose_acceleration_15m",
    "basal",
    "basal_missing",
    "bolus",
    "bolus_missing",
    "insulin",
    "insulin_missing",
    "carbs",
    "carbs_missing",
    "hour_sin",
    "hour_cos",
)
COLUMNS = ["id", "source_file", "date", *BASE_FEATURES, "subject_split_across_traintest"]


def _num(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _feature_vector(ts: pd.Timestamp, values: dict[str, float], glucose_history: dict[pd.Timestamp, float]) -> np.ndarray:
    g = values["CGM"]
    g5 = glucose_history.get(ts - pd.Timedelta(minutes=5), np.nan)
    g15 = glucose_history.get(ts - pd.Timedelta(minutes=15), np.nan)
    g30 = glucose_history.get(ts - pd.Timedelta(minutes=30), np.nan)
    g_prev30 = glucose_history.get(ts - pd.Timedelta(minutes=30), np.nan)
    g_prev15 = glucose_history.get(ts - pd.Timedelta(minutes=15), np.nan)

    d5 = g - g5 if np.isfinite(g5) else 0.0
    d15 = g - g15 if np.isfinite(g15) else 0.0
    d30 = g - g30 if np.isfinite(g30) else 0.0
    prev_d15 = g_prev15 - g_prev30 if np.isfinite(g_prev15) and np.isfinite(g_prev30) else 0.0
    acceleration = d15 - prev_d15

    hour = ts.hour + ts.minute / 60.0
    out = [g, d5, d15, d30, acceleration]
    for name in ("basal", "bolus", "insulin", "carbs"):
        x = values[name]
        missing = 0.0 if np.isfinite(x) else 1.0
        out.extend([0.0 if missing else x, missing])
    out.extend([
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
    ])
    return np.asarray(out, dtype=np.float32)


def _reservoir_add(reservoir: list, item, seen: int, limit: int, rng: random.Random) -> None:
    if len(reservoir) < limit:
        reservoir.append(item)
    else:
        j = rng.randrange(seen)
        if j < limit:
            reservoir[j] = item


def _write_subject(outdir: Path, split: str, source: str, pid: str, reservoir: list) -> int:
    if not reservoir:
        return 0
    split_dir = outdir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    safe_source = "".join(c if c.isalnum() or c in "-_" else "_" for c in source)
    safe_pid = "".join(c if c.isalnum() or c in "-_" else "_" for c in pid)
    path = split_dir / f"{safe_source}__{safe_pid}.npz"
    x = np.stack([r[0] for r in reservoir]).astype(np.float32)
    y = np.stack([r[1] for r in reservoir]).astype(np.float32)
    ts = np.asarray([str(r[2]) for r in reservoir])
    np.savez_compressed(path, x=x, y=y, timestamp=ts)
    return len(reservoir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", default="ml/data/v14_1")
    p.add_argument("--train-windows-per-patient", type=int, default=24)
    p.add_argument("--val-windows-per-patient", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=25000)
    p.add_argument("--memory-limit", default="2GB")
    args = p.parse_args()

    data_path = Path(args.data_dir) / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.data_dir) / ".v14_duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(data_path)
    parquet_sql_path = str(data_path).replace("'", "''")
    tmp_sql = str(tmp).replace("'", "''")
    con = duckdb.connect(str(tmp / "v14_prepare.duckdb"))
    con.execute(f"SET temp_directory='{tmp_sql}'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute("SET preserve_insertion_order=false")

    select_cols = ", ".join(f'"{c}"' for c in COLUMNS)
    query = f"""
        SELECT {select_cols}
        FROM read_parquet('{parquet_sql_path}')
        WHERE id IS NOT NULL
          AND source_file IS NOT NULL
          AND date IS NOT NULL
          AND CGM IS NOT NULL
        ORDER BY source_file, id, date
    """

    print(f"V14.1 dataset preparation from {pf.metadata.num_rows:,} parquet rows")
    print(f"24h history = {SEQ_LEN} steps; targets = {HORIZONS}")
    print(f"train cap={args.train_windows_per_patient}, validation cap={args.val_windows_per_patient} windows/patient")
    print("MetaboNet overlap subjects are excluded")

    reader = con.execute(query).fetch_record_batch(rows_per_batch=args.batch_size)
    rng = random.Random(SEED)

    current_key = None
    current_split = None
    overlap = False
    history = deque()  # (timestamp, feature_vector)
    glucose_history: dict[pd.Timestamp, float] = {}
    feature_by_time: dict[pd.Timestamp, np.ndarray] = {}
    glucose_by_time: dict[pd.Timestamp, float] = {}
    reservoir: list = []
    candidates_seen = 0

    stats = {"train_subjects": 0, "validation_subjects": 0, "overlap_subjects": 0,
             "train_windows": 0, "validation_windows": 0, "rows_scanned": 0}

    def flush_subject():
        nonlocal reservoir, candidates_seen
        if current_key is None:
            return
        source, pid = current_key
        if overlap or current_split == "metabonet_overlap_excluded":
            stats["overlap_subjects"] += 1
        else:
            n = _write_subject(outdir, current_split, source, pid, reservoir)
            stats[f"{current_split}_subjects"] += 1
            stats[f"{current_split}_windows"] += n
        reservoir = []
        candidates_seen = 0

    for batch_idx, batch in enumerate(reader, start=1):
        frame = batch.to_pandas()
        stats["rows_scanned"] += len(frame)
        for r in frame.itertuples(index=False):
            source, pid = str(r.source_file), str(r.id)
            key = (source, pid)
            if key != current_key:
                flush_subject()
                current_key = key
                overlap = bool(r.subject_split_across_traintest) if r.subject_split_across_traintest is not None else False
                current_split = development_split(source, pid, overlap)
                history = deque()
                glucose_history = {}
                feature_by_time = {}
                glucose_by_time = {}

            ts = pd.Timestamp(r.date)
            vals = {name: _num(getattr(r, name)) for name in BASE_FEATURES}
            g = vals["CGM"]
            if not np.isfinite(g):
                continue

            feat = _feature_vector(ts, vals, glucose_history)
            history.append((ts, feat))
            glucose_history[ts] = g
            feature_by_time[ts] = feat
            glucose_by_time[ts] = g

            # Anchor is 120 minutes behind the newest observation, so every target is known.
            anchor = ts - pd.Timedelta(minutes=120)
            if not overlap and anchor in glucose_by_time:
                hist_start = anchor - pd.Timedelta(minutes=HISTORY_MINUTES - STEP_MINUTES)
                expected_times = pd.date_range(hist_start, anchor, freq=f"{STEP_MINUTES}min")
                if len(expected_times) == SEQ_LEN and all(t in feature_by_time for t in expected_times):
                    target_times = [anchor + pd.Timedelta(minutes=h) for h in HORIZONS]
                    if all(t in glucose_by_time for t in target_times):
                        x = np.stack([feature_by_time[t] for t in expected_times]).astype(np.float32)
                        y = np.asarray([glucose_by_time[t] for t in target_times], dtype=np.float32)
                        candidates_seen += 1
                        limit = args.train_windows_per_patient if current_split == "train" else args.val_windows_per_patient
                        _reservoir_add(reservoir, (x, y, anchor), candidates_seen, limit, rng)

            # Keep only what is needed for 24 h history + 120 min future.
            cutoff = ts - pd.Timedelta(minutes=HISTORY_MINUTES + 120)
            while history and history[0][0] < cutoff:
                old_ts, _ = history.popleft()
                glucose_history.pop(old_ts, None)
                feature_by_time.pop(old_ts, None)
                glucose_by_time.pop(old_ts, None)

        if batch_idx % 20 == 0:
            pct = min(stats["rows_scanned"] / max(pf.metadata.num_rows, 1) * 100.0, 100.0)
            print(f"  scanned {stats['rows_scanned']:,}/{pf.metadata.num_rows:,} rows ({pct:.1f}%)", flush=True)

    flush_subject()
    con.close()

    metadata = {
        "version": "v14.1",
        "seed": SEED,
        "history_minutes": HISTORY_MINUTES,
        "sequence_length": SEQ_LEN,
        "sample_minutes": STEP_MINUTES,
        "horizons_minutes": list(HORIZONS),
        "features": list(FEATURE_NAMES),
        "split_policy": "deterministic patient-level; MetaboNet subject_split_across_traintest excluded",
        "train_windows_per_patient_cap": args.train_windows_per_patient,
        "validation_windows_per_patient_cap": args.val_windows_per_patient,
        **stats,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nV14.1 dataset ready:")
    print(json.dumps(metadata, indent=2))
    print(f"Artifacts written to {outdir}")

if __name__ == "__main__":
    main()
