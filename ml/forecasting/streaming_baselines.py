"""Streaming V14.0 baseline evaluation for large MetaboNet parquet files.

Reads only id/source/date/CGM in bounded PyArrow batches and keeps at most 120
minutes of history per subject. This avoids materializing train.parquet in RAM.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HORIZONS = (30, 60, 90, 120)
STABLE_BAND = 5.0
COLUMNS = ["id", "source_file", "date", "CGM"]


@dataclass
class Agg:
    n: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    ard_sum: float = 0.0
    direction_correct: int = 0

    def add(self, current: float, target: float, pred: float) -> None:
        if not (math.isfinite(current) and math.isfinite(target) and math.isfinite(pred)):
            return
        err = pred - target
        self.n += 1
        self.abs_sum += abs(err)
        self.sq_sum += err * err
        self.ard_sum += abs(err) / max(abs(target), 1e-6)
        td = target - current
        pd_ = pred - current
        true_dir = 1 if td > STABLE_BAND else -1 if td < -STABLE_BAND else 0
        pred_dir = 1 if pd_ > STABLE_BAND else -1 if pd_ < -STABLE_BAND else 0
        self.direction_correct += int(true_dir == pred_dir)

    def row(self) -> dict:
        if not self.n:
            return {"n": 0, "mae": np.nan, "rmse": np.nan, "mard": np.nan, "direction_accuracy": np.nan}
        return {
            "n": self.n,
            "mae": self.abs_sum / self.n,
            "rmse": math.sqrt(self.sq_sum / self.n),
            "mard": self.ard_sum / self.n * 100.0,
            "direction_accuracy": self.direction_correct / self.n,
        }


def _slice_name(target: float) -> list[str]:
    names = []
    if target < 70:
        names.append("hypoglycemia_<70")
    if 70 <= target <= 180:
        names.append("target_70_180")
    if target > 180:
        names.append("hyperglycemia_>180")
    if target > 250:
        names.append("severe_hyper_>250")
    return names


def evaluate_metabonet_train_streaming(
    data_dir: str | Path,
    batch_size: int = 100_000,
    progress_every_batches: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = Path(data_dir) / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected MetaboNet training file not found: {path}")

    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema_arrow.names)
    missing = [c for c in COLUMNS if c not in schema_names]
    if missing:
        raise ValueError(f"train.parquet is missing required columns: {missing}")

    metrics = {(model, h): Agg() for model in ("persistence", "linear_trend") for h in HORIZONS}
    slices = {(model, h, s): Agg() for model in ("persistence", "linear_trend") for h in HORIZONS for s in (
        "hypoglycemia_<70", "target_70_180", "hyperglycemia_>180", "severe_hyper_>250"
    )}

    # Per subject: timestamp -> (glucose, delta15). deque supports bounded pruning.
    history: dict[tuple[str, str], dict[pd.Timestamp, tuple[float, float | None]]] = defaultdict(dict)
    order: dict[tuple[str, str], deque[pd.Timestamp]] = defaultdict(deque)
    last_ts: dict[tuple[str, str], pd.Timestamp] = {}
    subjects: set[tuple[str, str]] = set()
    datasets: set[str] = set()

    rows = 0
    valid_rows = 0
    glucose_min = math.inf
    glucose_max = -math.inf
    hypo_rows = 0
    hyper_rows = 0

    print(f"Streaming {path.name}: {pf.metadata.num_rows:,} parquet rows, {pf.num_row_groups} row groups")
    print(f"Batch size: {batch_size:,} rows; test.parquet remains excluded")

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=COLUMNS), start=1):
        frame = batch.to_pandas()
        rows += len(frame)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["CGM"] = pd.to_numeric(frame["CGM"], errors="coerce")
        frame = frame.dropna(subset=COLUMNS)

        for r in frame.itertuples(index=False):
            source = str(r.source_file)
            pid = str(r.id)
            ts = pd.Timestamp(r.date)
            glucose = float(r.CGM)
            if not math.isfinite(glucose):
                continue

            key = (source, pid)
            subjects.add(key)
            datasets.add(source)
            valid_rows += 1
            glucose_min = min(glucose_min, glucose)
            glucose_max = max(glucose_max, glucose)
            hypo_rows += int(glucose < 70)
            hyper_rows += int(glucose > 180)

            prev_ts = last_ts.get(key)
            if prev_ts is not None and ts < prev_ts:
                raise ValueError(
                    f"train.parquet is not time-ordered within subject {key}: {ts} < {prev_ts}. "
                    "Streaming evaluation stopped rather than silently producing invalid horizons."
                )
            if ts in history[key]:
                raise ValueError(f"Duplicate timestamp for subject {key}: {ts}")

            hmap = history[key]
            g15 = hmap.get(ts - pd.Timedelta(minutes=15))
            delta15 = glucose - g15[0] if g15 is not None else None

            # Current glucose is the exact future target for origins t-h.
            for horizon in HORIZONS:
                origin = hmap.get(ts - pd.Timedelta(minutes=horizon))
                if origin is None:
                    continue
                origin_glucose, origin_delta15 = origin
                pred_persistence = origin_glucose
                metrics[("persistence", horizon)].add(origin_glucose, glucose, pred_persistence)
                for s in _slice_name(glucose):
                    slices[("persistence", horizon, s)].add(origin_glucose, glucose, pred_persistence)

                if origin_delta15 is not None:
                    pred_linear = float(np.clip(origin_glucose + (origin_delta15 / 15.0) * horizon, 40.0, 400.0))
                    metrics[("linear_trend", horizon)].add(origin_glucose, glucose, pred_linear)
                    for s in _slice_name(glucose):
                        slices[("linear_trend", horizon, s)].add(origin_glucose, glucose, pred_linear)

            hmap[ts] = (glucose, delta15)
            order[key].append(ts)
            last_ts[key] = ts

            cutoff = ts - pd.Timedelta(minutes=120)
            q = order[key]
            while q and q[0] < cutoff:
                old = q.popleft()
                hmap.pop(old, None)

        if batch_idx % progress_every_batches == 0:
            pct = min(rows / max(pf.metadata.num_rows, 1) * 100.0, 100.0)
            print(f"  processed {rows:,}/{pf.metadata.num_rows:,} rows ({pct:.1f}%)", flush=True)

    metric_rows = []
    for model in ("persistence", "linear_trend"):
        for horizon in HORIZONS:
            metric_rows.append({"model": model, "horizon_minutes": horizon, **metrics[(model, horizon)].row()})
    metrics_df = pd.DataFrame(metric_rows)

    slice_rows = []
    for model in ("persistence", "linear_trend"):
        for horizon in HORIZONS:
            for s in ("hypoglycemia_<70", "target_70_180", "hyperglycemia_>180", "severe_hyper_>250"):
                row = slices[(model, horizon, s)].row()
                slice_rows.append({"model": model, "horizon_minutes": horizon, "slice": s, **row})
    slices_df = pd.DataFrame(slice_rows)

    summary = {
        "file": path.name,
        "holdout_policy": "test.parquet excluded from V14 development",
        "parquet_rows": int(pf.metadata.num_rows),
        "rows_scanned": int(rows),
        "valid_cgm_rows": int(valid_rows),
        "subjects": int(len(subjects)),
        "datasets": int(len(datasets)),
        "glucose_min": None if glucose_min is math.inf else float(glucose_min),
        "glucose_max": None if glucose_max is -math.inf else float(glucose_max),
        "hypoglycemia_rows": int(hypo_rows),
        "hyperglycemia_rows": int(hyper_rows),
        "streaming_batch_size": int(batch_size),
    }
    return metrics_df, slices_df, summary
