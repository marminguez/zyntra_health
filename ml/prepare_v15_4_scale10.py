"""Build resumable V15.4 Scale-10 training data with frozen validation.

Controlled experiment:
- training cap increases from 24 to 240 windows/patient;
- split/history/targets/features remain V15.1/V15.4 compatible;
- future-known cache contains insulin/basal/bolus/carbs + missing masks;
- validation shards are copied byte-for-byte from frozen ml/data/v15_master;
- no future CGM is used as an input.

Resume semantics:
- a checkpoint is written only after a whole subject has been completed;
- on restart, the SQL scan begins after the last completed (source_file, id);
- if interrupted mid-subject, only that subject is recomputed;
- reservoir sampling uses a deterministic subject-specific RNG derived from seed 42,
  so resumed and uninterrupted runs produce the same subject shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from forecasting.splits_v14 import development_split
from prepare_v14_training_data import (
    BASE_FEATURES,
    FEATURE_NAMES,
    HISTORY_MINUTES,
    HORIZONS,
    SEED,
    SEQ_LEN,
    STEP_MINUTES,
    _feature_vector,
    _num,
    _reservoir_add,
)

FUTURE_STEPS = 24
FUTURE_NUMERIC = ("insulin", "basal", "bolus", "carbs")
COLUMNS = ["id", "source_file", "date", *BASE_FEATURES, "subject_split_across_traintest"]
CHECKPOINT_VERSION = 1


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def subject_rng(source: str, pid: str) -> random.Random:
    payload = f"{SEED}|{source}|{pid}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def write_subject(outdir: Path, source: str, pid: str, reservoir: list) -> int:
    if not reservoir:
        return 0
    train_dir = outdir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    path = train_dir / f"{safe_name(source)}__{safe_name(pid)}.npz"
    np.savez_compressed(
        path,
        x=np.stack([r[0] for r in reservoir]).astype(np.float32),
        future_known=np.stack([r[1] for r in reservoir]).astype(np.float32),
        y=np.stack([r[2] for r in reservoir]).astype(np.float32),
        timestamp=np.asarray([str(r[3]) for r in reservoir]),
    )
    return len(reservoir)


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise RuntimeError(f"Unsupported checkpoint version in {path}")
    return obj


def save_checkpoint(path: Path, last_source: str, last_pid: str, stats: dict) -> None:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "last_completed_source": last_source,
        "last_completed_id": last_pid,
        "stats": stats,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--frozen-master", default="ml/data/v15_master")
    ap.add_argument("--outdir", default="ml/data/v15_4_scale10")
    ap.add_argument("--train-windows-per-patient", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=25000)
    ap.add_argument("--memory-limit", default="3GB")
    ap.add_argument("--reset", action="store_true", help="Delete prior Scale-10 train/checkpoint and start from zero")
    a = ap.parse_args()

    data = Path(a.train_parquet)
    frozen = Path(a.frozen_master)
    out = Path(a.outdir)
    checkpoint_path = out / "checkpoint.json"
    out.mkdir(parents=True, exist_ok=True)

    if not data.exists():
        raise FileNotFoundError(data)
    if not (frozen / "validation").exists():
        raise FileNotFoundError(frozen / "validation")
    if a.train_windows_per_patient != 240:
        print(f"WARNING: non-registered cap {a.train_windows_per_patient}; Scale-10 reference is 240")

    if a.reset:
        shutil.rmtree(out / "train", ignore_errors=True)
        checkpoint_path.unlink(missing_ok=True)
        (out / "metadata.json").unlink(missing_ok=True)
        print("Reset requested: prior Scale-10 train/checkpoint removed")

    checkpoint = load_checkpoint(checkpoint_path)
    train_dir = out / "train"
    if checkpoint is None and train_dir.exists() and any(train_dir.glob("*.npz")):
        raise RuntimeError(
            f"{train_dir} contains shards but no checkpoint. Use --reset to start clean "
            "or restore checkpoint.json before continuing."
        )

    # Freeze validation exactly by copying existing NPZ bytes.
    validation_dir = out / "validation"
    validation_dir.mkdir(exist_ok=True)
    for p in sorted((frozen / "validation").glob("*.npz")):
        target = validation_dir / p.name
        if not target.exists():
            shutil.copy2(p, target)
    frozen_val = sum(
        len(np.load(p, allow_pickle=False)["y"])
        for p in sorted(validation_dir.glob("*.npz"))
    )

    stats = {
        "rows_scanned_this_run": 0,
        "train_subjects": 0,
        "train_windows": 0,
        "train_candidates_before_cap": 0,
        "validation_frozen_windows": frozen_val,
        "overlap_subjects": 0,
        "validation_subjects_skipped": 0,
        "completed_subjects": 0,
    }
    last_source = None
    last_pid = None
    if checkpoint:
        saved = checkpoint.get("stats", {})
        for k in stats:
            if k != "rows_scanned_this_run" and k in saved:
                stats[k] = saved[k]
        last_source = str(checkpoint["last_completed_source"])
        last_pid = str(checkpoint["last_completed_id"])
        print(f"RESUME: continuing after {last_source} / {last_pid}")
        print(f"Already completed subjects: {stats['completed_subjects']:,}")
        print(f"Already written train windows: {stats['train_windows']:,}")
    else:
        print("Starting Scale-10 from the beginning")

    pf = pq.ParquetFile(data)
    tmp = data.parent / ".v15_scale10_duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect(str(tmp / "prepare.duckdb"))
    con.execute(f"SET temp_directory='{str(tmp).replace(chr(39), chr(39) * 2)}'")
    con.execute(f"SET memory_limit='{a.memory_limit}'")
    con.execute("SET preserve_insertion_order=false")

    ps = str(data).replace("'", "''")
    select_cols = ", ".join(f'"{c}"' for c in COLUMNS)
    resume_sql = ""
    if last_source is not None:
        ls = last_source.replace("'", "''")
        lp = last_pid.replace("'", "''")
        resume_sql = (
            f" AND (source_file > '{ls}' OR "
            f"(source_file = '{ls}' AND CAST(id AS VARCHAR) > '{lp}'))"
        )

    query = f"""
        SELECT {select_cols}
        FROM read_parquet('{ps}')
        WHERE id IS NOT NULL
          AND source_file IS NOT NULL
          AND date IS NOT NULL
          {resume_sql}
        ORDER BY source_file, CAST(id AS VARCHAR), date
    """
    reader = con.execute(query).fetch_record_batch(rows_per_batch=a.batch_size)

    key = None
    split = None
    overlap = False
    glucose_history = {}
    feature_by_time = {}
    glucose_by_time = {}
    future_rows = {}
    reservoir = []
    candidates_seen = 0
    rng = None

    def flush_subject() -> None:
        nonlocal reservoir, candidates_seen
        if key is None:
            return
        source, pid = key
        if overlap or split == "metabonet_overlap_excluded":
            stats["overlap_subjects"] += 1
        elif split == "train":
            stats["train_subjects"] += 1
            stats["train_candidates_before_cap"] += candidates_seen
            stats["train_windows"] += write_subject(out, source, pid, reservoir)
        elif split == "validation":
            # Validation is deliberately frozen and never regenerated here.
            stats["validation_subjects_skipped"] += 1
        stats["completed_subjects"] += 1
        save_checkpoint(checkpoint_path, source, pid, stats)
        if stats["completed_subjects"] % 25 == 0:
            print(
                f"checkpoint: {source}/{pid} | completed={stats['completed_subjects']:,} "
                f"| train_windows={stats['train_windows']:,}",
                flush=True,
            )
        reservoir = []
        candidates_seen = 0

    for batch_idx, batch in enumerate(reader, start=1):
        frame = batch.to_pandas()
        stats["rows_scanned_this_run"] += len(frame)
        for r in frame.itertuples(index=False):
            current = (str(r.source_file), str(r.id))
            if current != key:
                flush_subject()
                key = current
                overlap = (
                    bool(r.subject_split_across_traintest)
                    if r.subject_split_across_traintest is not None
                    else False
                )
                split = development_split(*current, overlap)
                rng = subject_rng(*current)
                glucose_history = {}
                feature_by_time = {}
                glucose_by_time = {}
                future_rows = {}

            ts = pd.Timestamp(r.date)
            vals = {name: _num(getattr(r, name)) for name in BASE_FEATURES}
            future_rows[ts] = vals
            g = vals["CGM"]
            if np.isfinite(g):
                feature_by_time[ts] = _feature_vector(ts, vals, glucose_history)
                glucose_history[ts] = g
                glucose_by_time[ts] = g

            anchor = ts - pd.Timedelta(minutes=120)
            if not overlap and split == "train" and anchor in glucose_by_time:
                hist_times = pd.date_range(
                    anchor - pd.Timedelta(minutes=HISTORY_MINUTES - STEP_MINUTES),
                    anchor,
                    freq=f"{STEP_MINUTES}min",
                )
                target_times = [anchor + pd.Timedelta(minutes=h) for h in HORIZONS]
                future_times = pd.date_range(
                    anchor + pd.Timedelta(minutes=STEP_MINUTES),
                    anchor + pd.Timedelta(minutes=120),
                    freq=f"{STEP_MINUTES}min",
                )
                if (
                    len(hist_times) == SEQ_LEN
                    and all(t in feature_by_time for t in hist_times)
                    and all(t in glucose_by_time for t in target_times)
                    and len(future_times) == FUTURE_STEPS
                    and all(t in future_rows for t in future_times)
                ):
                    x = np.stack([feature_by_time[t] for t in hist_times]).astype(np.float32)
                    fk = []
                    for t in future_times:
                        row = future_rows[t]
                        vector = []
                        for name in FUTURE_NUMERIC:
                            value = row[name]
                            missing = 0.0 if np.isfinite(value) else 1.0
                            vector.extend([0.0 if missing else value, missing])
                        fk.append(vector)
                    y = np.asarray([glucose_by_time[t] for t in target_times], dtype=np.float32)
                    candidates_seen += 1
                    _reservoir_add(
                        reservoir,
                        (x, np.asarray(fk, dtype=np.float32), y, anchor),
                        candidates_seen,
                        a.train_windows_per_patient,
                        rng,
                    )

            cutoff = ts - pd.Timedelta(minutes=HISTORY_MINUTES + 120)
            for store in (glucose_history, feature_by_time, glucose_by_time, future_rows):
                for old in [z for z in store if z < cutoff]:
                    store.pop(old, None)

        if batch_idx % 20 == 0:
            print(
                f"this run scanned {stats['rows_scanned_this_run']:,} rows "
                f"| completed subjects={stats['completed_subjects']:,} "
                f"| train windows={stats['train_windows']:,}",
                flush=True,
            )

    flush_subject()
    con.close()

    final_stats = dict(stats)
    final_stats["rows_scanned_this_run"] = stats["rows_scanned_this_run"]
    metadata = {
        "version": "v15.4-scale10-data",
        "controlled_change": "train windows/patient cap 24 -> 240 only",
        "seed": SEED,
        "reservoir_rng": "deterministic subject-specific SHA256(seed|source|id)",
        "train_windows_per_patient_cap": a.train_windows_per_patient,
        "validation_source": "byte copy of frozen v15_master validation",
        "future_channels": [
            v for c in FUTURE_NUMERIC for v in (c, f"{c}_missing")
        ],
        "future_cgm_input": False,
        "resumable": True,
        **final_stats,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n=== V15.4 SCALE-10 DATASET ===")
    print(json.dumps(metadata, indent=2))
    print(f"Artifacts: {out.resolve()}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
