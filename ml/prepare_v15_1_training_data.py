"""Build V15.1 training data: V14.1 history plus known future insulin.

The split, seed, history, horizons, caps and historical features match V14.1.
The sole experimental addition is a 24-step future-insulin sequence t+5..t+120.
Future CGM is used only as the supervised target at the four registered horizons;
it is never included in model inputs.
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
from prepare_v14_training_data import (
    BASE_FEATURES, FEATURE_NAMES, HISTORY_MINUTES, HORIZONS, SEED, SEQ_LEN,
    STEP_MINUTES, _feature_vector, _num, _reservoir_add,
)

FUTURE_STEPS = 120 // STEP_MINUTES
COLUMNS = ["id", "source_file", "date", *BASE_FEATURES, "subject_split_across_traintest"]


def _write_subject(outdir: Path, split: str, source: str, pid: str, reservoir: list) -> int:
    if not reservoir:
        return 0
    split_dir = outdir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    safe_source = "".join(c if c.isalnum() or c in "-_" else "_" for c in source)
    safe_pid = "".join(c if c.isalnum() or c in "-_" else "_" for c in pid)
    path = split_dir / f"{safe_source}__{safe_pid}.npz"
    np.savez_compressed(
        path,
        x=np.stack([r[0] for r in reservoir]).astype(np.float32),
        future_insulin=np.stack([r[1] for r in reservoir]).astype(np.float32),
        y=np.stack([r[2] for r in reservoir]).astype(np.float32),
        timestamp=np.asarray([str(r[3]) for r in reservoir]),
    )
    return len(reservoir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", default="ml/data/v15_1")
    p.add_argument("--train-windows-per-patient", type=int, default=24)
    p.add_argument("--val-windows-per-patient", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=25000)
    p.add_argument("--memory-limit", default="2GB")
    args = p.parse_args()

    data_path = Path(args.data_dir) / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.data_dir) / ".v15_1_duckdb_tmp"; tmp.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(data_path)
    parquet_sql_path = str(data_path).replace("'", "''")
    tmp_sql = str(tmp).replace("'", "''")
    con = duckdb.connect(str(tmp / "v15_1_prepare.duckdb"))
    con.execute(f"SET temp_directory='{tmp_sql}'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute("SET preserve_insertion_order=false")

    select_cols = ", ".join(f'"{c}"' for c in COLUMNS)
    # Do NOT filter CGM here: future insulin rows must remain visible even when
    # future CGM is missing. Valid historical/target CGM is checked explicitly.
    query = f"""
        SELECT {select_cols}
        FROM read_parquet('{parquet_sql_path}')
        WHERE id IS NOT NULL AND source_file IS NOT NULL AND date IS NOT NULL
        ORDER BY source_file, id, date
    """

    print(f"V15.1 dataset preparation from {pf.metadata.num_rows:,} parquet rows")
    print(f"history={SEQ_LEN}x{len(FEATURE_NAMES)}; future insulin={FUTURE_STEPS}x2; targets={HORIZONS}")
    print("Controlled change vs V14.5: future insulin only; no future CGM input")

    reader = con.execute(query).fetch_record_batch(rows_per_batch=args.batch_size)
    rng = random.Random(SEED)
    current_key = None; current_split = None; overlap = False
    glucose_history = {}; feature_by_time = {}; glucose_by_time = {}; insulin_by_time = {}
    history = deque(); reservoir = []; candidates_seen = 0
    stats = {"train_subjects":0,"validation_subjects":0,"overlap_subjects":0,
             "train_windows":0,"validation_windows":0,"rows_scanned":0}

    def flush_subject():
        nonlocal reservoir, candidates_seen
        if current_key is None: return
        source,pid=current_key
        if overlap or current_split == "metabonet_overlap_excluded": stats["overlap_subjects"] += 1
        else:
            n=_write_subject(outdir,current_split,source,pid,reservoir)
            stats[f"{current_split}_subjects"] += 1; stats[f"{current_split}_windows"] += n
        reservoir=[]; candidates_seen=0

    for batch_idx,batch in enumerate(reader,start=1):
        frame=batch.to_pandas(); stats["rows_scanned"] += len(frame)
        for r in frame.itertuples(index=False):
            source,pid=str(r.source_file),str(r.id); key=(source,pid)
            if key != current_key:
                flush_subject(); current_key=key
                overlap=bool(r.subject_split_across_traintest) if r.subject_split_across_traintest is not None else False
                current_split=development_split(source,pid,overlap)
                glucose_history={}; feature_by_time={}; glucose_by_time={}; insulin_by_time={}; history=deque()

            ts=pd.Timestamp(r.date)
            vals={name:_num(getattr(r,name)) for name in BASE_FEATURES}
            insulin_by_time[ts]=vals["insulin"]
            g=vals["CGM"]
            if np.isfinite(g):
                feat=_feature_vector(ts,vals,glucose_history)
                history.append(ts); glucose_history[ts]=g; feature_by_time[ts]=feat; glucose_by_time[ts]=g

            anchor=ts-pd.Timedelta(minutes=120)
            if not overlap and anchor in glucose_by_time:
                hist_start=anchor-pd.Timedelta(minutes=HISTORY_MINUTES-STEP_MINUTES)
                hist_times=pd.date_range(hist_start,anchor,freq=f"{STEP_MINUTES}min")
                target_times=[anchor+pd.Timedelta(minutes=h) for h in HORIZONS]
                future_times=pd.date_range(anchor+pd.Timedelta(minutes=STEP_MINUTES),
                                           anchor+pd.Timedelta(minutes=120),freq=f"{STEP_MINUTES}min")
                if (len(hist_times)==SEQ_LEN and all(t in feature_by_time for t in hist_times)
                        and all(t in glucose_by_time for t in target_times)
                        and len(future_times)==FUTURE_STEPS and all(t in insulin_by_time for t in future_times)):
                    x=np.stack([feature_by_time[t] for t in hist_times]).astype(np.float32)
                    fi=[]
                    for t in future_times:
                        v=insulin_by_time[t]; missing=0.0 if np.isfinite(v) else 1.0
                        fi.append([0.0 if missing else v, missing])
                    fi=np.asarray(fi,dtype=np.float32)
                    y=np.asarray([glucose_by_time[t] for t in target_times],dtype=np.float32)
                    candidates_seen += 1
                    limit=args.train_windows_per_patient if current_split=="train" else args.val_windows_per_patient
                    _reservoir_add(reservoir,(x,fi,y,anchor),candidates_seen,limit,rng)

            cutoff=ts-pd.Timedelta(minutes=HISTORY_MINUTES+120)
            for store in (glucose_history,feature_by_time,glucose_by_time,insulin_by_time):
                for old in [k for k in store.keys() if k < cutoff]: store.pop(old,None)
            while history and history[0] < cutoff: history.popleft()

        if batch_idx % 20 == 0:
            pct=min(stats["rows_scanned"]/max(pf.metadata.num_rows,1)*100,100)
            print(f"  scanned {stats['rows_scanned']:,}/{pf.metadata.num_rows:,} rows ({pct:.1f}%)",flush=True)

    flush_subject(); con.close()
    metadata={"version":"v15.1","seed":SEED,"history_minutes":HISTORY_MINUTES,
              "sequence_length":SEQ_LEN,"sample_minutes":STEP_MINUTES,"horizons_minutes":list(HORIZONS),
              "historical_features":list(FEATURE_NAMES),"future_features":["insulin","insulin_missing"],
              "future_steps":FUTURE_STEPS,"future_window":"t+5..t+120",
              "future_cgm_as_input":False,
              "split_policy":"same deterministic patient-level policy as V14.1; MetaboNet overlap excluded",
              "train_windows_per_patient_cap":args.train_windows_per_patient,
              "validation_windows_per_patient_cap":args.val_windows_per_patient,**stats}
    (outdir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("\nV15.1 dataset ready:"); print(json.dumps(metadata,indent=2)); print(f"Artifacts written to {outdir}")

if __name__ == "__main__": main()
