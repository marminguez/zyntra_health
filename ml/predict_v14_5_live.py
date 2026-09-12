"""Generate a MetaboNet Live Leaderboard submission with frozen Zyntra V14.5.

No fitting or target access occurs here. The script:
- reuses the exact V14.1 15-feature construction;
- requires a strict 24 h / 288-step history at 5-minute cadence;
- applies the frozen V14.1 normalization used by V14.5;
- uses V14.5's pre-registered 50/50 absolute + reconstructed-delta fusion;
- preserves the official template row order and keys exactly.

Rows without a valid V14.1 history are reported and left unresolved by default.
Use --fallback persistence only if an explicit fallback policy is desired.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prepare_v14_training_data import (
    BASE_FEATURES,
    COLUMNS,
    FEATURE_NAMES,
    HISTORY_MINUTES,
    SEQ_LEN,
    STEP_MINUTES,
    _feature_vector,
    _num,
)

PRED_COLS = ("pred_30", "pred_60", "pred_90", "pred_120")
KEY_COLS = ("id", "source_file", "date")


def _load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        if "mean" in z and "std" in z:
            mean, std = z["mean"], z["std"]
        elif "x_mean" in z and "x_std" in z:
            mean, std = z["x_mean"], z["x_std"]
        else:
            raise KeyError(f"Unsupported normalization keys: {z.files}")
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if len(mean) != len(FEATURE_NAMES) or len(std) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} normalization features, got {len(mean)}")
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def _decode_v14_5(outputs, current_glucose: np.ndarray) -> np.ndarray:
    """Return fixed hybrid predictions in horizon order 30/60/90/120."""
    if isinstance(outputs, dict):
        names = list(outputs)
        abs_keys = [k for k in names if "abs" in k.lower()]
        delta_keys = [k for k in names if "delta" in k.lower()]
        if len(abs_keys) != 4 or len(delta_keys) != 4:
            raise ValueError(f"Could not identify 4 absolute and 4 delta heads: {names}")
        def hnum(k):
            for h in (30, 60, 90, 120):
                if str(h) in k:
                    return h
            return 999
        abs_vals = [np.asarray(outputs[k]).reshape(-1) for k in sorted(abs_keys, key=hnum)]
        delta_vals = [np.asarray(outputs[k]).reshape(-1) for k in sorted(delta_keys, key=hnum)]
    else:
        vals = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        if len(vals) != 8:
            raise ValueError(f"Expected 8 V14.5 outputs, got {len(vals)}")
        abs_vals = [np.asarray(v).reshape(-1) for v in vals[:4]]
        delta_vals = [np.asarray(v).reshape(-1) for v in vals[4:]]
    absolute = np.stack(abs_vals, axis=1).astype(np.float32)
    delta = np.stack(delta_vals, axis=1).astype(np.float32)
    reconstructed = current_glucose[:, None] + delta
    return (0.5 * absolute + 0.5 * reconstructed).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, help="MetaboNet test.parquet")
    p.add_argument("--template", required=True, help="Official live leaderboard template.parquet")
    p.add_argument("--model", default="ml/results/v14_5/best_model.keras")
    p.add_argument("--normalization", default="ml/results/v14_1/normalization.npz")
    p.add_argument("--output", default="ml/results/v14_5_live/submission_v14_5.parquet")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--read-batch-size", type=int, default=50000)
    p.add_argument("--memory-limit", default="3GB")
    p.add_argument("--fallback", choices=("none", "persistence"), default="none")
    args = p.parse_args()

    test_path = Path(args.test).resolve()
    template_path = Path(args.template).resolve()
    model_path = Path(args.model).resolve()
    norm_path = Path(args.normalization).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for path in (test_path, template_path, model_path, norm_path):
        if not path.exists():
            raise FileNotFoundError(path)

    template = pq.read_table(template_path).to_pandas()
    expected_cols = [*KEY_COLS, *PRED_COLS]
    if list(template.columns) != expected_cols:
        raise ValueError(f"Unexpected template columns: {list(template.columns)}")
    template["date"] = pd.to_datetime(template["date"], errors="raise")

    print(f"Template rows: {len(template):,}")
    print(f"Features: {len(FEATURE_NAMES)}; sequence length: {SEQ_LEN}")

    mean, std = _load_normalization(norm_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Map key -> exact template row. Use enumerate rather than a leading-underscore
    # namedtuple field because pandas renames such fields in itertuples().
    key_to_row: dict[tuple[str, str, pd.Timestamp], int] = {}
    for row_idx, r in enumerate(template.itertuples(index=False)):
        key = (str(r.source_file), str(r.id), pd.Timestamp(r.date))
        if key in key_to_row:
            raise ValueError(f"Duplicate template key: {key}")
        key_to_row[key] = row_idx

    preds = np.full((len(template), 4), np.nan, dtype=np.float32)
    persistence = np.full(len(template), np.nan, dtype=np.float32)

    tmp = output_path.parent / ".v14_5_live_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(tmp / "infer.duckdb"))
    con.execute(f"SET temp_directory='{str(tmp).replace(chr(39), chr(39)*2)}'")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute("SET preserve_insertion_order=false")

    select_cols = ", ".join(f'"{c}"' for c in COLUMNS if c != "subject_split_across_traintest")
    test_sql = str(test_path).replace("'", "''")
    query = f"""
        SELECT {select_cols}
        FROM read_parquet('{test_sql}')
        WHERE id IS NOT NULL AND source_file IS NOT NULL AND date IS NOT NULL AND CGM IS NOT NULL
        ORDER BY source_file, id, date
    """
    reader = con.execute(query).fetch_record_batch(rows_per_batch=args.read_batch_size)

    current_key = None
    feature_by_time: dict[pd.Timestamp, np.ndarray] = {}
    glucose_by_time: dict[pd.Timestamp, float] = {}
    time_queue = deque()
    pending_x: list[np.ndarray] = []
    pending_rows: list[int] = []
    pending_g: list[float] = []
    scanned = 0
    matched_anchors = 0
    valid_windows = 0

    def flush_predictions() -> None:
        nonlocal pending_x, pending_rows, pending_g
        if not pending_x:
            return
        x = np.stack(pending_x).astype(np.float32)
        x = (x - mean[None, None, :]) / std[None, None, :]
        out = model.predict(x, batch_size=args.batch_size, verbose=0)
        hybrid = _decode_v14_5(out, np.asarray(pending_g, dtype=np.float32))
        preds[np.asarray(pending_rows, dtype=np.int64)] = hybrid
        pending_x, pending_rows, pending_g = [], [], []

    for batch_idx, batch in enumerate(reader, start=1):
        frame = batch.to_pandas()
        scanned += len(frame)
        for r in frame.itertuples(index=False):
            source, pid = str(r.source_file), str(r.id)
            subject = (source, pid)
            if subject != current_key:
                flush_predictions()
                current_key = subject
                feature_by_time = {}
                glucose_by_time = {}
                time_queue = deque()

            ts = pd.Timestamp(r.date)
            vals = {name: _num(getattr(r, name)) for name in BASE_FEATURES}
            g = vals["CGM"]
            if not np.isfinite(g):
                continue
            feat = _feature_vector(ts, vals, glucose_by_time)
            feature_by_time[ts] = feat
            glucose_by_time[ts] = g
            time_queue.append(ts)

            key = (source, pid, ts)
            row_idx = key_to_row.get(key)
            if row_idx is not None:
                matched_anchors += 1
                persistence[row_idx] = g
                hist_start = ts - pd.Timedelta(minutes=HISTORY_MINUTES - STEP_MINUTES)
                expected = pd.date_range(hist_start, ts, freq=f"{STEP_MINUTES}min")
                if len(expected) == SEQ_LEN and all(t in feature_by_time for t in expected):
                    pending_x.append(np.stack([feature_by_time[t] for t in expected]).astype(np.float32))
                    pending_rows.append(row_idx)
                    pending_g.append(g)
                    valid_windows += 1
                    if len(pending_x) >= args.batch_size:
                        flush_predictions()

            cutoff = ts - pd.Timedelta(minutes=HISTORY_MINUTES + 30)
            while time_queue and time_queue[0] < cutoff:
                old = time_queue.popleft()
                feature_by_time.pop(old, None)
                glucose_by_time.pop(old, None)

        if batch_idx % 20 == 0:
            print(f"  scanned={scanned:,} template_anchors_seen={matched_anchors:,} valid_windows={valid_windows:,}", flush=True)

    flush_predictions()
    con.close()

    missing = np.isnan(preds).any(axis=1)
    missing_n = int(missing.sum())
    if missing_n and args.fallback == "persistence":
        fallback_ok = missing & np.isfinite(persistence)
        preds[fallback_ok] = persistence[fallback_ok, None]
        missing = np.isnan(preds).any(axis=1)
        missing_n = int(missing.sum())
        print(f"Persistence fallback applied to {int(fallback_ok.sum()):,} rows")

    print("\nInference summary")
    print(f"  template rows:       {len(template):,}")
    print(f"  matched anchors:     {matched_anchors:,}")
    print(f"  valid V14.5 windows: {valid_windows:,}")
    print(f"  unresolved rows:     {missing_n:,}")

    if missing_n:
        report = output_path.with_suffix(".missing.parquet")
        miss = template.loc[missing, [*KEY_COLS]].copy()
        pq.write_table(pa.Table.from_pandas(miss, preserve_index=False), report, compression="snappy")
        raise RuntimeError(
            f"{missing_n:,} template rows have no valid prediction. "
            f"Missing-key report written to {report}. "
            "No submission was written. Re-run with --fallback persistence only after explicitly accepting that policy."
        )

    submission = template[[*KEY_COLS]].copy()
    for j, col in enumerate(PRED_COLS):
        submission[col] = preds[:, j].astype(np.float64)
    if submission[list(PRED_COLS)].isna().any().any():
        raise RuntimeError("NaN predictions remain")

    original = pq.read_table(template_path, columns=list(KEY_COLS)).to_pandas()
    original["date"] = pd.to_datetime(original["date"], errors="raise")
    if not submission[list(KEY_COLS)].reset_index(drop=True).equals(original.reset_index(drop=True)):
        raise RuntimeError("Submission keys/order differ from official template")

    pq.write_table(pa.Table.from_pandas(submission, preserve_index=False), output_path, compression="snappy")
    print(f"\nSubmission written: {output_path}")
    print(f"Rows: {len(submission):,}; predictions finite: {np.isfinite(preds).all()}")


if __name__ == "__main__":
    main()
