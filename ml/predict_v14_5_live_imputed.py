"""MetaboNet Live inference policy: frozen V14.5 + conservative CGM gap repair.

Pre-registered deployment policy (no target access):
- use the unchanged V14.5 model, normalization and 15-feature definition;
- build the same 288 x 5-minute / 24 h history;
- linearly interpolate INTERNAL CGM gaps only when each contiguous gap is <=30 min
  (<=6 missing 5-minute samples) and is bounded by observed CGM on both sides;
- if a timestamp itself is absent, synthesize only that grid row: CGM may be
  interpolated, while non-CGM covariates remain missing (the original feature
  builder converts them to 0 plus their missing flags);
- if the window is still unresolved, use current-glucose persistence for all
  four horizons;
- never read targets.parquet and never fit/tune anything.

The output preserves the official template keys and row order exactly and writes
coverage diagnostics alongside the submission.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
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
    BASE_FEATURES, COLUMNS, FEATURE_NAMES, HISTORY_MINUTES, SEQ_LEN,
    STEP_MINUTES, _feature_vector, _num,
)
from predict_v14_5_live import _decode_v14_5, _load_normalization

KEY_COLS = ("id", "source_file", "date")
PRED_COLS = ("pred_30", "pred_60", "pred_90", "pred_120")
MAX_GAP_STEPS = 6  # fixed a priori: 30 minutes at 5-minute cadence


def _repair_window(
    expected: pd.DatetimeIndex,
    vals_by_time: dict[pd.Timestamp, dict[str, float]],
) -> tuple[np.ndarray | None, int, int]:
    """Return feature matrix, imputed CGM count, synthesized timestamp count."""
    glucose = np.full(len(expected), np.nan, dtype=np.float64)
    synthesized = 0
    for i, ts in enumerate(expected):
        vals = vals_by_time.get(ts)
        if vals is None:
            synthesized += 1
        else:
            glucose[i] = vals["CGM"]

    missing_before = int(np.isnan(glucose).sum())
    if missing_before == 0:
        repaired = glucose
    else:
        repaired = glucose.copy()
        isnan = np.isnan(repaired)
        i = 0
        while i < len(repaired):
            if not isnan[i]:
                i += 1
                continue
            start = i
            while i < len(repaired) and isnan[i]:
                i += 1
            end = i - 1
            gap_len = end - start + 1
            left, right = start - 1, end + 1
            # Internal, bounded, short gaps only. No extrapolation at window edges.
            if gap_len <= MAX_GAP_STEPS and left >= 0 and right < len(repaired):
                if np.isfinite(repaired[left]) and np.isfinite(repaired[right]):
                    step = (repaired[right] - repaired[left]) / (gap_len + 1)
                    for j in range(gap_len):
                        repaired[start + j] = repaired[left] + step * (j + 1)

    if np.isnan(repaired).any():
        return None, 0, synthesized

    # Rebuild all glucose-derived features causally from the repaired 5-minute series.
    g_history: dict[pd.Timestamp, float] = {}
    feats: list[np.ndarray] = []
    for ts, g in zip(expected, repaired):
        original = vals_by_time.get(ts)
        if original is None:
            vals = {name: np.nan for name in BASE_FEATURES}
        else:
            vals = dict(original)
        vals["CGM"] = float(g)
        feats.append(_feature_vector(ts, vals, g_history))
        g_history[ts] = float(g)

    return np.stack(feats).astype(np.float32), missing_before, synthesized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--model", default="ml/results/v14_5/best_model.keras")
    ap.add_argument("--normalization", default="ml/results/v14_1/normalization.npz")
    ap.add_argument("--output", default="ml/results/v14_5_live/submission_v14_5_impute30.parquet")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--read-batch-size", type=int, default=50000)
    ap.add_argument("--memory-limit", default="3GB")
    a = ap.parse_args()

    test_path, template_path, model_path, norm_path, output_path = map(
        lambda x: Path(x).resolve(),
        [a.test, a.template, a.model, a.normalization, a.output],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (test_path, template_path, model_path, norm_path):
        if not p.exists():
            raise FileNotFoundError(p)

    template = pq.read_table(template_path).to_pandas()
    if list(template.columns) != [*KEY_COLS, *PRED_COLS]:
        raise ValueError(f"Unexpected template columns: {list(template.columns)}")
    template["date"] = pd.to_datetime(template["date"], errors="raise")
    key_to_row = {}
    for row_idx, r in enumerate(template.itertuples(index=False)):
        key = (str(r.source_file), str(r.id), pd.Timestamp(r.date))
        if key in key_to_row:
            raise ValueError(f"Duplicate template key: {key}")
        key_to_row[key] = row_idx

    print(f"Template rows: {len(template):,}")
    print(f"Policy: V14.5 + internal linear CGM interpolation <=30 min + persistence fallback")
    print(f"Features: {len(FEATURE_NAMES)}; sequence length: {SEQ_LEN}")

    mean, std = _load_normalization(norm_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = np.full((len(template), 4), np.nan, dtype=np.float32)
    current_glucose = np.full(len(template), np.nan, dtype=np.float32)
    source_mode = np.zeros(len(template), dtype=np.uint8)  # 1 strict, 2 imputed, 3 persistence

    tmp = output_path.parent / ".v14_5_live_impute_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(tmp / "infer.duckdb"))
    con.execute(f"SET temp_directory='{str(tmp).replace(chr(39), chr(39)*2)}'")
    con.execute(f"SET memory_limit='{a.memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    select_cols = ", ".join(f'"{c}"' for c in COLUMNS if c != "subject_split_across_traintest")
    test_sql = str(test_path).replace("'", "''")
    # Keep rows even when CGM is missing: they carry covariates and are repairable grid positions.
    query = f"""
        SELECT {select_cols}
        FROM read_parquet('{test_sql}')
        WHERE id IS NOT NULL AND source_file IS NOT NULL AND date IS NOT NULL
        ORDER BY source_file, id, date
    """
    reader = con.execute(query).fetch_record_batch(rows_per_batch=a.read_batch_size)

    current_subject = None
    vals_by_time: dict[pd.Timestamp, dict[str, float]] = {}
    time_queue = deque()
    pending_x, pending_rows, pending_g = [], [], []
    scanned = anchors = strict_n = imputed_n = imputed_points = synthesized_points = 0

    def flush() -> None:
        nonlocal pending_x, pending_rows, pending_g
        if not pending_x:
            return
        x = np.stack(pending_x).astype(np.float32)
        x = (x - mean[None, None, :]) / std[None, None, :]
        out = model.predict(x, batch_size=a.batch_size, verbose=0)
        preds[np.asarray(pending_rows, dtype=np.int64)] = _decode_v14_5(
            out, np.asarray(pending_g, dtype=np.float32)
        )
        pending_x, pending_rows, pending_g = [], [], []

    for batch_idx, batch in enumerate(reader, 1):
        frame = batch.to_pandas()
        scanned += len(frame)
        for r in frame.itertuples(index=False):
            source, pid, ts = str(r.source_file), str(r.id), pd.Timestamp(r.date)
            subject = (source, pid)
            if subject != current_subject:
                flush()
                current_subject = subject
                vals_by_time = {}
                time_queue = deque()

            vals = {name: _num(getattr(r, name)) for name in BASE_FEATURES}
            vals_by_time[ts] = vals
            time_queue.append(ts)

            row_idx = key_to_row.get((source, pid, ts))
            if row_idx is not None:
                anchors += 1
                g_now = vals["CGM"]
                if np.isfinite(g_now):
                    current_glucose[row_idx] = g_now
                    start = ts - pd.Timedelta(minutes=HISTORY_MINUTES - STEP_MINUTES)
                    expected = pd.date_range(start, ts, freq=f"{STEP_MINUTES}min")
                    if len(expected) == SEQ_LEN:
                        x, n_imp, n_synth = _repair_window(expected, vals_by_time)
                        if x is not None:
                            pending_x.append(x)
                            pending_rows.append(row_idx)
                            pending_g.append(float(g_now))
                            if n_imp == 0:
                                strict_n += 1
                                source_mode[row_idx] = 1
                            else:
                                imputed_n += 1
                                imputed_points += n_imp
                                synthesized_points += n_synth
                                source_mode[row_idx] = 2
                            if len(pending_x) >= a.batch_size:
                                flush()

            cutoff = ts - pd.Timedelta(minutes=HISTORY_MINUTES + 30)
            while time_queue and time_queue[0] < cutoff:
                old = time_queue.popleft()
                vals_by_time.pop(old, None)

        if batch_idx % 20 == 0:
            print(
                f"  scanned={scanned:,} anchors={anchors:,} strict={strict_n:,} "
                f"imputed={imputed_n:,}", flush=True
            )

    flush()
    con.close()

    unresolved = np.isnan(preds).any(axis=1)
    fallback_ok = unresolved & np.isfinite(current_glucose)
    preds[fallback_ok] = current_glucose[fallback_ok, None]
    source_mode[fallback_ok] = 3
    unresolved = np.isnan(preds).any(axis=1)
    unresolved_n = int(unresolved.sum())

    print("\nCoverage summary")
    print(f"  template rows:        {len(template):,}")
    print(f"  matched anchors:      {anchors:,}")
    print(f"  strict V14.5:         {int((source_mode==1).sum()):,}")
    print(f"  imputed <=30m V14.5:  {int((source_mode==2).sum()):,}")
    print(f"  persistence fallback: {int((source_mode==3).sum()):,}")
    print(f"  unresolved:           {unresolved_n:,}")
    print(f"  CGM points imputed:   {imputed_points:,}")
    print(f"  timestamps synthesized inside accepted windows: {synthesized_points:,}")

    if unresolved_n:
        report = output_path.with_suffix(".unresolved.parquet")
        pq.write_table(
            pa.Table.from_pandas(template.loc[unresolved, list(KEY_COLS)], preserve_index=False),
            report, compression="snappy"
        )
        raise RuntimeError(f"{unresolved_n:,} rows remain unresolved; report: {report}")

    submission = template[list(KEY_COLS)].copy()
    for j, col in enumerate(PRED_COLS):
        submission[col] = preds[:, j].astype(np.float64)
    original = pq.read_table(template_path, columns=list(KEY_COLS)).to_pandas()
    original["date"] = pd.to_datetime(original["date"], errors="raise")
    if not submission[list(KEY_COLS)].reset_index(drop=True).equals(original.reset_index(drop=True)):
        raise RuntimeError("Submission keys/order differ from official template")
    if not np.isfinite(preds).all():
        raise RuntimeError("Non-finite predictions remain")

    pq.write_table(pa.Table.from_pandas(submission, preserve_index=False), output_path, compression="snappy")
    diag = {
        "policy": "frozen V14.5 + internal linear CGM interpolation <=30min + persistence fallback",
        "max_gap_minutes": 30,
        "template_rows": len(template),
        "strict_v14_5": int((source_mode == 1).sum()),
        "imputed_v14_5": int((source_mode == 2).sum()),
        "persistence_fallback": int((source_mode == 3).sum()),
        "unresolved": 0,
        "cgm_points_imputed_in_accepted_windows": int(imputed_points),
        "synthesized_timestamps_in_accepted_windows": int(synthesized_points),
        "targets_used": False,
        "model_retrained": False,
    }
    output_path.with_suffix(".coverage.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"\nSubmission written: {output_path}")
    print(f"Coverage report: {output_path.with_suffix('.coverage.json')}")


if __name__ == "__main__":
    main()
