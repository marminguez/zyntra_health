"""Build causal 24h CGM sidecars aligned 1:1 to existing Scale-10 shards.

This does NOT run or approximate GlucoFM. It only reconstructs the official-style
24h/5-min CGM context that a future official GlucoFM encoder can consume.

For every existing Scale-10 anchor, output exactly 288 positions:
    anchor - 23h55m, ..., anchor
No post-anchor CGM is ever included.

Missing CGM is encoded as 0.0 with observed_mask=0. Observed finite CGM keeps its
raw value with observed_mask=1. Shard order and timestamps are preserved exactly.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from prepare_v15_4_scale10 import safe_name

STEP_MINUTES = 5
STEPS = 288
LOOKBACK_MINUTES = (STEPS - 1) * STEP_MINUTES  # 1435 min, inclusive of anchor


@dataclass
class ShardSpec:
    split: str
    path: Path
    out_path: Path
    timestamps: np.ndarray
    anchors: pd.DatetimeIndex
    actual_source: str | None = None
    actual_pid: str | None = None

    @property
    def safe_stem(self) -> str:
        return self.path.stem

    @property
    def min_needed(self) -> pd.Timestamp:
        return pd.Timestamp(self.anchors.min()) - pd.Timedelta(minutes=LOOKBACK_MINUTES)

    @property
    def max_needed(self) -> pd.Timestamp:
        return pd.Timestamp(self.anchors.max())


def load_specs(reference: Path, out: Path) -> list[ShardSpec]:
    specs: list[ShardSpec] = []
    for split in ("train", "validation"):
        src_dir = reference / split
        if not src_dir.exists():
            raise FileNotFoundError(src_dir)
        for path in sorted(src_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                if "timestamp" not in z.files:
                    raise AssertionError(f"{path}: missing timestamp")
                ts = np.asarray(z["timestamp"])
            anchors = pd.to_datetime(ts)
            if anchors.hasnans:
                raise AssertionError(f"{path}: invalid anchor timestamp")
            specs.append(
                ShardSpec(
                    split=split,
                    path=path,
                    out_path=out / split / path.name,
                    timestamps=ts,
                    anchors=pd.DatetimeIndex(anchors),
                )
            )
    if not specs:
        raise RuntimeError(f"No reference shards found under {reference}")
    return specs


def resolve_actual_subjects(con: duckdb.DuckDBPyConnection, parquet_path: Path, specs: list[ShardSpec]) -> None:
    """Resolve safe shard names back to exact source_file/id values and detect collisions."""
    ps = str(parquet_path).replace("'", "''")
    rows = con.execute(
        f"""
        SELECT DISTINCT CAST(source_file AS VARCHAR) AS source_file,
                        CAST(id AS VARCHAR) AS pid
        FROM read_parquet('{ps}')
        WHERE source_file IS NOT NULL AND id IS NOT NULL
        """
    ).fetchall()

    mapping: dict[str, tuple[str, str]] = {}
    collisions: dict[str, list[tuple[str, str]]] = {}
    for source, pid in rows:
        stem = f"{safe_name(str(source))}__{safe_name(str(pid))}"
        pair = (str(source), str(pid))
        if stem in mapping and mapping[stem] != pair:
            collisions.setdefault(stem, [mapping[stem]]).append(pair)
        else:
            mapping[stem] = pair

    wanted_stems = {s.safe_stem for s in specs}
    relevant_collisions = {k: v for k, v in collisions.items() if k in wanted_stems}
    if relevant_collisions:
        first = next(iter(relevant_collisions.items()))
        raise RuntimeError(f"Safe-name collision for reference shard {first[0]}: {first[1]}")

    missing = []
    for spec in specs:
        pair = mapping.get(spec.safe_stem)
        if pair is None:
            missing.append(spec.safe_stem)
        else:
            spec.actual_source, spec.actual_pid = pair
    if missing:
        raise RuntimeError(f"Could not resolve {len(missing)} shard subjects; first: {missing[:5]}")


def create_wanted_table(con: duckdb.DuckDBPyConnection, specs: list[ShardSpec]) -> None:
    frame = pd.DataFrame(
        {
            "source_file": [s.actual_source for s in specs],
            "pid": [s.actual_pid for s in specs],
            "min_needed": [s.min_needed for s in specs],
            "max_needed": [s.max_needed for s in specs],
        }
    )
    con.register("wanted_df", frame)
    con.execute("CREATE TEMP TABLE wanted AS SELECT * FROM wanted_df")
    con.unregister("wanted_df")


def build_one(spec: ShardSpec, glucose_by_time: dict[pd.Timestamp, float]) -> tuple[int, int]:
    n = len(spec.anchors)
    cgm = np.zeros((n, STEPS), dtype=np.float32)
    mask = np.zeros((n, STEPS), dtype=np.float32)

    for i, anchor in enumerate(spec.anchors):
        times = pd.date_range(
            pd.Timestamp(anchor) - pd.Timedelta(minutes=LOOKBACK_MINUTES),
            pd.Timestamp(anchor),
            freq=f"{STEP_MINUTES}min",
        )
        if len(times) != STEPS or times[-1] != anchor:
            raise AssertionError(f"{spec.path}: bad causal grid for {anchor}")
        for j, ts in enumerate(times):
            value = glucose_by_time.get(pd.Timestamp(ts))
            if value is not None and np.isfinite(value):
                cgm[i, j] = np.float32(value)
                mask[i, j] = 1.0

    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        spec.out_path,
        cgm_24h=cgm,
        observed_mask=mask,
        timestamp=spec.timestamps,
    )
    observed = int(mask.sum())
    return observed, int(mask.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--reference", default="ml/data/v15_4_scale10")
    ap.add_argument("--outdir", default="ml/data/glucofm_scale10")
    ap.add_argument("--batch-size", type=int, default=100000)
    ap.add_argument("--memory-limit", default="3GB")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    parquet_path = Path(a.train_parquet)
    reference = Path(a.reference)
    out = Path(a.outdir)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if a.reset:
        shutil.rmtree(out, ignore_errors=True)

    specs = load_specs(reference, out)
    print("=== GLUCOFM 24H SIDECAR BUILD ===")
    print(f"reference shards: {len(specs):,}")
    print(f"anchors:          {sum(len(s.anchors) for s in specs):,}")
    print("context:          288 x 5 min = 24h inclusive of anchor")
    print("future CGM:       False")

    tmp = parquet_path.parent / ".glucofm_sidecar_duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect(str(tmp / "prepare.duckdb"))
    con.execute(f"SET temp_directory='{str(tmp).replace(chr(39), chr(39) * 2)}'")
    con.execute(f"SET memory_limit='{a.memory_limit}'")
    con.execute("SET preserve_insertion_order=false")

    print("Resolving exact source_file/id values from reference shard names...", flush=True)
    resolve_actual_subjects(con, parquet_path, specs)
    create_wanted_table(con, specs)

    by_subject: dict[tuple[str, str], ShardSpec] = {
        (str(s.actual_source), str(s.actual_pid)): s for s in specs
    }
    if len(by_subject) != len(specs):
        raise RuntimeError("Expected exactly one Scale-10 shard per source_file/id")

    ps = str(parquet_path).replace("'", "''")
    query = f"""
        SELECT CAST(p.source_file AS VARCHAR) AS source_file,
               CAST(p.id AS VARCHAR) AS pid,
               p.date,
               p.CGM
        FROM read_parquet('{ps}') p
        INNER JOIN wanted w
          ON CAST(p.source_file AS VARCHAR) = w.source_file
         AND CAST(p.id AS VARCHAR) = w.pid
         AND p.date BETWEEN w.min_needed AND w.max_needed
        WHERE p.date IS NOT NULL
        ORDER BY source_file, pid, date
    """

    reader = con.execute(query).fetch_record_batch(rows_per_batch=a.batch_size)
    current_key: tuple[str, str] | None = None
    glucose_by_time: dict[pd.Timestamp, float] = {}
    written = 0
    observed_total = 0
    slots_total = 0
    duplicate_timestamps = 0
    conflicting_duplicates = 0

    def flush() -> None:
        nonlocal written, observed_total, slots_total, glucose_by_time
        if current_key is None:
            return
        spec = by_subject[current_key]
        observed, slots = build_one(spec, glucose_by_time)
        written += 1
        observed_total += observed
        slots_total += slots
        if written % 25 == 0:
            print(f"written {written:,}/{len(specs):,} shards", flush=True)
        glucose_by_time = {}

    for batch in reader:
        frame = batch.to_pandas()
        for r in frame.itertuples(index=False):
            key = (str(r.source_file), str(r.pid))
            if key != current_key:
                flush()
                current_key = key
            ts = pd.Timestamp(r.date)
            value = float(r.CGM) if r.CGM is not None else np.nan
            if ts in glucose_by_time:
                duplicate_timestamps += 1
                old = glucose_by_time[ts]
                if np.isfinite(old) and np.isfinite(value) and not np.isclose(old, value):
                    conflicting_duplicates += 1
            glucose_by_time[ts] = value
    flush()
    con.close()

    missing_outputs = [str(s.out_path) for s in specs if not s.out_path.exists()]
    if missing_outputs:
        raise RuntimeError(f"Missing {len(missing_outputs)} output shards; first: {missing_outputs[:5]}")

    metadata = {
        "version": "glucofm-scale10-sidecar-v1",
        "reference": str(reference),
        "reference_shards": len(specs),
        "anchors": int(sum(len(s.anchors) for s in specs)),
        "context_steps": STEPS,
        "step_minutes": STEP_MINUTES,
        "context_minutes_inclusive": LOOKBACK_MINUTES,
        "window": "anchor-23h55m through anchor inclusive",
        "future_cgm_input": False,
        "missing_cgm_fill": 0.0,
        "observed_mask": True,
        "observed_fraction": (observed_total / slots_total) if slots_total else None,
        "duplicate_timestamps_seen": duplicate_timestamps,
        "conflicting_duplicate_cgm": conflicting_duplicates,
        "glucofm_weights_used": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== GLUCOFM SIDECAR COMPLETE ===")
    print(json.dumps(metadata, indent=2))
    print(f"Artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
