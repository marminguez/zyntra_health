"""Profile MetaboNet train feature coverage without loading the full parquet.

Outputs non-null coverage for candidate V14.1 inputs and distributions of
split-related flags that can affect leakage-safe subject partitioning.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import duckdb

NUMERIC_CANDIDATES = [
    "CGM", "basal", "bolus", "insulin", "carbs",
    "heartrate", "steps", "calories_burned",
    "workout_duration", "workout_intensity",
    "skin_temp", "galvanic_skin_response", "air_temp",
    "age", "height", "weight", "age_of_diagnosis",
]
FLAG_COLUMNS = ["is_test", "subject_split_across_traintest"]


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()

    path = Path(args.data_dir) / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    con = duckdb.connect()
    parquet = str(path).replace("'", "''")

    total = con.execute(f"SELECT count(*) FROM read_parquet('{parquet}')").fetchone()[0]
    print(f"train rows: {total:,}\n")
    print("FEATURE COVERAGE")
    print("feature\tnon_null\tcoverage")
    for col in NUMERIC_CANDIDATES:
        n = con.execute(
            f"SELECT count({qident(col)}) FROM read_parquet('{parquet}')"
        ).fetchone()[0]
        print(f"{col}\t{n:,}\t{n/total:.2%}")

    print("\nSPLIT FLAGS")
    for col in FLAG_COLUMNS:
        print(f"\n{col}:")
        rows = con.execute(
            f"SELECT {qident(col)}, count(*) AS n FROM read_parquet('{parquet}') GROUP BY 1 ORDER BY 1 NULLS LAST"
        ).fetchall()
        for value, n in rows:
            print(f"  {value}: {n:,} ({n/total:.2%})")

    print("\nSUBJECT-LEVEL SPLIT FLAGS")
    rows = con.execute(f"""
        SELECT subject_split_across_traintest, count(*) AS subjects
        FROM (
            SELECT source_file, id, max(CAST(subject_split_across_traintest AS INTEGER))::BOOLEAN AS subject_split_across_traintest
            FROM read_parquet('{parquet}')
            GROUP BY source_file, id
        )
        GROUP BY 1
        ORDER BY 1 NULLS LAST
    """).fetchall()
    for value, n in rows:
        print(f"  {value}: {n:,} subjects")

    print("\nDATASETS / SUBJECTS")
    rows = con.execute(f"""
        SELECT source_file, count(DISTINCT id) AS subjects, count(*) AS rows
        FROM read_parquet('{parquet}')
        GROUP BY source_file
        ORDER BY source_file
    """).fetchall()
    for source, subjects, n in rows:
        print(f"  {source}: {subjects:,} subjects, {n:,} rows")

    con.close()

if __name__ == "__main__":
    main()
