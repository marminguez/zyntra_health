"""Inspect MetaboNet parquet schema/metadata without loading the dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    for name in ("train.parquet", "test.parquet"):
        path = Path(args.data_dir) / name
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        print(f"\n=== {name} ===")
        print(f"rows: {pf.metadata.num_rows:,}")
        print(f"row_groups: {pf.num_row_groups}")
        print("columns:")
        for field in pf.schema_arrow:
            print(f"  {field.name}: {field.type}")

if __name__ == "__main__":
    main()
