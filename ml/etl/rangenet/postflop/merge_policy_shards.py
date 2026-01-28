#!/usr/bin/env python3

"""
Merge sharded postflop policy parquet files into a single parquet.

This script is intentionally dumb:
- no feature engineering
- no reshaping
- no vocab logic
- optional filtering + dedup only

It is safe to rerun and easy to audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def list_parquet_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {input_dir}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="Directory containing sharded parquet files")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output merged parquet path")
    ap.add_argument("--drop-invalid", action="store_true",
                    help="Drop rows where valid == false")
    ap.add_argument("--dedup", action="store_true",
                    help="Drop exact duplicate rows")
    ap.add_argument("--row-group-size", type=int, default=100_000,
                    help="Row group size for output parquet")
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    output_path: Path = args.output

    files = list_parquet_files(input_dir)
    print(f"🔗 merging {len(files)} parquet shards from {input_dir}")

    dataset = ds.dataset(
        [str(p) for p in files],
        format="parquet"
    )

    table = dataset.to_table()

    # --- optional filters ---
    if args.drop_invalid and "valid" in table.schema.names:
        before = table.num_rows
        mask = pa.compute.equal(table["valid"], True)
        table = table.filter(mask)
        print(f"🧹 drop_invalid: {before} → {table.num_rows}")

    if args.dedup:
        before = table.num_rows
        table = table.drop_duplicates()
        print(f"🧹 dedup: {before} → {table.num_rows}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        table,
        output_path,
        compression="zstd",
        row_group_size=args.row_group_size,
        use_dictionary=True,
    )

    print(f"✅ wrote merged parquet: {output_path}")
    print(f"   rows={table.num_rows} cols={table.num_columns}")


if __name__ == "__main__":
    main()