#!/usr/bin/env python3
import os, sys, argparse, math
from pathlib import Path
from typing import List

import dotenv
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.inference.policy import ACTION_VOCAB

dotenv.load_dotenv()
# We use s3fs transparently via pandas; ensure engine matches your env
PARQUET_ENGINE = os.environ.get("PANDAS_PARQUET_ENGINE", None)  # 'pyarrow' or 'fastparquet'

def list_s3_parts(s3_prefix: str, pattern: str = "part-*.parquet") -> List[str]:
    """
    s3_prefix: 's3://bucket/path/to/dir' OR 'bucket/path/to/dir'
    pattern  : glob for file names in that dir
               e.g. 'part-*.parquet' or 'shard-*-part-*.parquet'
    """
    import s3fs
    # normalize prefix for s3fs (no scheme)
    if s3_prefix.startswith("s3://"):
        s3_prefix = s3_prefix[5:]
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    fs = s3fs.S3FileSystem(anon=False)
    keys = fs.glob(s3_prefix + pattern)
    keys.sort()
    return ["s3://" + k for k in keys]

def merge_parts_to_single(parts: List[str], out_parquet: str) -> int:
    """
    Load each part into memory and append; for 18 parts this is fine.
    If you scale to hundreds, switch to pyarrow ParquetWriter chunking.
    Returns total rows written.
    """
    if not parts:
        raise SystemExit("No part files found under the given S3 prefix.")

    dfs = []
    for i, p in enumerate(parts, 1):
        print(f"[merge] reading {i}/{len(parts)} -> {p}")
        dfp = pd.read_parquet(p, engine=PARQUET_ENGINE)
        dfs.append(dfp)

    out = pd.concat(dfs, ignore_index=True)
    # basic hygiene
    ycols = [c for c in out.columns if c.startswith("y_")]
    if ycols:
        out[ycols] = out[ycols].fillna(0.0)

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True) if os.path.dirname(out_parquet) else None
    out.to_parquet(out_parquet, index=False, engine=PARQUET_ENGINE)
    print(f"[merge] wrote {out_parquet} with {len(out):,} rows")
    return len(out)

def sanity_check(out_parquet: str, expected_unique_s3: int) -> None:
    df = pd.read_parquet(out_parquet, engine=PARQUET_ENGINE)
    n = len(df)
    print("\n=== SANITY CHECK ===")
    print(f"file: {out_parquet}")
    print(f"rows: {n:,}")

    # Unique s3 keys (if present)
    if "s3_key" in df.columns:
        uniq_keys = int(df["s3_key"].nunique())
        print(f"unique s3_key: {uniq_keys:,}  (expected: {expected_unique_s3:,})"
              f"{'  ✅' if uniq_keys == expected_unique_s3 else '  ⚠️'}")
    else:
        print("unique s3_key: N/A (column not present)")

    # Prefer ACTION_VOCAB soft columns (policy parquet)
    # Fallback to y_* if present (other pipelines)
    action_cols = [c for c in df.columns if c in ACTION_VOCAB]
    ycols = [c for c in df.columns if c.startswith("y_")]
    if action_cols:
        probs = df[action_cols].to_numpy(dtype=np.float64, copy=False)
        sums = probs.sum(axis=1)
        bad = np.flatnonzero(np.abs(sums - 1.0) > 1e-6).size
        print(f"policy cols: {len(action_cols)} ({', '.join(action_cols)})")
        print(f"non-normalized rows (|sum-1|>1e-6): {bad} {'✅' if bad==0 else '⚠️'}")

        # Per-bucket totals (what you need to spot zero raises/folds)
        totals = {c: float(df[c].sum()) for c in action_cols}
        print("\n--- totals by action bucket (should see mass in RAISE_* / FOLD) ---")
        for k in ACTION_VOCAB:
            if k in totals:
                print(f"{k:>10s} : {totals[k]:.6f}")

        # quick peek by actor/action
        if "actor" in df.columns and "action" in df.columns:
            print("\nby actor (top-5):")
            print(df["actor"].value_counts().head(5).to_string())
            print("\nby hard action (top-10):")
            print(df["action"].value_counts().head(10).to_string())
    elif ycols:
        y = df[ycols].to_numpy(dtype=np.float64, copy=False)
        sums = y.sum(axis=1)
        bad = np.flatnonzero(np.abs(sums - 1.0) > 1e-6).size
        print(f"label cols: {len(ycols)} (y_0..y_{len(ycols)-1})")
        print(f"non-normalized rows (|sum-1|>1e-6): {bad} {'✅' if bad==0 else '⚠️'}")
    else:
        print("No policy/y_* columns found; nothing to validate.")

    # Context distribution (optional)
    for key in ("ctx", "street", "bet_sizing_id", "actor", "action"):
        if key in df.columns:
            top = df[key].value_counts().head(5)
            print(f"\nby {key} (top-5):")
            for k, v in top.items():
                print(f"  {k}: {v:,}")

def main():
    ap = argparse.ArgumentParser("Merge postflop part parquets from S3 and sanity-check")
    ap.add_argument("--s3-prefix", required=True,
                    help="s3://bucket/prefix containing parts")
    ap.add_argument("--pattern", default="part-*.parquet",
                    help="filename glob (e.g. 'part-*.parquet' or 'shard-*-part-*.parquet')")
    ap.add_argument("--out-parquet", default="data/datasets/rangenet_postflop_merged.parquet",
                    help="Local output parquet path")
    ap.add_argument("--expected-unique-s3", type=int, default=14848,
                    help="Expected unique s3_key count for a full merge")
    args = ap.parse_args()

    parts = list_s3_parts(args.s3_prefix, args.pattern)
    print(f"found {len(parts)} part files under {args.s3_prefix} (pattern={args.pattern})")
    if not parts:
        sys.exit(1)

    merge_parts_to_single(parts, args.out_parquet)
    sanity_check(args.out_parquet, args.expected_unique_s3)

if __name__ == "__main__":
    main()