#!/usr/bin/env python3
import os, sys, argparse, math
from typing import List

import dotenv
import pandas as pd
import numpy as np

dotenv.load_dotenv()
# We use s3fs transparently via pandas; ensure engine matches your env
PARQUET_ENGINE = os.environ.get("PANDAS_PARQUET_ENGINE", None)  # 'pyarrow' or 'fastparquet'

def list_s3_parts(s3_prefix: str) -> List[str]:
    # Normalize prefix (allow with/without trailing slash)
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    try:
        import s3fs  # noqa: F401
    except Exception as e:
        print("ERROR: s3fs is required for s3:// IO. pip install s3fs", file=sys.stderr)
        raise
    # s3fs works with pandas read_parquet globbing
    # We’ll just build a list via s3fs to be explicit:
    import s3fs
    fs = s3fs.S3FileSystem(anon=False)
    files = fs.glob(s3_prefix + "part-s*-*.parquet")
    files.sort()
    return ["s3://" + f for f in files]

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
    ycols = [c for c in df.columns if c.startswith("y_")]

    print("\n=== SANITY CHECK ===")
    print(f"file: {out_parquet}")
    print(f"rows: {n:,}")
    if "s3_key" in df.columns:
        uniq_keys = int(df["s3_key"].nunique())
        print(f"unique s3_key: {uniq_keys:,}  (expected: {expected_unique_s3:,})"
              f"{'  ✅' if uniq_keys == expected_unique_s3 else '  ⚠️'}")
    else:
        print("unique s3_key: N/A (column not present)")

    if ycols:
        y = df[ycols].to_numpy(dtype=np.float64, copy=False)
        sums = y.sum(axis=1)
        bad = np.flatnonzero(np.abs(sums - 1.0) > 1e-6).size
        print(f"label cols: {len(ycols)} (y_0..y_{len(ycols)-1})")
        print(f"non-normalized rows (|sum-1|>1e-6): {bad} "
              f"{'✅' if bad==0 else '⚠️'}")
        print(f"min(y): {y.min():.6g}  max(y): {y.max():.6g}")

        # quick entropy stats (natural logs)
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = -(y * np.where(y>0, np.log(y), 0)).sum(axis=1)
        finite_ent = ent[np.isfinite(ent)]
        if finite_ent.size:
            print(f"entropy nat mean={finite_ent.mean():.3f}  "
                  f"min={finite_ent.min():.3f}  max={finite_ent.max():.3f}")
    else:
        print("No y_* columns found; skipping label checks.")

    # quick distribution of some key meta cols if present
    for key in ("ctx", "street", "bet_sizing_id", "actor", "action"):
        if key in df.columns:
            top = df[key].value_counts().head(5)
            print(f"by {key} (top-5):")
            for k, v in top.items():
                print(f"  {k}: {v:,}")

def main():
    ap = argparse.ArgumentParser("Merge postflop part parquets from S3 and sanity-check")
    ap.add_argument("--s3-prefix", required=True,
                    help="S3 prefix containing part-sNN-*.parquet (e.g. s3://pokeraistore/datasets/rangenet_postflop/parts)")
    ap.add_argument("--out-parquet", default="data/datasets/rangenet_postflop_merged.parquet",
                    help="Local output parquet path")
    ap.add_argument("--expected-unique-s3", type=int, default=14848,
                    help="Expected unique s3_key count for a full merge")
    args = ap.parse_args()

    parts = list_s3_parts(args.s3_prefix)
    print(f"found {len(parts)} part files under {args.s3_prefix}")
    if not parts:
        sys.exit(1)

    merge_parts_to_single(parts, args.out_parquet)
    sanity_check(args.out_parquet, args.expected_unique_s3)

if __name__ == "__main__":
    main()