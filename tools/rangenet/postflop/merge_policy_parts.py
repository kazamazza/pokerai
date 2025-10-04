#!/usr/bin/env python3
# file: tools/rangenet/postflop/merge_policy_parts.py
import os, sys, glob, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser("Merge & dedupe postflop policy parts")
    ap.add_argument("--parts-dir", default="data/datasets/postflop_policy_parts",
                    help="Directory containing shard parquet parts")
    ap.add_argument("--out", default="data/datasets/postflop_policy_merged.parquet",
                    help="Output merged parquet path")
    ap.add_argument("--expected-total", type=int, default=19426,
                    help="Expected final row count after dedupe")
    ap.add_argument("--dedupe-keys", default="s3_key,node_key",
                    help="Comma-separated columns to dedupe on (must exist)")
    args = ap.parse_args()

    parts_dir = args.parts_dir
    files = sorted(glob.glob(os.path.join(parts_dir, "*.parquet")))
    if not files:
        print(f"[err] no parquet parts found under {parts_dir}")
        sys.exit(1)

    print(f"[info] found {len(files)} part(s) → reading...")
    dfs = []
    for i, fp in enumerate(files, 1):
        df = pd.read_parquet(fp)
        dfs.append(df)
        if i % 25 == 0 or i == len(files):
            print(f"  read {i}/{len(files)} parts")

    df = pd.concat(dfs, ignore_index=True)
    before = len(df)

    # Deduping
    keys = [c.strip() for c in args.dedupe_keys.split(",") if c.strip()]
    if all(k in df.columns for k in keys) and keys:
        df = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
        print(f"[info] deduped on {keys}")
    else:
        print(f"[warn] dedupe keys {keys} not fully present; skipping dedupe")

    after = len(df)
    removed = before - after

    # Save
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ wrote {out}")
    print(f"rows: before={before:,}  removed_dups={removed:,}  after={after:,}")

    # Check expected
    ok = (args.expected_total is None) or (after == args.expected_total)
    if args.expected_total is not None:
        if ok:
            print(f"✅ count matches expected_total={args.expected_total:,}")
        else:
            print(f"❌ count mismatch: got {after:,}, expected {args.expected_total:,}")
            sys.exit(2)

if __name__ == "__main__":
    main()