#!/usr/bin/env python3
# tools/merge_preflop_shards.py

from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd

SHARD_RE = re.compile(r"(.*)\.shard(\d+)of(\d+)\.parquet$", re.IGNORECASE)

def _discover_shards(input_path: str) -> Tuple[List[Path], Path]:
    """
    Accepts either:
      - a single shard path like data/evnet_preflop.shard0of4.parquet
      - a directory or stem path like data/evnet_preflop.parquet (we'll glob neighbors)
    Returns (ordered_shard_paths, suggested_output_path)
    """
    p = Path(input_path)
    if p.is_file():
        m = SHARD_RE.match(p.name)
        if not m:
            raise SystemExit(f"Not a shard: {p.name}")
        stem_base = m.group(1)
        total = int(m.group(3))
        shard_paths = []
        for i in range(total):
            sp = p.with_name(f"{stem_base}.shard{i}of{total}.parquet")
            if not sp.exists():
                raise SystemExit(f"Missing shard: {sp}")
            shard_paths.append(sp)
        out = p.with_name(f"{stem_base}.parquet")
        return shard_paths, out

    # If it's a directory or non-file: search for *.shard*of*.parquet in the same dir / pattern
    if p.is_dir():
        cand = sorted(p.glob("*.shard*of*.parquet"))
        if not cand:
            raise SystemExit(f"No shards found in {p}")
        # group by base stem
        by_base = {}
        for c in cand:
            m = SHARD_RE.match(c.name)
            if not m:
                continue
            by_base.setdefault(m.group(1), []).append(c)
        if len(by_base) != 1:
            raise SystemExit(f"Found multiple shard groups: {list(by_base.keys())}. "
                             f"Pass a specific shard file instead.")
        base, files = next(iter(by_base.items()))
        # sort and verify completeness
        idx_tot = []
        for f in files:
            m = SHARD_RE.match(f.name); idx_tot.append((int(m.group(2)), int(m.group(3)), f))
        idx_tot.sort(key=lambda x: x[0])
        total = idx_tot[0][1]
        paths = [f for i,t,f in idx_tot]
        if len(paths) != total or [i for i,_,_ in idx_tot] != list(range(total)):
            raise SystemExit(f"Incomplete or non-contiguous shards for base '{base}'.")
        out = p / f"{base}.parquet"
        return paths, out

    # Treat as stem path string ending with .parquet (without shard suffix)
    # Look for siblings with shard suffix
    if input_path.endswith(".parquet"):
        p = Path(input_path)
        base = p.stem
        cand = sorted(p.parent.glob(f"{base}.shard*of*.parquet"))
        if not cand:
            raise SystemExit(f"No shards found matching {p.parent}/{base}.shard*of*.parquet")
        # recurse via directory logic
        return _discover_shards(str(p.parent))
    raise SystemExit(f"Invalid input: {input_path}")

def main():
    ap = argparse.ArgumentParser(description="Merge EV preflop shard parquets into a single parquet.")
    ap.add_argument("input", help="One shard file, a directory containing shards, or base .parquet stem")
    ap.add_argument("--out", help="Output parquet path (default: base without shard suffix)")
    ap.add_argument("--drop-dupes", action="store_true", help="Drop duplicate rows after concat")
    args = ap.parse_args()

    shard_paths, suggested_out = _discover_shards(args.input)
    out_path = Path(args.out) if args.out else suggested_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(shard_paths)} shards:")
    for p in shard_paths:
        print(f"  - {p}")

    # Read & sanity schema
    dfs = []
    cols_ref = None
    for i, sp in enumerate(shard_paths):
        df = pd.read_parquet(sp)
        if cols_ref is None:
            cols_ref = list(df.columns)
        else:
            if list(df.columns) != cols_ref:
                raise SystemExit(f"Schema mismatch in {sp.name}")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    if args.drop_dupes:
        before = len(merged)
        merged = merged.drop_duplicates()
        print(f"Dropped {before - len(merged)} duplicate rows")

    merged.to_parquet(out_path, index=False)
    print(f"✅ Merged → {out_path} | rows={len(merged)} | cols={len(merged.columns)}")

    # Quick spot-check
    for col in ("facing_flag","free_check","hero_pos"):
        if col in merged.columns:
            vc = merged[col].value_counts(dropna=False).to_dict()
            print(f"  {col} distribution: {vc}")

if __name__ == "__main__":
    main()