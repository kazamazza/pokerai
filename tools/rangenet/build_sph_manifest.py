#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List, Dict
import pandas as pd

# tools/sph/build_sph_manifest.py
import argparse, pandas as pd
from pathlib import Path

PAIR_SEPS = ("_", "v")  # support "BB_SB" or "BBvSB"

def _split_pair(name: str):
    s = name.upper()
    for sep in PAIR_SEPS:
        if sep in s:
            a, b = s.split(sep, 1)
            return a.strip(), b.strip()
    return None, None

def _rows_from_ctx_dir(ctx_dir: Path, ctx: str) -> list[dict]:
    rows = []
    for stack_dir in sorted(ctx_dir.iterdir()):
        if not stack_dir.is_dir():
            continue
        try:
            stack = int(stack_dir.name)
        except Exception:
            continue

        for pair_dir in sorted(stack_dir.iterdir()):
            if not pair_dir.is_dir():
                continue

            ip, oop = _split_pair(pair_dir.name)
            if not ip or not oop:
                continue

            ip_csv  = pair_dir / "ip.csv"
            oop_csv = pair_dir / "oop.csv"

            # Rows are “file pointers” for SphIndex/lookup to materialize later
            if ip_csv.exists():
                rows.append({
                    "stack_bb": stack,
                    "ctx": ctx.upper(),
                    "ip_pos": ip,
                    "oop_pos": oop,
                    "hero_pos": "IP",
                    "rel_path": f"sph/{ctx}/{stack}/{ip}_{oop}/ip.csv",
                    "abs_path": None,
                })
            if oop_csv.exists():
                rows.append({
                    "stack_bb": stack,
                    "ctx": ctx.upper(),
                    "ip_pos": ip,
                    "oop_pos": oop,
                    "hero_pos": "OOP",
                    "rel_path": f"sph/{ctx}/{stack}/{ip}_{oop}/oop.csv",
                    "abs_path": None,
                })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Scan SPH cache (ip.csv/oop.csv) into a manifest parquet.")
    ap.add_argument("--cache-root", type=Path, default=Path("data/vendor_cache/sph"),
                    help="Root containing SRP/LIMP_SINGLE/LIMP_MULTI subdirs")
    ap.add_argument("--out", type=Path, default=Path("data/artifacts/sph_manifest.parquet"))
    ap.add_argument("--ctx", type=str, nargs="*", default=["SRP", "LIMP_SINGLE", "LIMP_MULTI"],
                    help="Contexts to include (default: SRP LIMP_SINGLE LIMP_MULTI)")
    args = ap.parse_args()

    rows: list[dict] = []
    for ctx in args.ctx:
        ctx_dir = args.cache_root / ctx
        if ctx_dir.exists():
            rows.extend(_rows_from_ctx_dir(ctx_dir, ctx))

    if not rows:
        raise SystemExit(f"No SPH ip/oop CSVs found under {args.cache_root}")

    df = pd.DataFrame(rows)
    # tidy types + order + dedupe
    df["stack_bb"] = pd.to_numeric(df["stack_bb"], errors="coerce").astype("Int64")
    df["ctx"] = df["ctx"].astype(str).str.upper()
    for c in ("ip_pos","oop_pos","hero_pos","rel_path"):
        df[c] = df[c].astype(str)
    df = (df.dropna(subset=["stack_bb"])
            .drop_duplicates(subset=["ctx","stack_bb","ip_pos","oop_pos","hero_pos","rel_path"])
            .sort_values(["ctx","stack_bb","ip_pos","oop_pos","hero_pos"])
            .reset_index(drop=True))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"✅ wrote SPH manifest → {args.out} rows={len(df)}")
    print(df.head(12).to_string(index=False))

if __name__ == "__main__":
    main()