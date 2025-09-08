#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List, Dict
import pandas as pd

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
            try:
                ip, oop = pair_dir.name.split("_")
            except Exception:
                continue

            # We expect pack_all to have written these:
            ip_csv  = pair_dir / "ip.csv"
            oop_csv = pair_dir / "oop.csv"

            # Build S3-relative keys (RELATIVE to your vendor root, e.g. "data/vendor")
            # This matches SphIndex/_resolve_path which joins: s3_vendor + rel_path
            if ip_csv.exists():
                rows.append({
                    "stack_bb": stack,
                    "ctx": ctx,
                    "ip_pos": ip.upper(),
                    "oop_pos": oop.upper(),
                    "hero_pos": "IP",
                    "rel_path": f"sph/{ctx}/{stack}/{ip.upper()}_{oop.upper()}/ip.csv",
                    "abs_path": "",  # force S3 fetch (or local vendor_cache fallback)
                })
            if oop_csv.exists():
                rows.append({
                    "stack_bb": stack,
                    "ctx": ctx,
                    "ip_pos": ip.upper(),
                    "oop_pos": oop.upper(),
                    "hero_pos": "OOP",
                    "rel_path": f"sph/{ctx}/{stack}/{ip.upper()}_{oop.upper()}/oop.csv",
                    "abs_path": "",
                })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Scan SPH cache (ip.csv/oop.csv) into a manifest parquet.")
    ap.add_argument("--cache-root", type=Path, default=Path("data/vendor_cache/sph"),
                    help="Root with SRP/LIMP_SINGLE/LIMP_MULTI subdirs")
    ap.add_argument("--out", type=Path, default=Path("data/artifacts/sph_manifest.parquet"))
    args = ap.parse_args()

    rows: list[dict] = []
    for ctx in ("SRP", "LIMP_SINGLE", "LIMP_MULTI"):
        ctx_dir = args.cache_root / ctx
        if ctx_dir.exists():
            rows.extend(_rows_from_ctx_dir(ctx_dir, ctx))

    if not rows:
        raise SystemExit(f"No SPH ip/oop CSVs found under {args.cache_root}")

    df = pd.DataFrame(rows).sort_values(
        ["ctx", "stack_bb", "ip_pos", "oop_pos", "hero_pos"]
    ).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"✅ wrote SPH manifest → {args.out}")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()