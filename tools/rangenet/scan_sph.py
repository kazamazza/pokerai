#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List, Dict
import pandas as pd

def _rows_from_ctx_dir(ctx_dir: Path, ctx: str, base: Path) -> List[Dict]:
    """
    Expect structure:
      base/CTX/STACK/IP_OOP/ip.csv
      base/CTX/STACK/IP_OOP/oop.csv
    Produce two rows per pair: hero=IP (ip.csv) and hero=OOP (oop.csv).
    rel_path is under "sph/..." so S3 key becomes data/vendor/<rel_path>.
    """
    rows: List[Dict] = []
    for stack_dir in sorted(ctx_dir.iterdir()):
        if not stack_dir.is_dir():
            continue
        try:
            stack = int(stack_dir.name)
        except Exception:
            continue

        for pair_dir in sorted(p for p in stack_dir.iterdir() if p.is_dir()):
            name = pair_dir.name  # "UTG_BB"
            try:
                ip, oop = name.split("_")
            except Exception:
                continue

            ip_path  = pair_dir / "ip.csv"
            oop_path = pair_dir / "oop.csv"
            if not ip_path.exists() or not oop_path.exists():
                # quietly skip incomplete pairs
                continue

            # rel_path must be relative to vendor root ("data/vendor"),
            # we standardize it as "sph/CTX/STACK/IP_OOP/<file>"
            rel_ip  = Path("sph") / ctx / str(stack) / name / "ip.csv"
            rel_oop = Path("sph") / ctx / str(stack) / name / "oop.csv"

            rows.append({
                "stack_bb": stack,
                "ctx": ctx,
                "ip_pos": ip.upper(),
                "oop_pos": oop.upper(),
                "hero_pos": ip.upper(),   # tells loader to pick IP side
                "rel_path": rel_ip.as_posix(),
                "abs_path": str(ip_path.resolve()),
            })
            rows.append({
                "stack_bb": stack,
                "ctx": ctx,
                "ip_pos": ip.upper(),
                "oop_pos": oop.upper(),
                "hero_pos": oop.upper(),  # tells loader to pick OOP side
                "rel_path": rel_oop.as_posix(),
                "abs_path": str(oop_path.resolve()),
            })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Scan SPH cache (ip.csv/oop.csv) into a manifest parquet.")
    ap.add_argument("--cache-root", type=Path, default=Path("data/vendor_cache/sph"),
                    help="Root with SRP/LIMP_SINGLE/LIMP_MULTI subdirs")
    ap.add_argument("--out", type=Path, default=Path("data/artifacts/sph_manifest.parquet"))
    args = ap.parse_args()

    base = args.cache_root
    rows: List[Dict] = []
    for ctx in ("SRP", "LIMP_SINGLE", "LIMP_MULTI"):
        ctx_dir = base / ctx
        if ctx_dir.exists():
            rows.extend(_rows_from_ctx_dir(ctx_dir, ctx, base))

    if not rows:
        raise SystemExit(f"No SPH ip/oop CSVs found under {base}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["ctx", "stack_bb", "ip_pos", "oop_pos", "hero_pos"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"✅ wrote SPH manifest → {args.out}")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()