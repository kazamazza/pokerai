#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

def scan(root: Path, out_parquet: Path):
    rows = []
    for p in root.rglob("*.json"):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        meta = obj.get("meta", {})
        stack = int(meta.get("stack_bb", 0))
        ctx   = str(meta.get("ctx", "SRP")).upper()
        ip    = str(meta.get("ip_pos", "")).upper()
        oop   = str(meta.get("oop_pos", "")).upper()
        if not (stack and ip and oop):
            continue
        rel = p.relative_to(root.parent) if p.is_absolute() and p.as_posix().find(root.parent.as_posix()) == 0 else p
        rel = rel.as_posix()
        abs_path = str(p.resolve())
        rows.append({"stack_bb":stack,"ip_pos":ip,"oop_pos":oop,"ctx":ctx,"hero_pos":"IP","rel_path":rel,"abs_path":abs_path})
        rows.append({"stack_bb":stack,"ip_pos":ip,"oop_pos":oop,"ctx":ctx,"hero_pos":"OOP","rel_path":rel,"abs_path":abs_path})
    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"✅ wrote manifest: {out_parquet} rows={len(df):,}")

def main():
    ap = argparse.ArgumentParser(description="Scan canonical SPH range files → sph_manifest.parquet")
    ap.add_argument("--cache-root", type=Path, default=Path("data/vendor/sph"),
                    help="Root folder containing SPH canonical files")
    ap.add_argument("--out", type=Path, default=Path("data/vendor_cache/sph_manifest.parquet"))
    args = ap.parse_args()
    scan(args.cache_root, args.out)

if __name__ == "__main__":
    main()