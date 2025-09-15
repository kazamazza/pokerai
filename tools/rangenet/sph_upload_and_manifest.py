import argparse, os, json
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client


def find_packed_jsons(local_root: Path) -> List[Path]:
    return [p for p in local_root.rglob("*.json") if p.is_file()]

def infer_ctx_stack_pair(p: Path, local_root: Path):
    """
    Expect files like: <local_root>/SRP/25/UTG_BB.json  (or LIMP_SINGLE/LIMP_MULTI)
    Returns ctx, stack(int), ip_pos, oop_pos and a rel_path beginning with 'sph/...'
    """
    rel = p.relative_to(local_root).as_posix()
    parts = rel.split("/")
    if len(parts) < 3:
        raise ValueError(f"Unexpected path depth for SPH file: {p}")
    ctx = parts[0].upper()              # SRP / LIMP_SINGLE / LIMP_MULTI
    stack = int(parts[1])               # 25 / 60 / 100 / 150
    base = parts[-1]                    # e.g. UTG_BB.json
    stem = Path(base).stem              # UTG_BB
    if "_" not in stem:
        raise ValueError(f"Expected file name like IP_OOP.json, got: {base}")
    ip_pos, oop_pos = stem.split("_", 1)
    # Canonical rel_path under vendor_cache: sph/CTX/STACK/IP_OOP.json
    rel_path = f"sph/{ctx}/{stack}/{ip_pos}_{oop_pos}.json"
    return ctx, stack, ip_pos.upper(), oop_pos.upper(), rel_path

def upload_files(s3: S3Client, s3_prefix: str, local_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    files = find_packed_jsons(local_root)
    if not files:
        raise SystemExit(f"No SPH JSON files found under {local_root}")

    s3_prefix = s3_prefix.rstrip("/")
    for p in files:
        ctx, stack, ip, oop, rel_path = infer_ctx_stack_pair(p, local_root)
        s3_key = "/".join([s3_prefix, rel_path])  # e.g. s3://bucket/data/vendor_cache/sph/SRP/25/UTG_BB.json

        # Upload
        s3.upload_file(str(p), s3_key)

        # Manifest rows: one for IP and one for OOP (both point to same rel_path)
        rows.append({
            "stack_bb": int(stack),
            "ip_pos": ip,
            "oop_pos": oop,
            "ctx": ctx,
            "hero_pos": "IP",
            "rel_path": rel_path,
            "abs_path": "",   # remote only; loader will download to cache_dir
        })
        rows.append({
            "stack_bb": int(stack),
            "ip_pos": ip,
            "oop_pos": oop,
            "ctx": ctx,
            "hero_pos": "OOP",
            "rel_path": rel_path,
            "abs_path": "",
        })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Upload packed SPH ranges to S3 and write sph_manifest.parquet")
    ap.add_argument("--local-root", type=Path, default=Path("data/vendor_cache/sph"),
                    help="Root folder containing packed SPH JSONs (ctx/stack/IP_OOP.json)")
    ap.add_argument("--s3-prefix", type=str, required=True,
                    help="Base S3 prefix where files will be stored, e.g. s3://bucket/data/")
    ap.add_argument("--manifest-out", type=Path, default=Path("data/artifacts/sph_manifest.parquet"),
                    help="Path to write the SPH manifest parquet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.local_root.exists():
        raise SystemExit(f"Local root not found: {args.local_root}")

    s3 = S3Client()

    rows = []
    files = find_packed_jsons(args.local_root)
    if not files:
        raise SystemExit(f"No SPH JSON files under {args.local_root}")

    for p in files:
        ctx, stack, ip, oop, rel_path = infer_ctx_stack_pair(p, args.local_root)
        s3_key = "/".join([args.s3_prefix.rstrip("/"), rel_path])
        print(f"→ {p}  →  {s3_key}")
    if args.dry_run:
        print("\n(dry-run) nothing uploaded, manifest not written")
        return

    rows = upload_files(s3, args.s3_prefix, args.local_root)
    df = pd.DataFrame(rows)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.manifest_out, index=False)
    print(f"\n✅ uploaded {len(rows)//2} SPH files (IP+OOP rows: {len(rows)})")
    print(f"💾 wrote manifest → {args.manifest_out}")

if __name__ == "__main__":
    main()