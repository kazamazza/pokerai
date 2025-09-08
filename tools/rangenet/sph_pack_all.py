import argparse, json, sys, re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.utils.range_utils import parse_range_text_to_grid, parse_abs_text_to_vec169, \
    abs_text_to_vec169, vec169_to_monker_string
from infra.storage.s3_client import S3Client


def _read_grid_any(path: Path) -> np.ndarray:
    p = Path(path)
    # if this is one of your ABS .txt exports, use the ABS parser
    if p.suffix.lower() == ".txt":
        return parse_abs_text_to_vec169(p)
    # else keep existing behavior (CSV 13×13, JSON, etc.)
    return parse_range_text_to_grid(p)

def write_canonical_json(out_path: Path, stack_bb: int, ip_pos: str, oop_pos: str, ctx: str,
                         ip_169: np.ndarray, oop_169: np.ndarray):
    out = {
        "meta": {
            "source": "SPH",
            "version": "v2.0.9",
            "stack_bb": int(stack_bb),
            "ctx": str(ctx).upper(),
            "ip_pos": ip_pos,
            "oop_pos": oop_pos,
            "notes": "IP=open prob, OOP=defend prob (1-fold)"
        },
        "ip":  [float(x) for x in ip_169.tolist()],
        "oop": [float(x) for x in oop_169.tolist()],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

# --------- pack logic ---------

CTX_DIRS = ["SRP", "LIMP_SINGLE", "LIMP_MULTI"]

def discover_pairs(ctx_dir: Path) -> List[Tuple[int, str]]:
    pairs = []
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
            pairs.append((stack, pair_dir.name))
    return pairs

def pack_one(ctx: str, stack: int, pair: str,
             vendor_root: Path, cache_root: Path) -> Tuple[bool, Optional[str], Path]:
    in_dir = vendor_root / ctx / str(stack) / pair
    out_dir = cache_root / ctx / str(stack) / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    ip_open_path = in_dir / "ip_open.txt"
    if not ip_open_path.exists():
        return False, f"missing ip_open.txt in {in_dir}", out_dir / "ip.csv"

    # ✅ ABS .txt → 169 via the canonical helper
    ip_open_169 = parse_abs_text_to_vec169(ip_open_path)

    # OOP: prefer merged numeric defend; else sum ABS call/raises
    oop_def_path = in_dir / "oop_defend.csv"
    if oop_def_path.exists():
        # this is a numeric CSV grid (13x13) we produced earlier
        arr = np.loadtxt(oop_def_path, delimiter=",", dtype=np.float32)
        if arr.size != 169:
            return False, f"bad shape in {oop_def_path} (size={arr.size})", out_dir / "oop.csv"
        oop_def_169 = arr.reshape(169)
    else:
        call_path = in_dir / "oop_call.txt"
        r1_path   = in_dir / "oop_raise_s1.txt"
        r2_path   = in_dir / "oop_raise_s2.txt"
        missing = [p.name for p in (call_path, r1_path, r2_path) if not p.exists()]
        if missing:
            return False, f"missing {missing} in {in_dir} (and no oop_defend.csv)", out_dir / "oop.csv"

        v_call = parse_abs_text_to_vec169(call_path)
        v_r1   = parse_abs_text_to_vec169(r1_path)
        v_r2   = parse_abs_text_to_vec169(r2_path)
        oop_def_169 = np.clip(v_call + v_r1 + v_r2, 0.0, 1.0)

    # Write Monker-style strings (compact) for both sides
    (out_dir / "ip.csv").write_text(vec169_to_monker_string(ip_open_169), encoding="utf-8")
    (out_dir / "oop.csv").write_text(vec169_to_monker_string(oop_def_169), encoding="utf-8")
    return True, None, out_dir

def run_pack_all(
    vendor_root: Path,
    cache_root: Path,
    only_ctx: Optional[str] = None,
    only_stack: Optional[int] = None,
    *,
    upload_key_root: Optional[str] = None,   # e.g. "data/vendor/sph"
    s3: Optional["S3Client"] = None,         # your client already bound to a bucket
):
    def _join_key(*parts: str) -> str:
        return "/".join(str(p).strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != "")

    errors = []
    written = 0
    ctxs = [only_ctx] if only_ctx else CTX_DIRS

    for ctx in ctxs:
        ctx_dir = vendor_root / ctx
        if not ctx_dir.exists():
            continue

        for (stack, pair) in discover_pairs(ctx_dir):
            if only_stack is not None and stack != only_stack:
                continue

            ok, err, out_dir = pack_one(ctx, stack, pair, vendor_root, cache_root)
            if ok:
                print(f"✅ {ctx}/{stack}/{pair} → {out_dir}")
                written += 1

                # Optional S3 upload (keys only, bucket lives in S3Client)
                if upload_key_root and s3:
                    for fname in ("ip.csv", "oop.csv"):
                        local_path = out_dir / fname
                        if local_path.exists():
                            key = _join_key(upload_key_root, ctx, str(stack), pair, fname)
                            s3.upload_file(local_path, key)
                            print(f"   ⬆ uploaded {fname} → {key}")
            else:
                print(f"❌ {ctx}/{stack}/{pair}: {err}")
                errors.append((ctx, stack, pair, err))

    print(f"\nSummary: wrote={written}  errors={len(errors)}")
    if errors:
        print("Errors:")
        for ctx, stack, pair, err in errors:
            print(f"  - {ctx}/{stack}/{pair}: {err}")
    return written, errors


def main():
    ap = argparse.ArgumentParser(description="Batch pack SPH vendor ranges into Monker-style ip/oop CSVs")
    ap.add_argument("--vendor-root", type=Path, default=Path("data/vendor/sph"))
    ap.add_argument("--cache-root", type=Path, default=Path("data/vendor_cache/sph"))
    ap.add_argument("--ctx", type=str, default=None, choices=[None, *CTX_DIRS])
    ap.add_argument("--stack", type=int, default=None)
    ap.add_argument("--upload-key-root", type=str, default=None, help="S3 key root, e.g. 'data/vendor/sph'")
    args = ap.parse_args()

    s3 = S3Client() if args.upload_key_root else None

    run_pack_all(
        vendor_root=args.vendor_root,
        cache_root=args.cache_root,
        only_ctx=args.ctx,
        only_stack=args.stack,
        upload_key_root=args.upload_key_root,  # <- pass the key root only
        s3=s3,
    )

if __name__ == "__main__":
    main()