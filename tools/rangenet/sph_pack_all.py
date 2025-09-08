import argparse, json, sys, re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.etl.utils.range_format import vec169_to_monker_string
from infra.storage.s3_client import S3Client
from ml.config.types_hands import RANKS



# --------- utils you already have / or drop-in ---------

def zeros_169():
    return np.zeros(169, dtype=np.float32)

def _hand_to_index(code: str) -> int:
    code = code.strip()
    if len(code) == 2:  # pair e.g., "AA"
        r = RANKS.index(code[0])
        return r * 13 + r
    if len(code) == 3:  # e.g., "AKs"/"AKo"
        r1, r2, s = code[0], code[1], code[2]
        i = RANKS.index(r1)
        j = RANKS.index(r2)
        if s == "s":  # suited row-major upper triangle
            return i * 13 + j
        elif s == "o":  # offsuit lower triangle
            return j * 13 + i
    raise ValueError(f"Bad hand code: {code}")

def _to_compact_index(cards: str) -> int:
    # AhKh -> AKs, AdKc -> AKo, etc.
    if len(cards) != 4:
        raise ValueError(f"Bad card spec: {cards}")
    r1, s1, r2, s2 = cards[0], cards[1], cards[2], cards[3]
    if r1 == r2:
        return _hand_to_index(r1 + r2)
    suited = (s1 == s2)
    hi, lo = (r1, r2) if RANKS.index(r1) < RANKS.index(r2) else (r2, r1)
    return _hand_to_index(hi + lo + ("s" if suited else "o"))

def parse_range_text_to_grid(path: Path) -> np.ndarray:
    """
    Parse SPH/Monker-like text into flat 169 array (values 0..1).
    Supports:
      - 13x13 CSV/whitespace (169 numbers)
      - Flat 169 list
      - CARD:VALUE (AA:1.0,A2s:0.024,...)
      - [xx.xx]AhKh,...[/xx.xx] (groups; xx.xx can be % or 0..1)
      - JSON list of 169 or {"range":[...]}
    """
    txt = path.read_text(encoding="utf-8").strip()

    # JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, list) and len(obj) == 169:
            return np.array(obj, dtype=np.float32)
        if isinstance(obj, dict):
            if "range" in obj and len(obj["range"]) == 169:
                return np.array(obj["range"], dtype=np.float32)
            # dict CARD:VALUE
            vals = [0.0] * 169
            ok = False
            for k, v in obj.items():
                try:
                    idx = _hand_to_index(k)
                    vals[idx] = float(v)
                    ok = True
                except Exception:
                    pass
            if ok:
                return np.array(vals, dtype=np.float32)
    except Exception:
        pass

    # CARD:VALUE plain text
    if ":" in txt and any(h in txt for h in ["AA", "AKs", "72o"]):
        vals = [0.0] * 169
        for tok in re.split(r"[,\s]+", txt):
            if not tok or ":" not in tok:
                continue
            hand, val = tok.split(":")
            vals[_hand_to_index(hand)] = float(val)
        return np.array(vals, dtype=np.float32)

    # [xx.xx] ... [/xx.xx] grouped ABS style
    if "[" in txt and "]" in txt and "/" in txt:
        vals = [0.0] * 169
        for m in re.finditer(r"\[(.*?)\](.*?)\[/\1\]", txt, flags=re.S):
            raw = m.group(1).strip()
            try:
                v = float(raw)
                if v > 1.0:  # treat as percent
                    v = v / 100.0
            except Exception:
                continue
            hands_blob = m.group(2)
            hands = [h for h in re.split(r"[,\s]+", hands_blob) if h]
            for h in hands:
                vals[_to_compact_index(h)] = v
        return np.array(vals, dtype=np.float32)

    # grids / flat numbers
    toks = re.split(r"[,\s]+", txt)
    nums = []
    for t in toks:
        if not t:
            continue
        if t.endswith("%"):
            nums.append(float(t[:-1]) / 100.0)
        else:
            try:
                nums.append(float(t))
            except Exception:
                pass
    if len(nums) == 169:
        return np.array(nums, dtype=np.float32)

    # 13x13 CSV with headers? try pandas quickly
    try:
        df = pd.read_csv(path, header=None)
        arr = df.to_numpy(dtype=float)
        if arr.size == 169:
            return arr.reshape(169).astype(np.float32)
    except Exception:
        pass

    raise ValueError(f"Unrecognized range format: {path}")

def _read_grid_any(path: Path) -> np.ndarray:
    arr = parse_range_text_to_grid(path)
    if arr.shape != (169,):
        arr = arr.reshape(169)
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return arr

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

    ip_open_169 = _read_grid_any(ip_open_path)

    # merged defend first if present, else sum call+raises
    oop_def_path = in_dir / "oop_defend.csv"
    if oop_def_path.exists():
        oop_def_169 = _read_grid_any(oop_def_path)
    else:
        call_path = in_dir / "oop_call.txt"
        r1_path   = in_dir / "oop_raise_s1.txt"
        r2_path   = in_dir / "oop_raise_s2.txt"
        missing = [p.name for p in [call_path, r1_path, r2_path] if not p.exists()]
        if missing:
            return False, f"missing {missing} in {in_dir} (and no oop_defend.csv)", out_dir / "oop.csv"
        parts = [_read_grid_any(call_path), _read_grid_any(r1_path), _read_grid_any(r2_path)]
        oop_def_169 = np.clip(np.sum(parts, axis=0), 0.0, 1.0)

    # write Monker CSV strings
    (out_dir / "ip.csv").write_text(vec169_to_monker_string(ip_open_169.tolist()), encoding="utf-8")
    (out_dir / "oop.csv").write_text(vec169_to_monker_string(oop_def_169.tolist()), encoding="utf-8")

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