import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.utils.range_utils import parse_abs_text_to_vec169


DEFAULT_VENDOR_ROOT = Path("data/vendor_cache/sph")
DEFAULT_CTX_DIRS = ["SRP", "LIMPED_SINGLE", "LIMPED_MULTI"]
LIMP_CTX = {"LIMPED_SINGLE", "LIMPED_MULTI"}

# Optional built-in SB limp list (≈ 47% combos ~ 80% hand-types).
# If you prefer to FORCE using only files, set USE_BUILTIN_SB_LIMP=False.
USE_BUILTIN_SB_LIMP = True
SB_LIMP_LIST_25BB = [
    "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
    "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
    "KQs","KJs","KTs","K9s","K8s","K7s","K6s","K5s","K4s","K3s","K2s",
    "QJs","QTs","Q9s","Q8s","Q7s","Q6s","Q5s","Q4s","Q3s","Q2s",
    "JTs","J9s","J8s","J7s","J6s","J5s","J4s","J3s","J2s",
    "T9s","T8s","T7s","T6s","T5s","T4s",
    "98s","97s","96s","95s","94s",
    "87s","86s","85s","84s",
    "76s","75s","74s",
    "65s","64s","63s",
    "54s","53s","52s",
    "43s",
    "AJo","ATo","A9o","A8o","A7o","A6o","A5o","A4o","A3o","A2o",
    "KQo","KJo","KTo","K9o",
    "QJo","QTo","Q9o",
    "JTo","J9o",
    "T9o","98o","87o","76o",
]
SB_LIMP_LIST_60BB = [
    # Pairs
    "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",

    # Suited A-x
    "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",

    # Suited K-x
    "KQs","KJs","KTs","K9s","K8s","K7s","K6s","K5s","K4s","K3s","K2s",

    # Suited Q-x
    "QJs","QTs","Q9s","Q8s","Q7s",

    # Suited J/T connectors
    "JTs","J9s","J8s",
    "T9s","T8s",

    # Suited connectors/gappers
    "98s","87s","76s","65s","54s","43s",

    # Off-suit A-x (trimmed)
    "AJo","ATo","A9o","A8o","A7o","A6o","A5o","A4o","A3o","A2o",

    # Off-suit broadways
    "KQo","KJo","KTo",
    "QJo","QTo",
    "JTo",

    # Off-suit connectors
    "T9o","98o","87o","76o",
]
SB_LIMP_LIST_100BB = [
    # Pairs
    "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",

    # Suited A-x
    "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",

    # Suited K-x
    "KQs","KJs","KTs","K9s","K8s","K7s","K6s","K5s","K4s","K3s","K2s",

    # Suited Q-x
    "QJs","QTs","Q9s","Q8s",

    # Suited J/T connectors
    "JTs","J9s","J8s",
    "T9s","T8s",

    # Suited connectors/gappers
    "98s","87s","76s","65s","54s","43s",

    # Off-suit A-x (tighter)
    "ATo","A9o","A8o","A7o","A6o","A5o","A4o","A3o","A2o",

    # Off-suit broadways (very top-heavy)
    "KQo","KJo",
    "QJo","JTo",

    # Off-suit connectors
    "T9o","98o",
]

# ----------------- IO HELPERS -----------------

def write_csv_grid(path: Path, vec169: np.ndarray) -> None:
    vec = np.clip(np.asarray(vec169).reshape(169), 0.0, 1.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in range(13):
            row = vec[r*13:(r+1)*13]
            f.write(",".join(f"{x:.6f}" for x in row) + "\n")

def build_hand_labels_169() -> List[str]:
    ranks = "AKQJT98765432"
    labels = []
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                labels.append(r1 + r2)        # pairs
            elif i < j:
                labels.append(r1 + r2 + "s")  # suited
            else:
                labels.append(r2 + r1 + "o")  # offsuit
    return labels

_LABELS_169 = build_hand_labels_169()
_INDEX_169 = {h: i for i, h in enumerate(_LABELS_169)}

def all_ones_vec169() -> np.ndarray:
    return np.ones(169, dtype=np.float32)

def zeros_vec169() -> np.ndarray:
    return np.zeros(169, dtype=np.float32)

def vec_from_hand_list(hands: List[str], weight: float = 1.0) -> np.ndarray:
    v = zeros_vec169()
    for raw in hands:
        h = raw.strip().upper()
        if not h:
            continue
        # Normalize formats like "AJOffsuit"/"ATSuited"
        h = (h
             .replace("OFFSUIT", "O")
             .replace("SUITED", "S"))
        if h in _INDEX_169:
            v[_INDEX_169[h]] = weight
            continue
        # Try to coerce e.g., "AK" -> both suited+offsuit? Assume offsuit if given.
        # We avoid expanding ambiguous labels to prevent accidental dilution.
        raise ValueError(f"Unrecognized hand label: '{raw}' (normalized: '{h}')")
    return v

def vec_from_grid_csv(path: Path) -> np.ndarray:
    arr = np.array(pd.read_csv(path, header=None), dtype=float)
    if arr.size != 169:
        raise ValueError(f"{path} shape={arr.shape}, expected 13x13")
    return arr.reshape(169).astype(np.float32)


def _iter_pairs(vendor_root: Path, ctx_dirs: List[str]) -> List[Tuple[str, Optional[int], Path]]:
    out: List[Tuple[str, Optional[int], Path]] = []
    for ctx in ctx_dirs:
        ctx_dir = vendor_root / ctx
        if not ctx_dir.exists():
            continue
        for stack_dir in sorted(ctx_dir.iterdir()):
            if not stack_dir.is_dir():
                continue
            try:
                stack = int(stack_dir.name)
            except Exception:
                stack = None
            for pair_dir in sorted(stack_dir.iterdir()):
                if pair_dir.is_dir():
                    out.append((ctx, stack, pair_dir))
    return out

def _pick_first(*cands: Optional[Path]) -> Optional[Path]:
    for p in cands:
        if p and isinstance(p, Path) and p.exists():
            return p
    return None

def _write_csv(path: Path, vec: np.ndarray, label: str) -> str:
    write_csv_grid(path, vec)
    return f"wrote {path.name} (nnz={int(np.count_nonzero(vec))}, sum={float(np.sum(vec)):.2f}, from {label})"

# ----------------- MERGE LOGIC -----------------
def vec_is_all_ones(v, tol: float = 1e-6) -> bool:
    """
    Returns True if a 169-length range vector is effectively all ones.
    Works for lists, NumPy arrays, or Pandas Series.

    Parameters
    ----------
    v : Sequence[float]
        The vector to check (length 169 expected).
    tol : float
        Tolerance for floating-point comparison. Defaults to 1e-6.

    Returns
    -------
    bool
        True if every element is within tol of 1.0.
    """
    # Convert to array for vectorized comparison
    arr = np.asarray(v, dtype=float)

    if arr.size != 169:
        # Warn if shape is wrong; better to fail early than silently
        print(f"[vec_is_all_ones] Warning: vector length {arr.size} != 169")
        return False

    # Check all elements are approximately 1.0
    return np.allclose(arr, 1.0, atol=tol)


def _materialize_limp_pair(pair_dir: Path, stack: Optional[int]) -> Tuple[bool, str]:
    """
    LIMPED_SINGLE / LIMPED_MULTI:
      - OOP (SB) limp subset -> oop.csv
      - IP  (BB) 100%        -> ip.csv (auto if missing)
    Accepts either .txt hand lists or 13x13 CSV grids.
    Canonical basenames: oop_call.txt, ip_check.txt
    """

    # prefer canonical; keep legacy as fallback
    sb_call = _pick_first(
        pair_dir / "oop_call.txt",   # canonical
        pair_dir / "sb_call.txt",    # legacy
        pair_dir / "oop.csv"         # pre-materialized grid
    )
    bb_check = _pick_first(
        pair_dir / "ip_check.txt",   # canonical
        pair_dir / "bb_check.txt",   # legacy
        pair_dir / "ip.csv"          # pre-materialized grid
    )

    if sb_call is None and not USE_BUILTIN_SB_LIMP:
        return False, "missing ['oop_call.txt' or 'sb_call.txt' or 'oop.csv'] and builtin limp disabled"

    # --- OOP build (SB limp subset) ---
    if sb_call is None and USE_BUILTIN_SB_LIMP:
        # choose builtin by stack (default to 25 if None/unknown)
        if stack == 60:
            src_list = SB_LIMP_LIST_60BB
        elif stack == 100:
            src_list = SB_LIMP_LIST_100BB
        else:
            src_list = SB_LIMP_LIST_25BB
        v_oop = vec_from_hand_list(src_list)
        msg_oop = _write_csv(pair_dir / "oop.csv", v_oop, f"BUILTIN_SB_LIMP_{stack or 25}bb")
    else:
        v_oop = (parse_abs_text_to_vec169(sb_call)
                 if sb_call.suffix.lower() == ".txt"
                 else vec_from_grid_csv(sb_call))
        msg_oop = _write_csv(pair_dir / "oop.csv", v_oop, sb_call.name)

    # --- IP build (BB = 100%) ---
    if bb_check is None:
        v_ip = all_ones_vec169()
        msg_ip = _write_csv(pair_dir / "ip.csv", v_ip, "AUTO_IP_100")
    else:
        v_ip = (parse_abs_text_to_vec169(bb_check)
                if bb_check.suffix.lower() == ".txt"
                else vec_from_grid_csv(bb_check))
        # sanity: force 100% for IP in limp pots
        if not vec_is_all_ones(v_ip):
            v_ip = all_ones_vec169()
            src = f"{bb_check.name}(!fixed_to_100)"
        else:
            src = bb_check.name
        msg_ip = _write_csv(pair_dir / "ip.csv", v_ip, src)

    return True, f"{msg_oop}; {msg_ip}"

def _materialize_srp_pair(pair_dir: Path) -> Tuple[bool, str]:
    """
    SRP:
      - OOP defend = call + up to two raises (clamped to 1.0)
      - IP = oop_call.txt or ip.csv (if exists)
    """
    call = _pick_first(pair_dir / "oop_call.txt")
    r1   = _pick_first(pair_dir / "oop_raise_s1.txt")
    r2   = _pick_first(pair_dir / "oop_raise_s2.txt")
    ip_open = _pick_first(pair_dir / "oop_call.txt", pair_dir / "ip.csv")

    if call is None:
        return False, "missing ['oop_call.txt']"

    # OOP defend
    v_call = parse_abs_text_to_vec169(call)
    v_r1   = parse_abs_text_to_vec169(r1) if r1 and r1.suffix.lower() == ".txt" else (vec_from_grid_csv(r1) if r1 else zeros_vec169())
    v_r2   = parse_abs_text_to_vec169(r2) if r2 and r2.suffix.lower() == ".txt" else (vec_from_grid_csv(r2) if r2 else zeros_vec169())

    defend = np.clip(v_call + v_r1 + v_r2, 0.0, 1.0)
    msg_def = _write_csv(pair_dir / "oop.csv", defend, "+".join([p.name for p in [call, r1, r2] if p]))

    # IP
    msg_ip = ""
    if ip_open is not None:
        if ip_open.suffix.lower() == ".txt":
            v_ip = parse_abs_text_to_vec169(ip_open)
        else:
            v_ip = vec_from_grid_csv(ip_open)
        msg_ip = "; " + _write_csv(pair_dir / "ip.csv", v_ip, ip_open.name)

    return True, msg_def + msg_ip

def _merge_one_pair_dir(pair_dir: Path, ctx: str, stack: Optional[int]) -> Tuple[bool, str]:
    ctx_up = str(ctx).upper()
    try:
        if ctx_up in LIMP_CTX:
            return _materialize_limp_pair(pair_dir, stack)
        else:
            return _materialize_srp_pair(pair_dir)
    except Exception as e:
        return False, f"merge failed: {e}"

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Materialize SPH ranges to 13x13 ip/oop CSVs for postflop.")
    ap.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT, help="Root containing context folders.")
    ap.add_argument("--ctx", nargs="*", default=DEFAULT_CTX_DIRS, help="Contexts to scan (e.g., SRP LIMPED_SINGLE)")
    ap.add_argument("--stacks", nargs="*", default=None, help="Restrict stacks (e.g., 25 60 100). Default: all found.")
    ap.add_argument("--pair-filter", type=str, default=None, help="Substring filter for pair dir names (e.g., SB_BB).")
    ap.add_argument("--no-builtin-limp", action="store_true", help="Disable builtin SB limp fallback list.")
    args = ap.parse_args()

    global USE_BUILTIN_SB_LIMP
    if args.no_builtin_limp:
        USE_BUILTIN_SB_LIMP = False

    vendor_root: Path = args.vendor_root
    ctx_dirs: List[str] = [c.upper() for c in args.ctx]

    pairs = _iter_pairs(vendor_root, ctx_dirs)
    if not pairs:
        print(f"Nothing found under {vendor_root}. Expected {ctx_dirs}/<stack>/<IP_OOP>/")
        sys.exit(0)

    stacks_filter = set(args.stacks) if args.stacks else None
    scanned = created = errors = 0

    for ctx, stack, pair_dir in pairs:
        if stacks_filter and (stack is None or str(stack) not in stacks_filter):
            continue
        if args.pair_filter and args.pair_filter not in pair_dir.name:
            continue

        scanned += 1
        ok, msg = _merge_one_pair_dir(pair_dir, ctx, stack)
        if ok:
            created += 1
            print(f"✅ {ctx}/{stack}/{pair_dir.name}: {msg}")
        else:
            errors += 1
            print(f"❌ {ctx}/{stack}/{pair_dir.name}: {msg}")

    print(f"\nDone. scanned={scanned} created={created} errors={errors}")
    if errors:
        sys.exit(1)

if __name__ == "__main__":
    main()