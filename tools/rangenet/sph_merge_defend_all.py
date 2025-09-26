import sys, json, re
from pathlib import Path
from typing import List, Tuple
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.utils.range_utils import parse_abs_text_to_vec169

VENDOR_ROOT = Path("data/vendor/sph")
CTX_DIRS = ("SRP", "LIMP_SINGLE", "LIMP_MULTI")

def write_csv_grid(path: Path, vec169: np.ndarray) -> None:
    vec = np.clip(vec169.reshape(169), 0.0, 1.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in range(13):
            row = vec[r*13:(r+1)*13]
            f.write(",".join(f"{x:.6f}" for x in row) + "\n")


def _iter_pairs() -> List[Tuple[str, int, Path]]:
    """Yield (ctx, stack, pair_dir) across the standard vendor layout."""
    out = []
    for ctx in CTX_DIRS:
        ctx_dir = VENDOR_ROOT / ctx
        if not ctx_dir.exists():
            continue
        for stack_dir in sorted(ctx_dir.iterdir()):
            if not stack_dir.is_dir():
                continue
            try:
                stack = int(stack_dir.name)
            except Exception:
                continue
            for pair_dir in sorted(stack_dir.iterdir()):
                if pair_dir.is_dir():
                    out.append((ctx, stack, pair_dir))
    return out

def _merge_one_pair_dir(pair_dir: Path, ctx: str) -> Tuple[bool, str]:
    """
    Merge (call + two raises) -> oop_defend.csv in this pair dir.
    Returns (created, message).
    """
    call = pair_dir / "oop_call.txt"
    r1   = pair_dir / "oop_raise_s1.txt"
    r2   = pair_dir / "oop_raise_s2.txt"
    out  = pair_dir / "oop_defend.csv"

    missing = [p.name for p in (call, r1, r2) if not p.exists()]
    if missing:
        return False, f"missing {missing}"

    try:
        v_call = parse_abs_text_to_vec169(call)
        v_r1   = parse_abs_text_to_vec169(r1)
        v_r2   = parse_abs_text_to_vec169(r2)
        defend = np.clip(v_call + v_r1 + v_r2, 0.0, 1.0)
        write_csv_grid(out, defend)
        nnz = int(np.count_nonzero(defend))
        s = float(defend.sum())
        return True, f"wrote {out.name} (nnz={nnz}, sum={s:.2f})"
    except Exception as e:
        return False, f"merge failed: {e}"


def main():
    scanned = 0
    created = 0
    skipped = 0
    errors  = 0

    pairs = _iter_pairs()
    if not pairs:
        print(f"Nothing found under {VENDOR_ROOT}. Expected {CTX_DIRS}/<stack>/<IP_OOP>/")
        sys.exit(0)

    for ctx, stack, pair_dir in pairs:
        scanned += 1
        ok, msg = _merge_one_pair_dir(pair_dir, ctx)
        if ok:
            created += 1
            print(f"✅ {ctx}/{stack}/{pair_dir.name}: {msg}")
        else:
            # If already has oop_defend.csv and we didn’t overwrite (we always overwrite here),
            # we’d call it skipped; otherwise call it error.
            if "missing" in msg:
                errors += 1
                print(f"❌ {ctx}/{stack}/{pair_dir.name}: {msg}")
            else:
                # treat other non-fatal as error too
                errors += 1
                print(f"❌ {ctx}/{stack}/{pair_dir.name}: {msg}")

    print(f"\nDone. scanned={scanned} created={created} errors={errors}")

if __name__ == "__main__":
    main()