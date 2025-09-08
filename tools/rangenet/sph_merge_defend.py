#!/usr/bin/env python3
import argparse, re, sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.utils.range_utils import parse_range_text_to_grid

RANKS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
R2I = {r:i for i,r in enumerate(RANKS)}

ABS_BLOCK_RE = re.compile(
    r"\[\s*(?P<pct>[+-]?\d+(?:\.\d+)?)\s*\](?P<body>.*?)\[/\s*(?P=pct)\s*\]",
    re.DOTALL
)

SUITS = set("cdhs")

def _load_grid_like(path: Path) -> Optional[np.ndarray]:
    """
    Try to read a 13x13 numeric grid (CSV or whitespace). Accepts 0..1 or percents.
    Returns np.ndarray shape (169,) or None if this isn't a plain grid.
    """
    txt = path.read_text(encoding="utf-8").strip()
    # collect numbers (allow percents)
    toks = re.split(r"[,\s]+", txt)
    vals: List[float] = []
    for t in toks:
        if not t:
            continue
        if re.fullmatch(r"[+-]?\d+(\.\d+)?%", t):
            vals.append(float(t[:-1]) / 100.0)
        elif re.fullmatch(r"[+-]?\d+(\.\d+)?", t):
            vals.append(float(t))
        else:
            # likely not a raw grid (maybe CARD:VALUE format)
            return None
    if len(vals) == 169:
        arr = np.array(vals, dtype=np.float64)
        # normalize if this looks like percentages > 1.0
        if arr.max() > 1.0:
            arr = arr / 100.0
        # clip for safety
        arr = np.clip(arr, 0.0, 1.0)
        return arr
    return None

def _card_to_rc(card: str) -> Tuple[int,int]:
    """
    Map hand like 'AKs','AJo','QQ' to (row, col) in 13x13 grid with RANKS order.
    Convention: diagonal = pairs; upper triangle (row < col) suited; lower offsuit.
    """
    card = card.strip()
    m = re.fullmatch(r"([AKQJT98765432])([AKQJT98765432])([so]?)", card)
    if not m:
        raise ValueError(f"Bad card token: {card}")
    r1, r2, sf = m.group(1), m.group(2), m.group(3) or ""
    i = R2I[r1]; j = R2I[r2]
    if i == j:
        # pair, ignore s/o flag
        return (i, j)
    if sf == "s":
        # suited lives in upper triangle -> enforce row < col
        if i > j:
            i, j = j, i
    elif sf == "o":
        # offsuit lives in lower triangle -> enforce row > col
        if i < j:
            i, j = j, i
    else:
        # no flag: pick conventional location (upper for suited combos with r1<r2)
        # We assume monotone AK (no flag) means suited in many dumps; if ambiguous,
        # prefer upper triangle for i<j.
        if i > j:
            i, j = j, i
    return (i, j)

def _load_dict_card_values(path: Path) -> Optional[np.ndarray]:
    """
    Parse 'CARD:VALUE' CSV/whitespace like 'AA:1.0,A5s:0.25,...' into 13x13 grid.
    Missing combos default to 0. Returns flat (169,) or None if not recognized.
    """
    txt = path.read_text(encoding="utf-8").strip()
    if ":" not in txt:
        return None
    grid = np.zeros((13,13), dtype=np.float64)
    # tolerate comma or whitespace separated
    parts = re.split(r"[,\s]+", txt)
    saw_any = False
    for p in parts:
        if not p or ":" not in p:
            continue
        k, v = p.split(":", 1)
        k = k.strip()
        v = v.strip().rstrip("%")
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0
            val = float(np.clip(val, 0.0, 1.0))
            r, c = _card_to_rc(k)
            grid[r, c] = val
            saw_any = True
        except Exception:
            # ignore unparseable tokens
            continue
    return grid.reshape(-1) if saw_any else None


def _rank_from_combo_token(tok: str) -> Optional[Tuple[str, str, bool]]:
    """
    Parse a single explicit combo like 'Kh2h', 'AdKd', '2d2h' into (r1, r2, suited).
    Returns None if not recognized.
    """
    tok = tok.strip()
    if len(tok) < 4:
        return None
    r1, s1, r2, s2 = tok[0], tok[1], tok[2], tok[3]
    if r1 not in R2I or r2 not in R2I or s1 not in SUITS or s2 not in SUITS:
        return None
    suited = (r1 != r2) and (s1 == s2)
    return (r1, r2, suited)

def _class_from_combo(tok: str) -> Optional[str]:
    """
    Convert an explicit suited-offsuited combo into a class label like 'A5s','KQo','QQ'.
    """
    parsed = _rank_from_combo_token(tok)
    if not parsed:
        return None
    r1, r2, suited = parsed
    if r1 == r2:
        return f"{r1}{r2}"  # pair
    # order ranks high-low by our rank order
    i, j = R2I[r1], R2I[r2]
    hi, lo = (r1, r2) if i < j else (r2, r1)
    return f"{hi}{lo}{'s' if suited else 'o'}"


def _load_sph_abs_strategy(txt: str) -> Optional[np.ndarray]:
    """
    Parse SPH 'Copy Abs. Strategy' text with blocks:
      [28.54] Kh2h, Kd2d, ... [/28.54], [0.02] ... [/0.02], ...
    Values may be percents (>1.0) or already 0..1.
    Returns flat (169,) in 0..1, or None if the text doesn't look like this format.
    """
    if "[" not in txt or "]" not in txt or "/" not in txt:
        return None

    # accumulate per-class sums and counts over explicit combos
    class_sum = { }   # class -> sum of values over listed combos
    class_cnt = { }   # class -> number of combos seen
    saw_block = False

    for m in ABS_BLOCK_RE.finditer(txt):
        saw_block = True
        val_raw = float(m.group("pct"))
        val = val_raw / 100.0 if val_raw > 1.0 else val_raw
        body = m.group("body")
        # split by commas
        combos = [t.strip() for t in re.split(r"[,\n]+", body) if t.strip()]
        for c in combos:
            cls = _class_from_combo(c)
            if not cls:
                continue
            class_sum[cls] = class_sum.get(cls, 0.0) + val
            class_cnt[cls] = class_cnt.get(cls, 0) + 1

    if not saw_block:
        return None

    # build 13x13 grid by averaging per class, default 0
    grid = np.zeros((13,13), dtype=np.float64)
    for cls, s in class_sum.items():
        cnt = max(1, class_cnt.get(cls, 1))
        avg = float(np.clip(s / cnt, 0.0, 1.0))
        try:
            r, c = _card_to_rc(cls)
            grid[r, c] = avg
        except Exception:
            # ignore any strange remnants
            continue

    return grid.reshape(-1)


def load_strategy(path: Path) -> np.ndarray:
    """
    Load a preflop strategy into flat 169 array in 0..1.
    Supports:
      - 13x13 grid (csv/whitespace, 0..1 or percents)
      - CARD:VALUE dictionary (0..1 or percents), e.g. 'A5s:0.25'
      - SPH Abs. Strategy blocks: [value] combos [/value]
    """
    path = Path(path)
    txt = path.read_text(encoding="utf-8").strip()

    # 1) plain 13x13 grid
    arr = _load_grid_like(path)
    if arr is not None:
        return arr.astype(np.float64)

    # 2) CARD:VALUE dictionary
    arr = _load_dict_card_values(path)
    if arr is not None:
        return arr.astype(np.float64)

    # 3) SPH Abs. Strategy blocks
    arr = _load_sph_abs_strategy(txt)
    if arr is not None:
        return arr.astype(np.float64)

    raise ValueError(f"Unrecognized format in {path} (expected 13x13 grid, CARD:VALUE, or SPH Abs. Strategy blocks)")

def write_csv_grid(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    a = arr.reshape(13,13)
    with path.open("w", encoding="utf-8") as f:
        for i in range(13):
            row = a[i]
            f.write(",".join(f"{x:.6f}" for x in row) + "\n")

def auto_discover_pair_dir(pair_dir: Path) -> Tuple[Path, List[Path]]:
    """
    Find call and up to two raises inside pair_dir. Expected filenames:
      oop_call.txt / .csv
      oop_raise_s1.txt / .csv
      oop_raise_s2.txt / .csv
    Falls back to any files matching patterns if exact names aren’t present.
    """
    pair_dir = Path(pair_dir)
    candidates = {
        "call": ["oop_call.txt","oop_call.csv"],
        "r1":   ["oop_raise_s1.txt","oop_raise_s1.csv"],
        "r2":   ["oop_raise_s2.txt","oop_raise_s2.csv"],
    }
    def _first_existing(names):
        for n in names:
            p = pair_dir / n
            if p.exists():
                return p
        return None

    call = _first_existing(candidates["call"])
    r1   = _first_existing(candidates["r1"])
    r2   = _first_existing(candidates["r2"])

    if call is None:
        # try glob fallback
        g = sorted(list(pair_dir.glob("oop_call.*")))
        if g:
            call = g[0]
    raises: List[Path] = []
    for pat in ("oop_raise_s1.*", "oop_raise_s2.*"):
        g = sorted(list(pair_dir.glob(pat)))
        if g:
            raises.append(g[0])

    if call is None:
        raise FileNotFoundError(f"No call file found in {pair_dir} (expected oop_call.txt/.csv)")
    if not raises:
        raise FileNotFoundError(f"No raise files found in {pair_dir} (expected oop_raise_s1/2 .txt/.csv)")
    # keep only first two
    raises = raises[:2]
    return call, raises

def merge_defend(call_path: Path, raise_paths: List[Path], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    defend = w_call*call + sum_i w_i*raise_i, clipped to [0,1]
    All inputs are in 0..1 per combo. weights default to 1.
    """
    call = load_strategy(call_path)
    parts = [load_strategy(p) for p in raise_paths]
    if weights is None:
        weights = [1.0] * (1 + len(parts))
    if len(weights) != 1 + len(parts):
        raise ValueError("weights length must equal 1 + number of raises")
    out = weights[0] * call
    for w, arr in zip(weights[1:], parts):
        out = out + w * arr
    out = np.clip(out, 0.0, 1.0)
    return out

CTX_DIRS = ("SRP", "LIMP_SINGLE", "LIMP_MULTI")

def _looks_ok_grid(arr) -> bool:
    try:
        import numpy as np
        a = np.asarray(arr)
        return a.size == 169 and np.isfinite(a).all()
    except Exception:
        return False

def _require_exists(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")

def _pair_has_required_files(pair_dir: Path, ctx: str, strict_multi: bool) -> tuple[Path, list[Path]]:
    """
    Return (call_path, [raise_paths]) if files exist. Raises if missing.
    """
    call = pair_dir / "oop_call.txt"
    r1   = pair_dir / "oop_raise_s1.txt"
    r2   = pair_dir / "oop_raise_s2.txt"
    _require_exists(call, "OOP call")
    _require_exists(r1, "OOP raise s1")
    _require_exists(r2, "OOP raise s2")

    # Optional sanity for LIMP_MULTI: ensure we actually captured from the caller branch
    if ctx == "LIMP_MULTI" and strict_multi:
        ic = pair_dir / "ip_call.txt"
        _require_exists(ic, "IP overlimp (ip_call.txt) for LIMP_MULTI")

    return call, [r1, r2]

def _merge_one_pair_dir(pair_dir: Path, ctx: str, *, overwrite: bool, strict_multi: bool, weights: list[float] | None):
    out = pair_dir / "oop_defend.csv"
    if out.exists() and not overwrite:
        return False, f"skip (exists) {out}"

    # discover/check required
    call, raises = _pair_has_required_files(pair_dir, ctx, strict_multi)

    # parse and check each file early to fail fast / better messages
    call_arr = parse_range_text_to_grid(call)
    raise_arrs = [parse_range_text_to_grid(p) for p in raises]
    if not _looks_ok_grid(call_arr):
        raise ValueError(f"Bad grid in {call}")
    for pth, arr in zip(raises, raise_arrs):
        if not _looks_ok_grid(arr):
            raise ValueError(f"Bad grid in {pth}")

    defend = merge_defend(call, raises, weights=weights)  # your existing function reads files itself
    if not _looks_ok_grid(defend):
        raise ValueError(f"Merged grid invalid for {pair_dir}")

    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv_grid(out, defend)
    return True, f"merged → {out}"

def _iter_pair_dirs(root: Path):
    """
    Yield (ctx, stack, pair_dir) under <root>/sph/<CTX>/<STACK>/<IP_OOP>/
    """
    sph_root = root / "sph"
    if not sph_root.exists():
        return
    for ctx in CTX_DIRS:
        ctx_dir = sph_root / ctx
        if not ctx_dir.exists():
            continue
        for stack_dir in sorted(ctx_dir.iterdir()):
            if not stack_dir.is_dir():
                continue
            for pair_dir in sorted(stack_dir.iterdir()):
                if pair_dir.is_dir():
                    yield ctx, stack_dir.name, pair_dir

def main():
    ap = argparse.ArgumentParser(description="Merge OOP defend (call + raises) into a single 13x13 CSV (0..1).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pair-dir", type=Path,
                      help="A single pair directory containing oop_call.txt and oop_raise_s{1,2}.txt")
    mode.add_argument("--call", type=Path, help="Explicit call file (.txt/.csv)")
    mode.add_argument("--root", type=Path,
                      help="Root data/vendor directory to batch all pairs: <root>/sph/<CTX>/<STACK>/<IP_OOP>/")

    ap.add_argument("--raises", type=Path, nargs="+",
                    help="One or two raise files when using --call mode")
    ap.add_argument("--out", type=Path,
                    help="Output CSV when using --pair-dir/--call (single-target mode)")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="Optional weights: first for call, then each raise (defaults all 1.0)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing oop_defend.csv")
    ap.add_argument("--strict-multi", action="store_true",
                    help="In LIMP_MULTI, require ip_call.txt to exist (ensures you pulled from the caller branch)")
    ap.add_argument("--dry-run", action="store_true", help="Scan and report, but do not write files")

    args = ap.parse_args()

    # ---- single pair: --pair-dir OR --call+--raises ----
    if args.pair_dir or args.call:
        if args.pair_dir:
            if not args.out:
                ap.error("--out is required in --pair-dir mode")
            # Guess ctx from path parts if present (best-effort; only used for strict-multi)
            parts = [p.name for p in args.pair_dir.resolve().parents]
            ctx = "LIMP_MULTI" if "LIMP_MULTI" in parts else ("LIMP_SINGLE" if "LIMP_SINGLE" in parts else "SRP")
            if args.dry_run:
                print(f"[dry] would merge pair {args.pair_dir} (ctx={ctx}) → {args.out}")
                return
            # Use auto-discovery inside the dir
            created, msg = _merge_one_pair_dir(args.pair_dir, ctx,
                                               overwrite=args.overwrite,
                                               strict_multi=args.strict_multi,
                                               weights=args.weights)
            print(("✅ " if created else "⏭️  ") + msg)
            return

        # explicit files
        if not args.raises:
            ap.error("--raises is required when using --call")
        if not args.out:
            ap.error("--out is required when using --call")
        if args.dry_run:
            print(f"[dry] would merge (call={args.call}, raises={args.raises[:2]}) → {args.out}")
            return
        defend = merge_defend(args.call, args.raises[:2], weights=args.weights)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_csv_grid(args.out, defend)
        print(f"✅ merged defend → {args.out}")
        return

    # ---- batch mode: --root ----
    root = args.root
    if not root:
        ap.error("missing --root")
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    total = 0
    merged = 0
    skipped = 0
    errors = 0

    for ctx, stack, pair_dir in _iter_pair_dirs(root):
        total += 1
        try:
            out = pair_dir / "oop_defend.csv"
            if out.exists() and not args.overwrite:
                skipped += 1
                print(f"⏭️  {ctx}/{stack}/{pair_dir.name}: exists → {out}")
                continue
            if args.dry_run:
                print(f"[dry] would merge {ctx}/{stack}/{pair_dir.name}")
                continue
            created, msg = _merge_one_pair_dir(pair_dir, ctx,
                                               overwrite=args.overwrite,
                                               strict_multi=args.strict_multi,
                                               weights=args.weights)
            if created:
                merged += 1
            else:
                skipped += 1
            print(("✅ " if created else "⏭️  ") + f"{ctx}/{stack}/{pair_dir.name}: {msg}")
        except Exception as e:
            errors += 1
            print(f"❌ {ctx}/{stack}/{pair_dir.name}: {e}", file=sys.stderr)

    print(f"\nDone. scanned={total} merged={merged} skipped={skipped} errors={errors}")

if __name__ == "__main__":
    main()