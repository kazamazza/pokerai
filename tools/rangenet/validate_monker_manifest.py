#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

try:
    import polars as pl
except ImportError:
    print("Please `pip install polars`", file=sys.stderr)
    sys.exit(1)

VALID_POS = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}
VALID_ACT = {"FOLD", "CALL", "RAISE", "ALL_IN", "CHECK", "BET"}  # allow postflop verbs too

STACK_RE = re.compile(r"(?:^|/)(\d+)bb(?:/|$)")

def _pick_col(df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _parse_actions(val: Any) -> Optional[List[Dict[str, Any]]]:
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            # sometimes it's already python-literal-like
            try:
                import ast
                return ast.literal_eval(s)
            except Exception:
                return None
    return None

def validate_manifest_parquet(parquet_path: Path, vendor_root: Path, max_errors: int = 50) -> None:
    df = pl.read_parquet(str(parquet_path))

    # Column mapping (robust to naming differences)
    PATH_COL     = _pick_col(df, ["path", "abs_path", "file"])
    RELPATH_COL  = _pick_col(df, ["relpath", "rel_path", "relative_path"])
    HERO_COL     = _pick_col(df, ["hero_pos", "hero"])
    STACK_COL    = _pick_col(df, ["stack", "stack_bb", "bb"])
    ACTIONS_COL  = _pick_col(df, ["actions", "action_seq", "sequence"])
    HASH_COL     = _pick_col(df, ["hash", "sha1", "fingerprint", "id"])

    errors: List[str] = []
    seen_hash = set()
    coverage = Counter()

    for i, row in enumerate(df.iter_rows(named=True), 1):
        # Resolve file to check
        cand: Optional[Path] = None
        if PATH_COL and row.get(PATH_COL):
            p = Path(str(row.get(PATH_COL)))
            cand = p if p.is_absolute() else (vendor_root / p)
        elif RELPATH_COL and row.get(RELPATH_COL):
            cand = vendor_root / str(row.get(RELPATH_COL))

        if cand is not None and not cand.exists():
            errors.append(f"[{i}] Missing file: {cand}")

        # hero position
        hero = row.get(HERO_COL) if HERO_COL else None
        if hero not in VALID_POS:
            errors.append(f"[{i}] Invalid hero_pos={hero!r}")

        # actions payload
        acts = _parse_actions(row.get(ACTIONS_COL)) if ACTIONS_COL else None
        if acts is None:
            errors.append(f"[{i}] Missing/invalid actions in column={ACTIONS_COL or 'N/A'}")
        else:
            for j, step in enumerate(acts, 1):
                pos = step.get("pos")
                act = step.get("action")
                if pos not in VALID_POS:
                    errors.append(f"[{i}] actions[{j}] invalid pos={pos!r}")
                if act not in VALID_ACT:
                    errors.append(f"[{i}] actions[{j}] invalid action={act!r}")

        # stack vs relpath folder (e.g., ".../12bb/...")
        stack_val = row.get(STACK_COL) if STACK_COL else None
        rel = str(row.get(RELPATH_COL) or row.get(PATH_COL) or "")
        m = STACK_RE.search(rel)
        stack_from_rel = int(m.group(1)) if m else None
        if stack_val is not None and stack_from_rel is not None:
            try:
                sv = int(stack_val)
                if sv != stack_from_rel:
                    errors.append(f"[{i}] stack={sv} mismatch relpath-stack={stack_from_rel} in {rel}")
            except Exception:
                errors.append(f"[{i}] non-integer stack value: {stack_val!r}")

        # duplicate hash
        h = row.get(HASH_COL) if HASH_COL else None
        if h is not None:
            if h in seen_hash:
                errors.append(f"[{i}] Duplicate hash {h!r}")
            seen_hash.add(h)

        # coverage
        cov_stack = str(stack_from_rel if stack_from_rel is not None else stack_val or "?")
        if hero:
            coverage[(cov_stack, str(hero))] += 1

        if len(errors) >= max_errors:
            # collect but stop early printing; we’ll show cap at the end
            pass

    # Report
    total = df.height
    print(f"✅ rows checked: {total}")
    print(f"❌ problems   : {len(errors)}")
    for e in errors[:max_errors]:
        print("   ", e)
    if len(errors) > max_errors:
        print(f"... ({len(errors) - max_errors} more)")

    print("\nCoverage by (stack, hero_pos):")
    for (stack, hero), n in sorted(coverage.items(), key=lambda kv: (int(kv[0][0]) if str(kv[0][0]).isdigit() else 9999, kv[0][1])):
        print(f"  {stack:>4}bb {hero:>3}: {n}")

def main():
    ap = argparse.ArgumentParser(description="Validate Monker manifest (Parquet) consistency")
    ap.add_argument("--manifest", type=Path, default=Path("data/artifacts/monker_manifest.parquet"),
                    help="Path to manifest parquet")
    ap.add_argument("--vendor-root", type=Path, default=Path("data/vendor/monker"),
                    help="Root folder to resolve relpaths/files against")
    ap.add_argument("--max-errors", type=int, default=50)
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    validate_manifest_parquet(args.manifest, args.vendor_root, max_errors=args.max_errors)

if __name__ == "__main__":
    main()