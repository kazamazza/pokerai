#!/usr/bin/env python3
import sys, json, re, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import polars as pl

# --- constants ---
POS_NAMES = {"UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB", "EP", "MP", "BU"}  # include common aliases
RANKS = set(list("23456789TJQKA"))
SUITS = set(list("shdc"))  # ♠ ♥ ♦ ♣ → s h d c

REQUIRED_COLS = [
    "street",
    "pot_bb",
    "effective_stack_bb",
    "board",
    "board_cluster_id",
    "range_ip",
    "range_oop",
    "positions",
    "bet_sizing_id",
    "accuracy",
    "max_iter",
    "allin_threshold",
    "solver_version",
    "sha1",
    "s3_key",
    "node_key",
    "weight",
]

# --- helpers ---
def is_flop_board_str(b: str) -> bool:
    """Expect exactly 3 cards concatenated, e.g. 'QsJh2h' (len=6), ranks A..2, suits shdc."""
    if not isinstance(b, str) or len(b) != 6:
        return False
    try:
        for i in range(0, 6, 2):
            r, s = b[i], b[i+1]
            if r not in RANKS or s not in SUITS:
                return False
        # ensure no duplicate exact cards like 'QsQs2h'
        cards = {b[0:2], b[2:4], b[4:6]}
        return len(cards) == 3
    except Exception:
        return False

def split_positions(s: str) -> Tuple[str, str] | None:
    """Expect 'IPposvOOPpos' concrete pair like 'BTN_BB' or 'SBvBB'."""
    if not isinstance(s, str) or "v" not in s:
        return None
    a, b = s.split("v", 1)
    if a in POS_NAMES and b in POS_NAMES:
        return a, b
    return None

def validate_manifest(path: Path, strict: bool = False, sample: int | None = 5) -> int:
    if not path.exists():
        print(f"❌ file not found: {path}", file=sys.stderr)
        return 2

    df = pl.read_parquet(str(path))
    print(f"Loaded: {path}  shape={df.shape}")

    # Required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return 1

    # Street must be flop-only (1)
    bad_street = df.filter(pl.col("street") != 1).height
    if bad_street:
        print(f"❌ Found {bad_street} rows with street != 1 (flop-only manifest expected).")
        if strict: return 1

    # Types/sanity for numerics
    num_checks = [
        ("pot_bb", 0.0, float("inf")),
        ("effective_stack_bb", 0.0, float("inf")),
        ("accuracy", 0.0, 5.0),
        ("max_iter", 1.0, float("inf")),
        ("allin_threshold", 0.0, 1.0),
        ("weight", 0.0, float("inf")),
        ("board_cluster_id", 0.0, float("inf")),
    ]
    bad_num = 0
    for col, lo, hi in num_checks:
        q = df.filter((pl.col(col) < lo) | (pl.col(col) > hi))
        if q.height:
            print(f"❌ Column {col} out of bounds [{lo},{hi}] → {q.height} rows")
            bad_num += q.height
    if bad_num and strict:
        return 1

    # Boards validity
    boards = df.get_column("board").to_list()
    bad_boards = [b for b in boards if not is_flop_board_str(b)]
    if bad_boards:
        print(f"❌ Invalid board strings (showing up to 5): {bad_boards[:5]}  total={len(bad_boards)}")
        if strict: return 1

    # Positions validity
    pos_list = df.get_column("positions").to_list()
    bad_pos = [p for p in pos_list if split_positions(p) is None]
    if bad_pos:
        print(f"❌ Invalid positions format (expect 'POSvPOS', POS in {sorted(POS_NAMES)}): "
              f"showing up to 5: {bad_pos[:5]}  total={len(bad_pos)}")
        if strict: return 1

    # Ranges presence (don’t deeply parse compact strings here, just non-empty)
    for col in ("range_ip", "range_oop"):
        n_empty = df.filter((pl.col(col).is_null()) | (pl.col(col).cast(pl.Utf8).str.len_chars() == 0)).height
        if n_empty:
            print(f"❌ Empty {col} in {n_empty} row(s)")
            if strict: return 1

    # Uniqueness checks
    dup_sha = (
        df.group_by("sha1")
          .agg(pl.len().alias("n"))
          .filter(pl.col("n") > 1)
    )
    if dup_sha.height:
        print(f"❌ Duplicate sha1 hashes: {dup_sha.height} (first few):")
        print(dup_sha.head().to_pandas().to_string(index=False))
        if strict: return 1

    dup_s3 = (
        df.group_by("s3_key")
          .agg(pl.len().alias("n"))
          .filter(pl.col("n") > 1)
    )
    if dup_s3.height:
        print(f"❌ Duplicate s3_key entries: {dup_s3.height} (first few):")
        print(dup_s3.head().to_pandas().to_string(index=False))
        if strict: return 1

    # Soft coverage summary
    print("\nCoverage summary:")
    by_stack = df.group_by("effective_stack_bb").agg(pl.len().alias("jobs")).sort("effective_stack_bb")
    print("  by stack:\n" + by_stack.to_pandas().to_string(index=False))

    by_pot = df.group_by("pot_bb").agg(pl.len().alias("jobs")).sort("pot_bb")
    print("\n  by pot:\n" + by_pot.to_pandas().to_string(index=False))

    # positions breakdown
    by_pos = df.group_by("positions").agg(pl.len().alias("jobs")).sort("jobs", descending=True)
    print("\n  by positions:\n" + by_pos.to_pandas().to_string(index=False))

    # boards per cluster distribution
    bpc = df.group_by("board_cluster_id").agg(pl.len().alias("boards")).sort("board_cluster_id")
    if bpc.height:
        boards_min = int(bpc.get_column("boards").min())
        boards_mean = float(bpc.get_column("boards").mean())
        boards_max = int(bpc.get_column("boards").max())
        print(f"\nBoards per cluster: min={boards_min} mean={boards_mean:.2f} max={boards_max} "
              f"(clusters covered={bpc.height})")

    print(f"\n✅ Manifest looks good enough. total_rows={df.height}")
    # Optional sampling
    if sample:
        import random
        n = min(sample, df.height)
        idxs = random.sample(range(df.height), n)
        print(f"\nSamples ({n}):")
        cols_show = ["effective_stack_bb","pot_bb","positions","board","board_cluster_id",
                     "bet_sizing_id","accuracy","max_iter"]
        for i in idxs:
            row = df[i]
            meta = {c: row[c] for c in cols_show}
            print("  -", meta)

    return 0

def main():
    ap = argparse.ArgumentParser(description="Validate RangeNet Postflop (FLOP-only) manifest parquet")
    ap.add_argument("--manifest", type=Path, required=True, help="path to rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--strict", action="store_true", help="fail on any issue")
    ap.add_argument("--sample", type=int, default=5, help="print N random rows")
    args = ap.parse_args()

    sys.exit(validate_manifest(args.manifest, strict=args.strict, sample=args.sample))

if __name__ == "__main__":
    main()