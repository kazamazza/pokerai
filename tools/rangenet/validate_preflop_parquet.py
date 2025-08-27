#!/usr/bin/env python3
import argparse
from pathlib import Path
import polars as pl

# Adjust if your project has a canonical list somewhere
POS_NAMES = {"UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB", "EP", "MP", "BU"}
ALLOWED_OPENER_ACTIONS = {"OPEN", "RAISE", "ALL_IN", "LIMP", "CALL"}  # be permissive


def find_y_cols(df: pl.DataFrame) -> list[str]:
    ys = [c for c in df.columns if c.startswith("y") or c.startswith("Y")]
    # sort numerically (y0..y168)
    def key(c: str):
        n = c[1:] if c[0] in ("y", "Y") else c
        try:
            return int(n)
        except:
            return 1_000_000
    ys.sort(key=key)
    return ys

def validate_parquet(path: Path, strict: bool = False, sample: int | None = None) -> int:
    import polars as pl
    import numpy as np
    import random

    df = pl.read_parquet(str(path))
    print(f"Loaded: {path}  shape={df.shape}")

    # Fallbacks if globals aren't imported in this module
    pos_names = globals().get("POS_NAMES", {"UTG","LJ","HJ","CO","BTN","SB","BB","EP","MP","BU"})
    allowed_actions = globals().get(
        "ALLOWED_OPENER_ACTIONS",
        {"OPEN","RAISE","ALL_IN","LIMP","CALL", None}
    )

    # Required feature columns (adapt if yours differ)
    required = ["stack_bb", "hero_pos", "opener_pos", "opener_action"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return 1

    # Normalize opener_action synonyms (safe no-op if not present)
    norm_map = {
        "Open": "OPEN",
        "Raise": "RAISE",
        "Min": "RAISE",
        "AI": "ALL_IN",
        "Allin": "ALL_IN",
        "Limp": "LIMP",
        "Call": "CALL",
    }
    if "opener_action" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("opener_action").is_not_null())
              .then(pl.col("opener_action").replace(norm_map))
              .otherwise(pl.lit(None))
              .alias("opener_action")
        )

    # Find y_0 … y_168 columns
    def find_y_cols(d: pl.DataFrame) -> list[str]:
        ys = [c for c in d.columns if c.startswith("y_")]
        # sort by index
        try:
            ys = sorted(ys, key=lambda c: int(c.split("_", 1)[1]))
        except Exception:
            pass
        return ys

    y_cols = find_y_cols(df)
    if len(y_cols) != 169:
        preview = ", ".join(y_cols[:5])
        print(f"❌ Expected 169 y-columns, found {len(y_cols)} ({preview} …)")
        return 1

    # Basic categorical sanity
    bad_pos = df.filter(~pl.col("hero_pos").is_in(list(pos_names))).height
    bad_op  = df.filter(~pl.col("opener_pos").is_in(list(pos_names))).height
    bad_act = df.filter(~pl.col("opener_action").is_in(list(allowed_actions))).height
    if bad_pos or bad_op or bad_act:
        print(f"⚠️ Categorical issues: hero_pos bad={bad_pos} opener_pos bad={bad_op} opener_action bad={bad_act}")

    # Null / NaN checks across all y columns
    has_null = df.select(pl.any_horizontal([pl.col(c).is_null() for c in y_cols]).any()).item()
    has_nan  = df.select(pl.any_horizontal([pl.col(c).is_nan()  for c in y_cols]).any()).item()
    if has_null or has_nan:
        print(f"❌ Found null/NaN in probability columns (null={bool(has_null)} nan={bool(has_nan)})")
        return 1

    # Range check [0,1]
    too_low  = df.select(pl.any_horizontal([pl.col(c) < 0 for c in y_cols]).any()).item()
    too_high = df.select(pl.any_horizontal([pl.col(c) > 1 for c in y_cols]).any()).item()
    if too_low or too_high:
        print(f"❌ Probabilities out of [0,1] bounds: <0={bool(too_low)} >1={bool(too_high)}")
        return 1

    # Sum-to-1 check (within tolerance)
    tol = 1e-3 if strict else 5e-3
    sums = df.select(pl.sum_horizontal([pl.col(c) for c in y_cols]).alias("sum")).get_column("sum")
    bad_sum = int(((sums < (1 - tol)) | (sums > (1 + tol))).sum())
    if bad_sum > 0:
        print(f"⚠️ {bad_sum} rows have probs-sum outside 1±{tol:.3g}")
        if strict:
            return 1

    # Optional: sample a few rows and print top-5 combos by prob (indices only)
    if sample:
        idxs = random.sample(range(df.height), min(sample, df.height))
        print("\nSamples:")
        for i in idxs:
            row = df[i]
            vec = np.array([row[c] for c in y_cols], dtype=float)
            top = vec.argsort()[-5:][::-1]
            meta = {k: row[k] for k in required}
            print(f"— row {i} {meta}")
            print("  top-5 idx: ", " ".join(f"{j}:{vec[j]:.3f}" for j in top))
            print(f"  sum={vec.sum():.6f}")

    # Summary
    print("✅ Validation complete.")
    print(f"rows={df.height}  y_cols=169  bad_sum={bad_sum}  strict={strict}")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Validate RangeNet Preflop parquet")
    ap.add_argument("--parquet", type=Path, required=True, help="path to rangenet_preflop parquet")
    ap.add_argument("--strict", action="store_true", help="fail if sum-to-1 is outside a tight tolerance")
    ap.add_argument("--sample", type=int, default=0, help="print n sampled rows with top-5 probs")
    args = ap.parse_args()
    raise SystemExit(validate_parquet(args.parquet, strict=args.strict, sample=args.sample or None))

if __name__ == "__main__":
    main()