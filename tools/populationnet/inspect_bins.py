#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import polars as pl

FOLD, CALL, RAISE, ALL_IN = 0, 1, 2, 5

def read_decisions(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pl.read_ndjson(str(p))  # works for .jsonl and .jsonl.gz

def safe_clip(expr: pl.Expr, lo: float, hi: float) -> pl.Expr:
    return pl.when(expr < lo).then(lo).when(expr > hi).then(hi).otherwise(expr)

def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    # normalize ALL_IN → RAISE
    df = df.with_columns(
        pl.when(pl.col("act_id") == ALL_IN).then(RAISE).otherwise(pl.col("act_id")).alias("act_id")
    )

    # facing_id: fallback to raises_before_hero if to_call_bb missing
    if "to_call_bb" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("to_call_bb").fill_null(0.0) > 0.0).then(1).otherwise(0).alias("facing_id")
        )
    elif "raises_before_hero" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("raises_before_hero").fill_null(0) > 0).then(1).otherwise(0).alias("facing_id")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("facing_id"))

    # No pot info available → skip bet_pct_of_pot (set sentinel)
    df = df.with_columns(pl.lit(-1.0).alias("bet_pct_of_pot"))

    # SPR (fallback if eff_stack_bb missing)
    if "eff_stack_bb" in df.columns:
        spr = (pl.col("eff_stack_bb") / pl.col("amount_bb").clip(lower_bound=1e-6)).clip(lower_bound=0.0, upper_bound=100.0)
    else:
        spr = pl.lit(10.0)
    df = df.with_columns(spr.alias("spr"))
    return df

def quantiles(df: pl.DataFrame, col: str, mask: pl.Expr | None = None):
    qs = [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]

    data = df.filter(mask) if mask is not None else df
    if col not in data.columns or data.height == 0:
        return None

    s = (
        data.get_column(col)
            .cast(pl.Float64, strict=False)
            .drop_nulls()
    )
    if s.len() == 0:
        return None

    vals = [float(s.quantile(q, interpolation="nearest")) for q in qs]
    return list(zip([int(q * 100) for q in qs], vals))

def make_spr_bin(spr_bins: list[float]) -> pl.Expr:
    # bins like [0,2,4,7,100] → 0..len-2
    edges = list(spr_bins)
    def idx_expr(x):
        # assign the first edge where x < edge; else last bin
        expr = pl.lit(len(edges)-2)
        for i in range(1, len(edges)):
            expr = pl.when(x < edges[i]).then(i-1).otherwise(expr)
        return expr.cast(pl.Int64)
    return idx_expr(pl.col("spr"))

def make_bet_bucket(thresholds: list[float]) -> pl.Expr:
    # thresholds like [0.33,0.5,0.66,1.0,1.5,2.5] → bucket 0..len
    t = sorted(thresholds)
    x = pl.col("bet_pct_of_pot")
    expr = pl.lit(len(t))  # above last threshold
    for i, th in enumerate(t):
        expr = pl.when(x < th).then(i).otherwise(expr)
    # keep sentinel -1 for non-facing
    expr = pl.when(x < 0).then(pl.lit(-1)).otherwise(expr)
    return expr.cast(pl.Int64)

def try_grouping(df: pl.DataFrame, grp_cols: list[str]):
    agg = df.group_by(grp_cols + ["act_id"]).count().rename({"count":"n"})
    pivot = agg.pivot(values="n", index=grp_cols, columns="act_id").fill_null(0)
    # ensure cols
    for need in ("0","1","2"):
        if need not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(need))
    pivot = pivot.rename({"0":"n_fold","1":"n_call","2":"n_raise"})
    pivot = pivot.with_columns((pl.col("n_fold")+pl.col("n_call")+pl.col("n_raise")).alias("n_rows"))
    # metrics
    cells = pivot.height
    median = pivot["n_rows"].median()
    p95 = pivot["n_rows"].quantile(0.95, interpolation="nearest")
    maxv = pivot["n_rows"].max()
    sparse = (pivot["n_rows"] < 20).sum() / cells if cells else 0.0
    return {"cells": cells, "median": median, "p95": p95, "max": maxv, "sparsity": float(sparse)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, help=".jsonl or .jsonl.gz")
    ap.add_argument("--spr_bins", default="0,2,4,7,100", help="comma-separated edges")
    ap.add_argument("--bet_thresholds", default="0.33,0.5,0.66,1.0,1.5,2.5", help="bet% thresholds")
    ap.add_argument("--include_facing", action="store_true")
    ap.add_argument("--include_vpos", action="store_true")
    args = ap.parse_args()

    df = read_decisions(args.decisions)
    df = add_derived(df)

    # print quantiles
    q_bet = quantiles(df, "bet_pct_of_pot", mask=(pl.col("facing_id")==1) & (pl.col("bet_pct_of_pot")>=0))
    q_spr = quantiles(df, "spr")
    print("bet_pct_of_pot quantiles (facing=1):", q_bet)
    print("spr quantiles:", q_spr)

    # parse bins
    spr_bins = [float(x) for x in args.spr_bins.split(",")]
    bet_thr = [float(x) for x in args.bet_thresholds.split(",")] if args.bet_thresholds else []

    # add bins
    df = df.with_columns(make_spr_bin(spr_bins).alias("spr_bin"))
    if bet_thr:
        df = df.with_columns(make_bet_bucket(bet_thr).alias("bet_size_bucket"))
    else:
        df = df.with_columns(pl.lit(-1).alias("bet_size_bucket"))

    # try a few groupings
    base = ["stakes_id","street_id","ctx_id","hero_pos_id","spr_bin","bet_size_bucket"]
    if args.include_facing:
        base = base + ["facing_id"]
    if args.include_vpos:
        base = base + ["villain_pos_id"]

    stats = try_grouping(df, base)
    print(f"\nGrouping: {base}")
    print(f"cells={stats['cells']:,}  median={int(stats['median'])}  p95={int(stats['p95'])}  "
          f"max={int(stats['max'])}  sparsity={stats['sparsity']:.3f}")

    # also show effect of removing bet bucket/spr quickly
    alt1 = [c for c in base if c != "bet_size_bucket"]
    alt2 = [c for c in base if c != "spr_bin"]
    for name, cols in [("no_bet_bucket", alt1), ("no_spr_bin", alt2)]:
        s = try_grouping(df, cols)
        print(f"{name:>14}: cells={s['cells']:,} median={int(s['median'])} p95={int(s['p95'])} "
              f"max={int(s['max'])} sparsity={s['sparsity']:.3f}")

if __name__ == "__main__":
    main()