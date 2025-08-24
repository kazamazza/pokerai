import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.coverage.coverage_populationnet import resolve_input_path, add_freqs_with_entropy
from ml.core.types import Act
import json
from pathlib import Path
import polars as pl

# Group-by keys for a “cell” (populationnet granularity)
GRP = ["stakes_id", "street_id", "ctx_id", "hero_pos_id", "villain_pos_id"]

def _rename_counts(pivot: pl.DataFrame) -> pl.DataFrame:
    colnames = set(pivot.columns)
    rename_map = {}
    if 0 in colnames:   rename_map[0]   = "n_fold"
    if "0" in colnames: rename_map["0"] = "n_fold"
    if 1 in colnames:   rename_map[1]   = "n_call"
    if "1" in colnames: rename_map["1"] = "n_call"
    if 2 in colnames:   rename_map[2]   = "n_raise"
    if "2" in colnames: rename_map["2"] = "n_raise"
    if rename_map:
        pivot = pivot.rename(rename_map)
    for need in ("n_fold", "n_call", "n_raise"):
        if need not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(need))
    return pivot

def _filter_with_ok_cells(pivot: pl.DataFrame, coverage_path: Path | None) -> pl.DataFrame:
    if not coverage_path or not coverage_path.exists():
        return pivot
    cov = json.loads(coverage_path.read_text())
    ok = cov["populationnet"]["ok_cells"]
    ok_df = pl.DataFrame(ok)  # keys match GRP
    return pivot.join(ok_df, on=GRP, how="inner")

def build_population_parquet(
    stake: int,
    decisions_in: str | None,
    coverage_json: str | None,
    out_parquet: str,
):
    """
    Build a populationnet training parquet from parsed decisions.
    Input: decisions .jsonl (possibly gzipped, local or S3).
    Output: Parquet with aggregated cells and soft labels.
    """
    s3 = S3Client()
    dec_path = resolve_input_path(stake, decisions_in, s3=s3)  # local unzipped .jsonl
    df = pl.read_ndjson(str(dec_path))

    # Normalize ALL_IN -> RAISE
    df = df.with_columns(
        pl.when(pl.col("act_id") == Act.ALL_IN.value)
          .then(Act.RAISE.value)
          .otherwise(pl.col("act_id"))
          .alias("act_id")
    )

    # Counts per (cell, action)
    agg = (
        df.group_by(GRP + ["act_id"])
          .count()
          .rename({"count": "n"})
    )
    pivot = (
        agg.pivot(values="n", index=GRP, columns="act_id")
           .fill_null(0)
    )
    pivot = _rename_counts(pivot)

    # Total rows
    pivot = pivot.with_columns(
        (pl.col("n_fold") + pl.col("n_call") + pl.col("n_raise")).alias("n_rows")
    )

    # Coverage filter (optional)
    pivot = _filter_with_ok_cells(pivot, Path(coverage_json) if coverage_json else None)

    # Probabilities (protect denom with clip lower-bound)
    denom = pl.col("n_rows").cast(pl.Float64).clip(lower_bound=1.0)
    pivot = pivot.with_columns([
        (pl.col("n_fold").cast(pl.Float64) / denom).alias("p_fold"),
        (pl.col("n_call").cast(pl.Float64) / denom).alias("p_call"),
        (pl.col("n_raise").cast(pl.Float64) / denom).alias("p_raise"),
    ])

    # Hard label y = argmax over probs
    pivot = pivot.with_columns(
        pl.concat_list([pl.col("p_fold"), pl.col("p_call"), pl.col("p_raise")]).list.arg_max().alias("y")
    )

    # Weight = number of observations (can log1p/sqrt later)
    pivot = pivot.with_columns(pl.col("n_rows").cast(pl.Float64).alias("w"))

    # Final training schema
    out = pivot.select(GRP + ["y", "w", "p_fold", "p_call", "p_raise", "n_rows"])

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(str(out_path))
    print(f"✅ wrote {out_path} with {out.height} cells")

def run_from_config(cfg: dict, overrides: dict | None = None) -> None:
    """
    Use config (with optional CLI overrides) to drive the builder.
    Expected in cfg:
      build:
        stake: 10
        decisions_in: s3://... or local path (jsonl/jsonl.gz)
        coverage_json: path or null
        out_parquet: data/datasets/populationnet_nl10.parquet
    """
    def get(path: str, default=None):
        cur = cfg
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    stake = get("build.stake", 10)
    decisions_in = get("build.decisions_in", None)
    coverage_json = get("build.coverage_json", None)
    out_parquet = get("build.out_parquet", f"data/datasets/populationnet_nl{stake}.parquet")

    # Apply CLI overrides if provided
    if overrides:
        if "stake" in overrides and overrides["stake"] is not None:
            stake = int(overrides["stake"])
        if "decisions_in" in overrides and overrides["decisions_in"]:
            decisions_in = overrides["decisions_in"]
        if "coverage_json" in overrides and overrides["coverage_json"]:
            coverage_json = overrides["coverage_json"]
        if "out_parquet" in overrides and overrides["out_parquet"]:
            out_parquet = overrides["out_parquet"]

    build_population_parquet(
        stake=stake,
        decisions_in=decisions_in,
        coverage_json=coverage_json,
        out_parquet=out_parquet,
    )

def main():
    import argparse
    import polars as pl
    from pathlib import Path
    import json

    from ml.utils.config import load_model_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="populationnet",
                    help="model name or path to YAML (resolved by load_model_config)")
    # lightweight overrides
    ap.add_argument("--stake", type=int, default=None)
    ap.add_argument("--input", type=str, default=None,
                    help="override coverage.input_decisions (path or s3://… to decisions .jsonl(.gz))")
    ap.add_argument("--out", type=str, default=None,
                    help="override coverage.output JSON path")
    ap.add_argument("--min_rows_per_cell", type=int, default=None)
    ap.add_argument("--min_cells_per_ctx", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    def get(path, default=None):
        cur = cfg
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    # defaults from YAML with CLI overrides
    stake = args.stake if args.stake is not None else int(get("build.stake", 10))
    input_decisions = args.input if args.input is not None else get("coverage.input_decisions", None)

    out_path = Path(args.out if args.out is not None else
                    get("coverage.output", f"ml/config/coverage/populationnet_nl{stake}.json"))

    min_rows_per_cell = (args.min_rows_per_cell
                         if args.min_rows_per_cell is not None
                         else int(get("coverage.min_rows_per_cell", 200)))
    min_cells_per_ctx = (args.min_cells_per_ctx
                         if args.min_cells_per_ctx is not None
                         else int(get("coverage.min_cells_per_ctx", 50)))
    alpha = (args.alpha
             if args.alpha is not None
             else float(get("coverage.alpha", 1.0)))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = S3Client()
    decisions_path = resolve_input_path(stake, input_decisions, s3=s3)  # local unzipped .jsonl
    df = pl.read_ndjson(str(decisions_path))

    # === the rest of your existing logic stays the same ===

    df = df.with_columns(
        pl.when(pl.col("act_id") == 5).then(2).otherwise(pl.col("act_id")).alias("act_id")
    )

    grp_keys = ["stakes_id", "street_id", "ctx_id", "hero_pos_id", "villain_pos_id"]

    agg = (
        df.group_by(grp_keys + ["act_id"])
          .count()
          .rename({"count": "n"})
    )

    pivot = agg.pivot(values="n", index=grp_keys, columns="act_id").fill_null(0)

    colnames = set(pivot.columns)
    rename_map = {}
    if 0 in colnames:   rename_map[0] = "n_fold"
    if "0" in colnames: rename_map["0"] = "n_fold"
    if 1 in colnames:   rename_map[1] = "n_call"
    if "1" in colnames: rename_map["1"] = "n_call"
    if 2 in colnames:   rename_map[2] = "n_raise"
    if "2" in colnames: rename_map["2"] = "n_raise"
    if rename_map:
        pivot = pivot.rename(rename_map)
    for need in ("n_fold", "n_call", "n_raise"):
        if need not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(need))

    pivot = pivot.with_columns([
        (pl.col("n_fold") + pl.col("n_call") + pl.col("n_raise")).alias("n_rows"),
    ]).with_columns([
        (pl.col("n_rows") >= min_rows_per_cell).alias("ok")
    ])

    ok_cells_tbl = add_freqs_with_entropy(pivot.filter(pl.col("ok")), alpha=alpha)

    ctx_cover = (
        pivot.group_by(["stakes_id", "street_id", "ctx_id"])
             .agg([
                 pl.len().alias("cells_total"),
                 pl.sum("ok").alias("cells_ok"),
                 pl.col("n_rows").sum().alias("rows_total"),
             ])
             .with_columns((pl.col("cells_ok") >= min_cells_per_ctx).alias("ctx_ok"))
    )

    CTX_NAME = {
        0: "OPEN", 1: "VS_OPEN", 2: "VS_3BET", 3: "VS_4BET",
        4: "BLIND_VS_STEAL", 5: "LIMPED_SINGLE", 6: "LIMPED_MULTI",
        10: "VS_CBET", 11: "VS_CBET_TURN", 13: "VS_CHECK_RAISE", 14: "VS_DONK",
    }
    STREET_NAME = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}

    include = []
    for r in ctx_cover.iter_rows(named=True):
        if r["ctx_ok"]:
            include.append({
                "stakes_id": int(r["stakes_id"]),
                "street": STREET_NAME.get(int(r["street_id"]), str(int(r["street_id"]))),
                "ctx": CTX_NAME.get(int(r["ctx_id"]), str(int(r["ctx_id"]))),
                "cells_ok": int(r["cells_ok"]),
                "cells_total": int(r["cells_total"]),
                "rows_total": int(r["rows_total"]),
                "min_rows_per_cell": min_rows_per_cell,
            })

    ok_cells = ok_cells_tbl.select(grp_keys).unique().to_dicts()
    for r in ok_cells:
        for k in ("stakes_id", "street_id", "ctx_id", "hero_pos_id", "villain_pos_id"):
            r[k] = int(r[k])

    freq_cells = (
        ok_cells_tbl
        .select(grp_keys + ["n_rows", "p_fold", "p_call", "p_raise", "entropy"])
        .to_dicts()
    )
    for r in freq_cells:
        r["stakes_id"] = int(r["stakes_id"])
        r["street_id"] = int(r["street_id"])
        r["ctx_id"] = int(r["ctx_id"])
        r["hero_pos_id"] = int(r["hero_pos_id"])
        r["villain_pos_id"] = int(r["villain_pos_id"])

    cfg_out = {
        "populationnet": {
            "generated_from": str(decisions_path),
            "thresholds": {
                "min_rows_per_cell": min_rows_per_cell,
                "min_cells_per_ctx": min_cells_per_ctx,
                "alpha": alpha,
            },
            "include_contexts": include,
            "ok_cells": ok_cells,
            "freq_cells": freq_cells,
            "ctx_summaries": (
                ok_cells_tbl
                .group_by(["stakes_id", "street_id", "ctx_id"])
                .agg([
                    pl.len().alias("cells_ok"),
                    pl.col("n_rows").sum().alias("rows_total"),
                    pl.col("p_fold").mean().alias("p_fold_mean"),
                    pl.col("p_call").mean().alias("p_call_mean"),
                    pl.col("p_raise").mean().alias("p_raise_mean"),
                    pl.col("entropy").mean().alias("entropy_mean"),
                    pl.col("entropy").median().alias("entropy_med"),
                ])
                .to_dicts()
            ),
            "actions": ["FOLD", "CALL", "RAISE"],
        }
    }

    # add friendly names to ctx_summaries
    for r in cfg_out["populationnet"]["ctx_summaries"]:
        r["stakes_id"] = int(r["stakes_id"])
        r["street_id"] = int(r["street_id"])
        r["ctx_id"] = int(r["ctx_id"])
        r["street"] = STREET_NAME.get(r["street_id"], str(r["street_id"]))
        r["ctx"] = CTX_NAME.get(r["ctx_id"], str(r["ctx_id"]))

    out_path.write_text(json.dumps(cfg_out, indent=2))
    print(f"✅ wrote coverage+freqs → {out_path}")

if __name__ == "__main__":
    main()