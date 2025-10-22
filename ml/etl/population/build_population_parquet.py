from __future__ import annotations
import sys
from pathlib import Path
from typing import Sequence, Optional
import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.utils.config import load_model_config

FOLD = 0
CALL = 1
RAISE = 2
ALL_IN = 5  # map to RAISE

def _rename_counts(pivot: pl.DataFrame) -> pl.DataFrame:
    colnames = set(pivot.columns)
    rename = {}
    if 0 in colnames:   rename[0]   = "n_fold"
    if "0" in colnames: rename["0"] = "n_fold"
    if 1 in colnames:   rename[1]   = "n_call"
    if "1" in colnames: rename["1"] = "n_call"
    if 2 in colnames:   rename[2]   = "n_raise"
    if "2" in colnames: rename["2"] = "n_raise"
    if rename:
        pivot = pivot.rename(rename)
    for need in ("n_fold", "n_call", "n_raise"):
        if need not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(need))
    return pivot

def build_population_from_enriched(
    parquet_in: str | Path,
    out_parquet: str | Path,
    *,
    x_cols: Sequence[str],
    weight_mode: str = "count",      # "count" | "sqrt" | "log1p"
    min_rows: Optional[int] = None,  # drop sparse cells (< min_rows)
) -> Path:
    df = pl.read_parquet(str(parquet_in))

    # normalize ALL_IN → RAISE
    df = df.with_columns(
        pl.when(pl.col("act_id") == ALL_IN).then(RAISE).otherwise(pl.col("act_id")).alias("act_id")
    )

    # group & pivot
    agg = df.group_by(list(x_cols) + ["act_id"]).len().rename({"len": "n"})
    pivot = agg.pivot(values="n", index=list(x_cols), on="act_id").fill_null(0)
    pivot = _rename_counts(pivot)

    # totals & soft labels
    pivot = pivot.with_columns((pl.col("n_fold") + pl.col("n_call") + pl.col("n_raise")).alias("n_rows"))
    if min_rows is not None:
        pivot = pivot.filter(pl.col("n_rows") >= int(min_rows))

    denom = pl.col("n_rows").cast(pl.Float64).clip(lower_bound=1.0)
    pivot = pivot.with_columns([
        (pl.col("n_fold").cast(pl.Float64) / denom).alias("p_fold"),
        (pl.col("n_call").cast(pl.Float64) / denom).alias("p_call"),
        (pl.col("n_raise").cast(pl.Float64) / denom).alias("p_raise"),
    ])

    # hard label (optional)
    pivot = pivot.with_columns(
        pl.concat_list([pl.col("p_fold"), pl.col("p_call"), pl.col("p_raise")]).list.arg_max().alias("y")
    )

    # weights
    if weight_mode == "count":
        w_expr = pl.col("n_rows").cast(pl.Float64)
    elif weight_mode == "sqrt":
        w_expr = pl.col("n_rows").cast(pl.Float64).sqrt()
    elif weight_mode == "log1p":
        w_expr = (pl.col("n_rows").cast(pl.Float64) + 1.0).log()
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    pivot = pivot.with_columns(w_expr.alias("w"))

    out_cols = list(x_cols) + ["p_fold", "p_call", "p_raise", "y", "w", "n_rows"]
    out_df = pivot.select(out_cols)

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(str(out_path))

    print(f"✅ wrote {out_path}  cells={out_df.height}")
    return out_path


# ---------- CLI wrapper (config-friendly) ----------
def _cfg_get(cfg: dict, path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def run_from_config(cfg: dict, *, parquet: str | None = None) -> Path:
    """
    Build PopulationNet dataset. If `parquet` is provided, it overrides any config value.
    """
    parquet_in = parquet
    out_parquet = _cfg_get(cfg, "build.out_parquet", "data/datasets/populationnet.parquet")
    x_cols = _cfg_get(cfg, "dataset.x_cols")
    if not parquet_in or not x_cols:
        raise ValueError("Need parquet (CLI or config) and dataset.x_cols in config.")
    weight_mode = _cfg_get(cfg, "build.weight_mode", "count")
    min_rows = _cfg_get(cfg, "build.min_rows", None)

    return build_population_from_enriched(
        parquet_in=parquet_in,
        out_parquet=out_parquet,
        x_cols=x_cols,
        weight_mode=weight_mode,
        min_rows=min_rows,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="populationnet",
                    help="Model name or YAML path")
    ap.add_argument("--parquet", type=str, required=True,
                    help="Path to enriched parquet for PopulationNet")
    args = ap.parse_args()
    cfg = load_model_config(args.config)
    run_from_config(cfg, parquet=args.parquet)