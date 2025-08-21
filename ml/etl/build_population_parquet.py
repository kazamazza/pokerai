import argparse
import sys
from pathlib import Path
import json
import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.config.coverage.coverage_populationnet import resolve_input_path
from ml.core.types import Act

# Group-by keys for a “cell” (populationnet granularity)
GRP = ["stakes_id", "street_id", "ctx_id", "hero_pos_id", "villain_pos_id"]


def _rename_counts(pivot: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize pivoted action count columns to fixed names:
    n_fold, n_call, n_raise
    """
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

    # Ensure all three action columns exist
    for need in ("n_fold", "n_call", "n_raise"):
        if need not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(need))
    return pivot


def _filter_with_ok_cells(pivot: pl.DataFrame, coverage_path: Path | None) -> pl.DataFrame:
    """
    Restrict to ok_cells from coverage JSON if provided.
    Otherwise, return pivot unchanged.
    """
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

    # Probabilities
    denom = pl.col("n_rows").cast(pl.Float64).clip(lower_bound=1.0)

    pivot = pivot.with_columns([
        (pl.col("n_fold").cast(pl.Float64) / denom).alias("p_fold"),
        (pl.col("n_call").cast(pl.Float64) / denom).alias("p_call"),
        (pl.col("n_raise").cast(pl.Float64) / denom).alias("p_raise"),
    ])

    # Hard label y = argmax over probs
    pivot = pivot.with_columns(
        pl.concat_list([
            pl.col("p_fold"),
            pl.col("p_call"),
            pl.col("p_raise"),
        ]).list.arg_max().alias("y")
    )

    # Weight = number of observations (can log1p/sqrt later if desired)
    pivot = pivot.with_columns(pl.col("n_rows").cast(pl.Float64).alias("w"))

    # Final training schema
    out = pivot.select(GRP + ["y", "w", "p_fold", "p_call", "p_raise", "n_rows"])

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(str(out_path))
    print(f"✅ wrote {out_path} with {out.height} cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=int, required=True, help="e.g. 10 for NL10")
    ap.add_argument("--input", type=str, default=None, help="decisions path (.jsonl or .jsonl.gz, local or s3://)")
    ap.add_argument("--coverage", type=str, default=None, help="coverage JSON (to filter ok_cells)")
    ap.add_argument("--out", type=str, default=None, help="output parquet path")
    args = ap.parse_args()

    out = args.out or f"data/datasets/populationnet_nl{args.stake}.parquet"
    build_population_parquet(args.stake, args.input, args.coverage, out)