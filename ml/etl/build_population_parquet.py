from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client

GRP = ["stakes_id", "street_id", "ctx_id", "hero_pos_id", "villain_pos_id"]

# Action IDs (must match your parser/enums)
FOLD = 0
CALL = 1
RAISE = 2
ALL_IN = 5  # will be normalized to RAISE


def _read_ndjson_auto(path: str | Path) -> pl.DataFrame:
    """
    Read .jsonl or .jsonl.gz into a Polars DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"decisions path not found: {p}")
    # polars can read both .jsonl and .jsonl.gz via read_ndjson
    return pl.read_ndjson(str(p))


def _rename_counts(pivot: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure we have n_fold, n_call, n_raise columns after pivot.
    Missing columns are filled with 0.
    """
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


def build_population_parquet(
    decisions_in: str | Path,
    out_parquet: str | Path,
    *,
    weight_mode: str = "count",  # "count" | "sqrt" | "log1p"
) -> Path:
    df = _read_ndjson_auto(decisions_in)

    # Normalize ALL_IN to RAISE
    df = df.with_columns(
        pl.when(pl.col("act_id") == ALL_IN).then(RAISE).otherwise(pl.col("act_id")).alias("act_id")
    )

    # Counts per (cell, action)
    agg = (
        df.group_by(GRP + ["act_id"])
          .count()
          .rename({"count": "n"})
    )

    # Pivot action counts into columns
    pivot = (
        agg.pivot(values="n", index=GRP, columns="act_id")
           .fill_null(0)
    )
    pivot = _rename_counts(pivot)

    # Total rows in cell
    pivot = pivot.with_columns(
        (pl.col("n_fold") + pl.col("n_call") + pl.col("n_raise")).alias("n_rows")
    )

    # Probabilities (soft labels)
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

    # Weight
    if weight_mode == "count":
        w_expr = pl.col("n_rows").cast(pl.Float64)
    elif weight_mode == "sqrt":
        w_expr = pl.col("n_rows").cast(pl.Float64).sqrt()
    elif weight_mode == "log1p":
        w_expr = (pl.col("n_rows").cast(pl.Float64) + 1.0).log()
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    pivot = pivot.with_columns(w_expr.alias("w"))

    # Final training schema
    out_df = pivot.select(GRP + ["p_fold", "p_call", "p_raise", "y", "w", "n_rows"])

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(str(out_path))
    print(f"✅ wrote {out_path} with {out_df.height} cells")
    return out_path


# ---------------- CLI wrapper (config-friendly) ----------------

def _cfg_get(cfg: dict, path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def run_from_config(cfg: dict, overrides: Optional[dict] = None) -> Path:
    """
    Expected cfg shape (example):
      populationnet:
        build:
          decisions_in: data/processed/nl10/decisions.jsonl.gz
          out_parquet: data/datasets/populationnet_nl10.parquet
          weight_mode: count  # or sqrt/log1p
    """
    sect = _cfg_get(cfg, "build", {})
    decisions_in = (overrides or {}).get("decisions_in", sect.get("decisions_in"))
    out_parquet  = (overrides or {}).get("out_parquet",  sect.get("out_parquet"))
    weight_mode  = (overrides or {}).get("weight_mode",  sect.get("weight_mode", "count"))

    if not decisions_in:
        raise ValueError("populationnet.build.decisions_in is required")
    if not out_parquet:
        raise ValueError("populationnet.build.out_parquet is required")

    return build_population_parquet(
        decisions_in=decisions_in,
        out_parquet=out_parquet,
        weight_mode=weight_mode,
    )


def main():
    import argparse
    from pathlib import Path

    from ml.utils.config import load_model_config


    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="populationnet",
                    help="model name or path to YAML")
    ap.add_argument("--stake", type=int, default=10, help="e.g. 10 for NL10")
    ap.add_argument("--s3-prefix", type=str, default="parsed",
                    help="S3 prefix under the bucket, e.g. 'parsed'")
    ap.add_argument("--bucket", type=str, default=None,
                    help="Override S3 bucket name if not in S3Client config")
    ap.add_argument("--local-cache", type=Path, default=Path("data/processed"),
                    help="Local cache dir for downloads (will create nl<stake>/)")
    ap.add_argument("--decisions", type=str, default=None,
                    help="override decisions_in path (.jsonl or .jsonl.gz); if omitted we download from S3")
    ap.add_argument("--out", type=str, default=None,
                    help="override out_parquet path")
    ap.add_argument("--weight", type=str, default=None,
                    help="override weight_mode: count|sqrt|log1p")
    args = ap.parse_args()

    # Load YAML config
    cfg = load_model_config(args.config)

    # Resolve decisions input: either user-provided path OR download from S3
    decisions_path: Path
    if args.decisions:
        decisions_path = Path(args.decisions)
    else:
        s3 = S3Client()
        stake_str = f"nl{args.stake}"
        local_dir = (args.local_cache / stake_str)
        local_dir.mkdir(parents=True, exist_ok=True)

        # S3 keys (match your validator layout)
        decisions_key = f"{args.s3_prefix}/{stake_str}/decisions.jsonl.gz"
        hands_key     = f"{args.s3_prefix}/{stake_str}/hands.jsonl.gz"

        # Local cache targets
        decisions_path = local_dir / f"decisions_{stake_str}.jsonl.gz"
        hands_path     = local_dir / f"hands_{stake_str}.jsonl.gz"

        # Best-effort downloader accommodating both .download and .download_file APIs
        def _dl(key: str, dest: Path, label: str):
            print(f"⬇️  downloading {label}: s3://{s3.bucket}/{key} → {dest}")
            if hasattr(s3, "download"):
                s3.download(key, dest)
            elif hasattr(s3, "download_file"):
                s3.download_file(key, dest)
            else:
                raise RuntimeError("S3Client has no download method (expected .download or .download_file)")
            print(f"✅ Downloaded: s3://{s3.bucket}/{key} → {dest}")

        # Download both (hands is not strictly needed for the builder, but useful for audits)
        _dl(decisions_key, decisions_path, "decisions")
        _dl(hands_key, hands_path, "hands")

    # Build overrides
    overrides = {
        "stake": args.stake,                 # let run_from_config use this stake
        "decisions_in": str(decisions_path)  # point builder at the cached file
    }
    if args.out:
        overrides["out_parquet"] = args.out
    if args.weight:
        overrides["weight_mode"] = args.weight

    # Kick off the build

    run_from_config(cfg, overrides=overrides)


if __name__ == "__main__":
    main()