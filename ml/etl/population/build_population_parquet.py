from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config
from ml.etl.population.derive_features import build_action_features_from_hands  # <- helper we wrote

GRP = [
    "stakes_id", "street_id", "ctx_id",
    "hero_pos_id", "villain_pos_id",
    "facing_id", "spr_bin", "bet_size_bucket",
    "board_cluster_id",
]

FOLD, CALL, RAISE, ALL_IN = 0, 1, 2, 5


def _read_ndjson_auto(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"path not found: {p}")
    return pl.read_ndjson(str(p))


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


def _pct_to_bucket_expr(expr: pl.Expr) -> pl.Expr:
    # bins: <25, <33, <50, <66, <75, <100, <150, <200, >=200; -1 for not facing
    return (
        pl.when(expr < 0.25).then(0)
          .when(expr < 0.33).then(1)
          .when(expr < 0.50).then(2)
          .when(expr < 0.66).then(3)
          .when(expr < 0.75).then(4)
          .when(expr < 1.00).then(5)
          .when(expr < 1.50).then(6)
          .when(expr < 2.00).then(7)
          .otherwise(8)
          .cast(pl.Int8)
    )


def _add_seq_in_street(dec: pl.DataFrame) -> pl.DataFrame:
    # deterministic within (hand, street, hero_pos)
    return (dec
            .sort(["hand_id", "street_id", "hero_pos_id"])
            .with_columns(
                pl.col("hand_id").cum_count().over(["hand_id", "street_id", "hero_pos_id"]).alias("seq_in_street")
            ))


def build_population_parquet(
    decisions_in: str | Path,
    hands_in: str | Path,
    out_parquet: str | Path,
    *,
    weight_mode: str = "count",  # "count" | "sqrt" | "log1p"
    clusterer_cfg: Optional[dict] = None,  # pass cfg["board_clustering"] block or whole cfg
) -> Path:
    # -- load inputs
    dec = _read_ndjson_auto(decisions_in)
    dec = _add_seq_in_street(dec)
    # normalize ALL_IN → RAISE
    dec = dec.with_columns(
        pl.when(pl.col("act_id") == ALL_IN).then(RAISE).otherwise(pl.col("act_id")).alias("act_id")
    )

    # -- build features from hands (pot/to_call/facing/spr/board_cluster)
    clusterer = None
    if clusterer_cfg is not None:
        try:
            # accepts full cfg; helper will read cfg["board_clustering"]
            clusterer = load_board_clusterer(clusterer_cfg)  # safe: rule or kmeans
        except Exception:
            clusterer = None  # fall back to 0 cluster id

    feat = build_action_features_from_hands(hands_in, clusterer=clusterer)

    # -- join (hand, street, pos, seq)
    dec = dec.join(
        feat,
        how="left",
        left_on=["hand_id", "street_id", "hero_pos_id", "seq_in_street"],
        right_on=["hand_id", "street_id", "actor_pos_id", "seq_in_street"],
    )

    # -- fill safe defaults if join misses
    dec = dec.with_columns([
        pl.col("facing_id").fill_null(0),
        pl.col("spr_bin").fill_null(3),
        pl.col("to_call_bb").fill_null(0.0),
        pl.col("pot_before_bb").fill_null(0.0),
        pl.col("board_cluster_id").fill_null(0),
    ])

    # -- bet_pct_of_pot & bucket (only when facing)
    bet_pct = (pl.col("to_call_bb") / pl.col("pot_before_bb").clip(lower_bound=1e-6)).clip(0.0, 50.0)
    dec = dec.with_columns(
        pl.when(pl.col("facing_id") == 1)
          .then(bet_pct)
          .otherwise(pl.lit(-1.0))
          .alias("bet_pct_of_pot")
    )
    dec = dec.with_columns(
        pl.when(pl.col("facing_id") == 1)
          .then(_pct_to_bucket_expr(pl.col("bet_pct_of_pot")))
          .otherwise(pl.lit(-1))
          .alias("bet_size_bucket")
    )

    # -- aggregate to soft labels
    agg = (
        dec.group_by(GRP + ["act_id"])
           .count()
           .rename({"count": "n"})
    )
    pivot = agg.pivot(values="n", index=GRP, columns="act_id").fill_null(0)
    pivot = _rename_counts(pivot)
    pivot = pivot.with_columns(
        (pl.col("n_fold") + pl.col("n_call") + pl.col("n_raise")).alias("n_rows")
    )

    denom = pl.col("n_rows").cast(pl.Float64).clip(lower_bound=1.0)
    pivot = pivot.with_columns([
        (pl.col("n_fold").cast(pl.Float64) / denom).alias("p_fold"),
        (pl.col("n_call").cast(pl.Float64) / denom).alias("p_call"),
        (pl.col("n_raise").cast(pl.Float64) / denom).alias("p_raise"),
    ])
    pivot = pivot.with_columns(
        pl.concat_list([pl.col("p_fold"), pl.col("p_call"), pl.col("p_raise")]).list.arg_max().alias("y")
    )

    if weight_mode == "count":
        w_expr = pl.col("n_rows").cast(pl.Float64)
    elif weight_mode == "sqrt":
        w_expr = pl.col("n_rows").cast(pl.Float64).sqrt()
    elif weight_mode == "log1p":
        w_expr = (pl.col("n_rows").cast(pl.Float64) + 1.0).log()
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    pivot = pivot.with_columns(w_expr.alias("w"))

    out_df = pivot.select(GRP + ["p_fold", "p_call", "p_raise", "y", "w", "n_rows"])

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(str(out_path))
    print(f"✅ wrote {out_path} with {out_df.height} cells")
    return out_path


# ---------------- config wrapper ----------------

def _cfg_get(cfg: dict, path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def run_from_config(cfg: dict, overrides: Optional[dict] = None) -> Path:
    """
    cfg:
      build:
        decisions_in: data/processed/nl10/decisions.jsonl.gz
        hands_in:     data/processed/nl10/hands.jsonl.gz
        out_parquet:  data/datasets/populationnet_nl10.parquet
        weight_mode:  count|sqrt|log1p
      board_clustering:
        type: rule|kmeans
        n_clusters: 128
        artifact: ...
    """
    sect = _cfg_get(cfg, "build", {})
    decisions_in = (overrides or {}).get("decisions_in", sect.get("decisions_in"))
    hands_in     = (overrides or {}).get("hands_in",     sect.get("hands_in"))
    out_parquet  = (overrides or {}).get("out_parquet",  sect.get("out_parquet"))
    weight_mode  = (overrides or {}).get("weight_mode",  sect.get("weight_mode", "count"))

    if not decisions_in:
        raise ValueError("populationnet.build.decisions_in is required")
    if not hands_in:
        raise ValueError("populationnet.build.hands_in is required")
    if not out_parquet:
        raise ValueError("populationnet.build.out_parquet is required")

    # pass full cfg so load_board_clusterer can read board_clustering section
    return build_population_parquet(
        decisions_in=decisions_in,
        hands_in=hands_in,
        out_parquet=out_parquet,
        weight_mode=weight_mode,
        clusterer_cfg=cfg,
    )


# ---------------- CLI ----------------

def main():
    import argparse
    from infra.storage.s3_client import S3Client

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="populationnet", help="model name or YAML path")
    ap.add_argument("--stake", type=int, default=10, help="e.g. 10 for NL10")
    ap.add_argument("--s3-prefix", type=str, default="parsed", help="S3 prefix (e.g. parsed)")
    ap.add_argument("--bucket", type=str, default=None, help="override S3 bucket if needed")
    ap.add_argument("--local-cache", type=Path, default=Path("data/processed"), help="cache dir")
    ap.add_argument("--decisions", type=str, default=None, help="override local decisions path")
    ap.add_argument("--hands", type=str, default=None, help="override local hands path")
    ap.add_argument("--out", type=str, default=None, help="override out parquet")
    ap.add_argument("--weight", type=str, default=None, help="count|sqrt|log1p")
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    if args.decisions and args.hands:
        decisions_path = Path(args.decisions)
        hands_path = Path(args.hands)
    else:
        s3 = S3Client()
        stake_str = f"nl{args.stake}"
        local_dir = (args.local_cache / stake_str)
        local_dir.mkdir(parents=True, exist_ok=True)

        decisions_key = f"{args.s3_prefix}/{stake_str}/decisions.jsonl.gz"
        hands_key     = f"{args.s3_prefix}/{stake_str}/hands.jsonl.gz"
        decisions_path = local_dir / f"decisions_{stake_str}.jsonl.gz"
        hands_path     = local_dir / f"hands_{stake_str}.jsonl.gz"

        def _dl(key: str, dest: Path, label: str):
            print(f"⬇️  downloading {label}: s3://{s3.bucket}/{key} → {dest}")
            if hasattr(s3, "download"):
                s3.download(key, dest)
            elif hasattr(s3, "download_file"):
                s3.download_file(key, dest)
            else:
                raise RuntimeError("S3Client needs .download or .download_file")
            print(f"✅ Downloaded: s3://{s3.bucket}/{key} → {dest}")

        _dl(decisions_key, decisions_path, "decisions")
        _dl(hands_key, hands_path, "hands")

    overrides = {
        "decisions_in": str(decisions_path),
        "hands_in": str(hands_path),
    }
    if args.out:
        overrides["out_parquet"] = args.out
    if args.weight:
        overrides["weight_mode"] = args.weight

    run_from_config(cfg, overrides=overrides)


if __name__ == "__main__":
    main()