# scripts/coverage/populationnet_coverage.py
import argparse, json, os
import sys
from pathlib import Path
import polars as pl



ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
from utils.files import parse_s3_url, gunzip_file
from infra.storage.s3_client import S3Client

s3 = S3Client()

def resolve_input_path(stake: int, inp: str | None, s3: S3Client | None = None) -> Path:
    """
    Returns a LOCAL **unzipped** path to decisions for the given stake.
    Accepts:
      - s3://bucket/key.jsonl.gz  (downloads & gunzips)
      - local .jsonl.gz           (gunzips)
      - local .jsonl              (as is)
      - None → defaults to decisions_nl{stake}.jsonl.gz in CWD
    """
    # choose defaults
    default_gz = Path(f"decisions_nl{stake}.jsonl.gz")
    default_jsonl = Path(f"decisions_nl{stake}.jsonl")

    if inp:
        if inp.startswith("s3://"):
            if s3 is None:
                raise ValueError("S3Client required to fetch s3:// input")
            bucket, key = parse_s3_url(inp)
            # if caller's S3Client has a fixed bucket, require it matches (optional)
            if s3.bucket and bucket and s3.bucket != bucket:
                raise ValueError(f"S3 bucket mismatch: resolver={bucket} client={s3.bucket}")

            local_gz = Path("tmp") / Path(key).name  # e.g., tmp/decisions_nl10.jsonl.gz
            local_jsonl = local_gz.with_suffix("")    # remove .gz → .jsonl
            s3.download_file_if_missing(key, local_gz)
            if not local_jsonl.exists():
                gunzip_file(local_gz, local_jsonl)
                print(f"✅ Unzipped: {local_gz} → {local_jsonl}")
            return local_jsonl

        # local path
        p = Path(inp)
        if not p.exists():
            raise FileNotFoundError(p)

        if p.suffix == ".gz":
            out = p.with_suffix("")  # strip .gz
            if not out.exists():
                gunzip_file(p, out)
                print(f"✅ Unzipped: {p} → {out}")
            return out

        # already uncompressed .jsonl
        return p

    # No --input provided: prefer local gz default
    if default_gz.exists():
        out = default_jsonl
        if not out.exists():
            gunzip_file(default_gz, out)
            print(f"✅ Unzipped: {default_gz} → {out}")
        return out

    # or fall back to plain .jsonl if present
    if default_jsonl.exists():
        return default_jsonl

    # last resort: try S3 default if caller passed an S3Client and uses std key layout
    if s3 is not None:
        key = f"parsed/{default_gz.name}"  # matches your uploader pattern
        local_gz = Path("tmp") / default_gz.name
        local_jsonl = Path("tmp") / default_jsonl.name
        s3.download_file_if_missing(key, local_gz)
        if not local_jsonl.exists():
            gunzip_file(local_gz, local_jsonl)
            print(f"✅ Unzipped: {local_gz} → {local_jsonl}")
        return local_jsonl

    raise FileNotFoundError(
        f"Could not resolve decisions file for stake NL{stake}. "
        f"Tried {default_gz}, {default_jsonl}, and S3 (if provided)."
    )

def add_freqs_with_entropy(tbl: "pl.DataFrame", alpha: float = 1.0) -> "pl.DataFrame":
    """
    Given a table with columns n_fold, n_call, n_raise, n_rows,
    add Laplace-smoothed probabilities and entropy.
    """
    import polars as pl

    t = tbl.with_columns([
        ((pl.col("n_fold")  + alpha) / (pl.col("n_rows") + 3*alpha)).alias("p_fold"),
        ((pl.col("n_call")  + alpha) / (pl.col("n_rows") + 3*alpha)).alias("p_call"),
        ((pl.col("n_raise") + alpha) / (pl.col("n_rows") + 3*alpha)).alias("p_raise"),
    ])
    # Numerical stability guard in logs
    eps = 1e-12
    t = t.with_columns(
        (
            -(pl.col("p_fold")  * (pl.col("p_fold")  + eps).log()
              + pl.col("p_call")  * (pl.col("p_call")  + eps).log()
              + pl.col("p_raise") * (pl.col("p_raise") + eps).log())
        ).alias("entropy")
    )
    return t

def main():
    import argparse
    import polars as pl
    from pathlib import Path
    import json

    # ---- CLI ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=int, required=True, help="e.g. 10 for NL10")
    ap.add_argument("--input", type=str, default=None,
                    help="path or s3://… to decisions .jsonl(.gz). If omitted, tries decisions_nl{stake}.jsonl(.gz)")
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON (default: ml/config/coverage/populationnet_nl{stake}.json)")
    ap.add_argument("--min_rows_per_cell", type=int, default=200)
    ap.add_argument("--min_cells_per_ctx", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing for action probs")
    args = ap.parse_args()

    # ---- Init S3 + resolve local, unzipped path ----
    s3 = S3Client()
    decisions_path = resolve_input_path(args.stake, args.input, s3=s3)  # Path to local .jsonl (unzipped)

    out_path = Path(args.out or f"ml/config/coverage/populationnet_nl{args.stake}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    df = pl.read_ndjson(str(decisions_path))

    # Normalize actions: merge ALL_IN (5) -> RAISE (2)
    df = df.with_columns(
        pl.when(pl.col("act_id") == 5).then(2).otherwise(pl.col("act_id")).alias("act_id")
    )

    grp_keys = ["stakes_id", "street_id", "ctx_id", "hero_pos_id", "villain_pos_id"]

    # ---- Counts per (cell, act) ----
    agg = (
        df.group_by(grp_keys + ["act_id"])
          .count()
          .rename({"count": "n"})
    )

    # Pivot acts into columns (robust to act_id dtype as str/int)
    pivot = agg.pivot(values="n", index=grp_keys, columns="act_id").fill_null(0)

    # Ensure we have n_fold/n_call/n_raise present
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

    # n_rows + coverage flag
    pivot = pivot.with_columns([
        (pl.col("n_fold") + pl.col("n_call") + pl.col("n_raise")).alias("n_rows"),
    ]).with_columns([
        (pl.col("n_rows") >= args.min_rows_per_cell).alias("ok")
    ])

    # ---- Add frequencies + entropy on OK cells ----
    ok_cells_tbl = add_freqs_with_entropy(pivot.filter(pl.col("ok")), alpha=args.alpha)

    # ---- Context coverage summary ----
    ctx_cover = (
        pivot.group_by(["stakes_id", "street_id", "ctx_id"])
             .agg([
                 pl.len().alias("cells_total"),
                 pl.sum("ok").alias("cells_ok"),
                 pl.col("n_rows").sum().alias("rows_total"),
             ])
             .with_columns((pl.col("cells_ok") >= args.min_cells_per_ctx).alias("ctx_ok"))
    )

    # Friendly names
    CTX_NAME = {
        0: "OPEN",
        1: "VS_OPEN",
        2: "VS_3BET",
        3: "VS_4BET",
        4: "BLIND_VS_STEAL",
        5: "LIMPED_SINGLE",
        6: "LIMPED_MULTI",
        10: "VS_CBET",
        11: "VS_CBET_TURN",  # if you use it separately
        13: "VS_CHECK_RAISE",
        14: "VS_DONK",
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
                "min_rows_per_cell": args.min_rows_per_cell,
            })

    # OK cell list (ids only)
    ok_cells = (
        ok_cells_tbl
        .select(grp_keys)
        .unique()
        .to_dicts()
    )
    # Cast to ints for JSON cleanliness
    for r in ok_cells:
        for k in ("stakes_id", "street_id", "ctx_id", "hero_pos_id", "villain_pos_id"):
            r[k] = int(r[k])

    # ---- Frequencies per OK cell ----
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
        # keep floats for probs/entropy

    # ---- Context-level summaries (means over cells) ----
    ctx_summaries = (
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
    )
    for r in ctx_summaries:
        r["stakes_id"] = int(r["stakes_id"])
        r["street_id"] = int(r["street_id"])
        r["ctx_id"] = int(r["ctx_id"])
        r["street"] = STREET_NAME.get(r["street_id"], str(r["street_id"]))
        r["ctx"] = CTX_NAME.get(r["ctx_id"], str(r["ctx_id"]))

    # ---- Build JSON ----
    cfg = {
        "populationnet": {
            "generated_from": str(decisions_path),
            "thresholds": {
                "min_rows_per_cell": args.min_rows_per_cell,
                "min_cells_per_ctx": args.min_cells_per_ctx,
                "alpha": args.alpha,
            },
            "include_contexts": include,
            "ok_cells": ok_cells,
            "freq_cells": freq_cells,
            "ctx_summaries": ctx_summaries,
            "actions": ["FOLD", "CALL", "RAISE"],
        }
    }

    out_path.write_text(json.dumps(cfg, indent=2))
    print(f"✅ wrote coverage+freqs → {out_path}")

if __name__ == "__main__":
    main()