# tools/rangenet/build_postflop_cat_vocabs.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ml.training.postflop.vocab import (
    build_categorical_vocabs_from_parquets,
    list_parquets_under,
    save_vocabs_json,
)

DEFAULT_CAT_COLS: List[str] = [
    "solver_version",
    "street",
    "ctx",
    "topology",
    "role",
    "bet_sizing_id",
    "ip_pos",
    "oop_pos",
    "board_cluster_id",
]


def main() -> None:
    ap = argparse.ArgumentParser("Build postflop categorical vocabs (root/facing parts)")
    ap.add_argument(
        "--root-parts",
        type=str,
        default=None,
        help="Dir (or parquet) for root parts. Example: data/datasets/postflop_policy_parts_root",
    )
    ap.add_argument(
        "--facing-parts",
        type=str,
        default=None,
        help="Dir (or parquet) for facing parts. Example: data/datasets/postflop_policy_parts_facing",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output vocab json path, e.g. data/artifacts/postflop_cat_vocabs.json",
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum frequency for a category token to get its own id (else UNK).",
    )
    ap.add_argument(
        "--cat-cols",
        type=str,
        default=",".join(DEFAULT_CAT_COLS),
        help=f"Comma-separated categorical columns. Default: {','.join(DEFAULT_CAT_COLS)}",
    )
    args = ap.parse_args()

    cat_cols = [c.strip() for c in str(args.cat_cols).split(",") if c.strip()]
    if not cat_cols:
        raise SystemExit("No cat-cols provided")

    if not args.root_parts and not args.facing_parts:
        raise SystemExit("Provide at least one of --root-parts or --facing-parts")

    parquet_paths: List[str] = []
    if args.root_parts:
        parquet_paths.extend(list_parquets_under(args.root_parts))
    if args.facing_parts:
        parquet_paths.extend(list_parquets_under(args.facing_parts))

    parquet_paths = sorted(set(parquet_paths))
    if not parquet_paths:
        raise SystemExit("No parquet files found under provided paths")

    vocabs = build_categorical_vocabs_from_parquets(
        parquet_paths,
        cat_cols=cat_cols,
        min_count=int(args.min_count),
    )

    meta = {
        "n_parquets": len(parquet_paths),
        "sources": {
            "root_parts": args.root_parts,
            "facing_parts": args.facing_parts,
        },
        "cat_cols": cat_cols,
        "min_count": int(args.min_count),
    }

    save_vocabs_json(vocabs, out_path=args.out, meta=meta)

    print("✅ wrote vocabs:", args.out)
    for c in cat_cols:
        v = vocabs.get(c)
        if v is None:
            continue
        # includes UNK
        print(f"  {c:16s} size={v.size}")


if __name__ == "__main__":
    main()