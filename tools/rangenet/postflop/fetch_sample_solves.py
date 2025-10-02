#!/usr/bin/env python3
# tools/rangenet/postflop/fetch_sample_solves.py
import argparse
import json
import re
from pathlib import Path

import boto3
import pandas as pd


DEFAULT_MENUS = [
    "srp_hu.PFR_IP",
    "3bet_hu.Aggressor_IP",
    "3bet_hu.Aggressor_OOP",
    "4bet_hu.Aggressor_IP",
    "4bet_hu.Aggressor_OOP",
    "limped_single.SB_IP",
    "limped_multi.Any",
]


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def pick_one(df: pd.DataFrame, menu: str) -> dict | None:
    d = df[(df["bet_sizing_id"] == menu) & (df["street"] == 1)]
    if "node_key" in d.columns:
        d = d[d["node_key"] == "root"]
    if d.empty:
        return None
    return d.iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser(
        "Fetch one sample solve per bet menu and prepare inputs for build_solve_maps.py"
    )
    ap.add_argument(
        "--manifest",
        default="data/artifacts/rangenet_postflop_flop_manifest.parquet",
        help="Parquet manifest with s3_key rows",
    )
    ap.add_argument(
        "--bucket",
        default="pokeraistore",
        help="S3 bucket name that stores solver outputs",
    )
    ap.add_argument(
        "--out-dir",
        default="data/debug_samples",
        help="Local dir to download sample JSON.GZ files",
    )
    ap.add_argument(
        "--menus",
        nargs="*",
        default=None,
        help="Explicit bet_sizing_id list. If omitted, uses a fixed 7-menu default.",
    )
    ap.add_argument(
        "--auto-from-manifest",
        action="store_true",
        help="Ignore --menus and auto-pick one per bet_sizing_id present in the manifest.",
    )
    ap.add_argument(
        "--pot",
        type=float,
        default=20.0,
        help="Default pot (bb) hint for downstream %→BB math in build_solve_maps.py",
    )
    ap.add_argument(
        "--stack",
        type=float,
        default=100.0,
        help="Default effective stack (bb) hint for downstream %→BB math",
    )
    ap.add_argument(
        "--out-map",
        default="data/artifacts/solve_map_inputs.json",
        help="Where to save the produced menu→localpath map",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest)

    if args.auto_from_manifest:
        menus = sorted(map(str, df["bet_sizing_id"].dropna().unique()))
    else:
        menus = args.menus or DEFAULT_MENUS

    # Sanity: which requested menus are missing in the manifest?
    have = set(map(str, df["bet_sizing_id"].dropna().unique()))
    missing = [m for m in menus if m not in have]
    if missing:
        print(f"⚠️  Missing menus in manifest (will skip): {missing}")

    s3 = boto3.client("s3")

    menu_to_local = {}
    for menu in menus:
        row = pick_one(df, menu)
        if not row:
            print(f"⚠️  No sample row for {menu}; skipping")
            continue
        s3_key = str(row["s3_key"])
        fn = f"{safe_name(menu)}.json.gz"
        local = out_dir / fn
        print(f"↓ {menu} ← s3://{args.bucket}/{s3_key}")
        s3.download_file(args.bucket, s3_key, str(local))
        menu_to_local[menu] = str(local)

    if not menu_to_local:
        raise SystemExit("No samples downloaded — nothing to do.")

    # Save the small map (menu → local gz file)
    out_map_path = Path(args.out_map)
    out_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pot_hint_bb": args.pot,
                "stack_hint_bb": args.stack,
                "inputs": menu_to_local,
            },
            f,
            indent=2,
        )

    # Print the exact follow-up command
    input_pairs = " ".join([f"{k}={v}" for k, v in menu_to_local.items()])
    cmd = (
        "python tools/rangenet/sanity/build_solve_maps.py"
        f" --input {input_pairs}"
        f" --pot {args.pot}"
        f" --stack {args.stack}"
        " --out data/artifacts/solve_map.json"
    )

    print(f"\n✅ Downloaded {len(menu_to_local)} sample solves → {out_dir}")
    print(f"🗺  Wrote: {out_map_path}")
    print("\nNext, run:")
    print(cmd)


if __name__ == "__main__":
    main()