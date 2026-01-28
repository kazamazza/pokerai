#!/usr/bin/env python3
"""
Sanity check dataset contracts against:
  1) manifest parquet (pre-solve spec)
  2) policy parts parquet dirs (post-extract training-ready)

Usage examples:
  # Check only manifest vs dataset config:
  python tools/rangenet/sanity/sanity_dataset_contract.py \
    --config rangenet/postflop_base.yaml \
    --manifest data/artifacts/rangenet_postflop_manifest.parquet \
    --dataset-key dataset_postflop_root

  # Check both root + facing parts against config:
  python tools/rangenet/sanity/sanity_dataset_contract.py \
    --config rangenet/postflop_base.yaml \
    --manifest data/artifacts/rangenet_postflop_manifest.parquet \
    --root-parts data/datasets/postflop_policy_root_parts \
    --facing-parts data/datasets/postflop_policy_facing_parts

Notes:
- We *expect* some columns only exist post-extraction (e.g. size_pct/faced_size_pct, valid, solver_key).
  So manifest checks only validate columns that should exist in manifest.
- Parts checks validate full dataset contract (x_cols + y_cols + weight/valid).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


# -------------------------
# YAML helpers
# -------------------------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def _get(cfg: Dict[str, Any], key: str, default=None):
    return cfg.get(key, default)


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


# -------------------------
# parquet helpers
# -------------------------
def read_one_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"parquet not found: {path}")
    return pd.read_parquet(path)


def find_first_parquet(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = sorted(dir_path.glob("*.parquet"))
    return files[0] if files else None


def require_columns(df_cols: List[str], required: List[str], *, label: str) -> None:
    missing = [c for c in required if c not in df_cols]
    if missing:
        raise RuntimeError(f"{label}: missing columns: {missing}")


def _print_col_summary(df: pd.DataFrame, cols: List[str], *, title: str) -> None:
    print(f"\n== {title} ==")
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        miss = int(s.isna().sum())
        n = len(s)
        uniq = int(s.nunique(dropna=True))
        print(f"  {c:20s}  missing={miss:8d}/{n:<8d}  unique={uniq:<8d}  dtype={str(s.dtype)}")


# -------------------------
# contract logic
# -------------------------
MANIFEST_EXPECTED = {
    # should exist in manifest
    "stake",
    "scenario",
    "street",
    "ctx",
    "topology",
    "role",
    "ip_pos",
    "oop_pos",
    "board_cluster_id",
    "board",
    "pot_bb",
    "effective_stack_bb",
    "bet_sizing_id",
    "bet_sizes",
    "accuracy",
    "max_iter",
    "allin_threshold",
    "solver_version",
    "sha1",
    "s3_key",
    "node_key",
    "weight",
}


def manifest_contract_check(manifest_path: Path, dataset_cfg: Dict[str, Any], *, dataset_key: str) -> None:
    df = read_one_parquet(manifest_path)
    cols = list(df.columns)

    x_cols = _as_list(dataset_cfg.get("x_cols"))
    y_cols = _as_list(dataset_cfg.get("y_cols"))
    weight_col = dataset_cfg.get("weight_col")
    valid_col = dataset_cfg.get("valid_col")

    # Only validate manifest for columns that should exist pre-extraction
    manifest_x = [c for c in x_cols if c in MANIFEST_EXPECTED]
    manifest_extra_expected = []
    if weight_col and weight_col in MANIFEST_EXPECTED:
        manifest_extra_expected.append(str(weight_col))
    if valid_col and valid_col in MANIFEST_EXPECTED:
        manifest_extra_expected.append(str(valid_col))

    print(f"\n--- Manifest check for dataset='{dataset_key}' ---")
    print(f"manifest rows={len(df):,} cols={len(cols):,}")

    require_columns(cols, manifest_x + manifest_extra_expected, label="manifest")

    _print_col_summary(df, manifest_x + manifest_extra_expected, title="manifest feature columns (that should exist pre-extract)")

    # y_cols are not expected in manifest (they are policy outputs), but we can sanity-print them if present
    present_y = [c for c in y_cols if c in cols]
    if present_y:
        _print_col_summary(df, present_y, title="manifest y_cols (unexpectedly present)")
    else:
        print("  (as expected) y_cols not present in manifest.")


def parts_contract_check(parts_dir: Path, dataset_cfg: Dict[str, Any], *, dataset_key: str) -> None:
    p = find_first_parquet(parts_dir)
    if p is None:
        raise RuntimeError(f"{dataset_key}: no parquet files found in dir: {parts_dir}")

    df = read_one_parquet(p)
    cols = list(df.columns)

    x_cols = _as_list(dataset_cfg.get("x_cols"))
    cont_cols = _as_list(dataset_cfg.get("cont_cols"))
    y_cols = _as_list(dataset_cfg.get("y_cols"))

    weight_col = dataset_cfg.get("weight_col")
    valid_col = dataset_cfg.get("valid_col")

    required = []
    required.extend(x_cols)
    required.extend(cont_cols)
    required.extend(y_cols)
    if weight_col:
        required.append(str(weight_col))
    if valid_col:
        required.append(str(valid_col))

    print(f"\n--- Parts check for dataset='{dataset_key}' ---")
    print(f"parts_dir={parts_dir}")
    print(f"sample_file={p.name} rows={len(df):,} cols={len(cols):,}")

    require_columns(cols, required, label=f"{dataset_key} parts")

    # Basic checks: valid rows should have weight > 0 and probs sum ~ 1 (for y vocab rows)
    if valid_col and valid_col in df.columns and weight_col and weight_col in df.columns:
        v = df[valid_col].astype("float").fillna(0.0)
        w = df[weight_col].astype("float").fillna(0.0)
        bad = int(((v > 0.5) & (w <= 0.0)).sum())
        if bad:
            raise RuntimeError(f"{dataset_key}: found {bad} rows with valid=1 but weight<=0 in sample file {p.name}")
        else:
            print("  ✅ valid/weight basic check passed (sample file).")

    # Probability mass check (sample file only)
    if y_cols:
        y = df[y_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        s = y.sum(axis=1)
        # only check valid rows if valid_col exists
        if valid_col and valid_col in df.columns:
            mask = df[valid_col].fillna(0).astype(int) == 1
            s = s[mask]
        if len(s) > 0:
            frac_bad = float(((s < 0.99) | (s > 1.01)).mean())
            print(f"  prob-sum check (sample file): bad_frac={frac_bad:.4f} (tolerance 0.99..1.01)")
            # do not hard-fail unless it's really bad; can be minor due to float serialization
            if frac_bad > 0.05:
                raise RuntimeError(f"{dataset_key}: too many rows with prob-sum outside [0.99,1.01] in {p.name}")

    _print_col_summary(df, x_cols + cont_cols, title=f"{dataset_key} x/cont columns (sample file)")
    _print_col_summary(df, y_cols, title=f"{dataset_key} y columns (sample file)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config containing dataset_* sections")
    ap.add_argument("--manifest", required=True, help="Manifest parquet path")

    ap.add_argument("--dataset-key", default=None, help="Check only one dataset key (e.g. dataset_postflop_root)")
    ap.add_argument("--root-parts", default=None, help="Optional dir of root parts parquet to validate")
    ap.add_argument("--facing-parts", default=None, help="Optional dir of facing parts parquet to validate")

    args = ap.parse_args()

    cfg = load_yaml(args.config)

    manifest_path = Path(args.manifest)

    # Determine dataset keys to check
    if args.dataset_key:
        keys = [args.dataset_key]
    else:
        # default: check both common ones if present
        keys = [k for k in ("dataset_postflop_root", "dataset_postflop_facing") if k in cfg]
        if not keys:
            raise RuntimeError("No dataset keys found. Provide --dataset-key or add dataset_* sections to YAML.")

    for key in keys:
        ds_cfg = cfg.get(key) or {}
        if not isinstance(ds_cfg, dict):
            raise RuntimeError(f"{key} must be a dict in YAML")

        manifest_contract_check(manifest_path, ds_cfg, dataset_key=key)

        # Optional parts checks
        if key.endswith("_root") and args.root_parts:
            parts_contract_check(Path(args.root_parts), ds_cfg, dataset_key=key)
        if key.endswith("_facing") and args.facing_parts:
            parts_contract_check(Path(args.facing_parts), ds_cfg, dataset_key=key)

    print("\n✅ dataset contract sanity checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n❌ sanity check FAILED: {e}")
        raise SystemExit(2)