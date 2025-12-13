#!/usr/bin/env python3
# tools/sanity_ev.py
import argparse, numpy as np, pandas as pd
import sys
from typing import List, Optional, Tuple, Dict

def _ev_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("ev_")]
    if not cols:
        raise ValueError("No ev_* columns found.")
    return cols

def _to_mask_matrix(df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    if col not in df.columns:
        return None
    raw = df[col].to_numpy()
    if len(raw) == 0:
        return None
    first = np.asarray(raw[0], dtype=np.float32)
    V = int(first.size)
    mat = np.zeros((len(raw), V), dtype=np.float32)
    for i, v in enumerate(raw):
        a = np.asarray(v, dtype=np.float32)
        if a.size != V:
            raise ValueError(f"Mask column '{col}' has inconsistent length at row {i}: {a.size} vs {V}")
        mat[i] = a
    if col.startswith("illegal_mask"):
        mat = 1.0 - mat  # convert illegal→legal mask
    return mat

def _coalesce_mask(df: pd.DataFrame, *cols: str) -> Optional[np.ndarray]:
    """
    Return the first available mask matrix among the given columns.
    Uses _to_mask_matrix(...) to parse list/array masks stored per-row.
    If none of the columns exist, returns None.
    """
    for c in cols:
        m = _to_mask_matrix(df, c)
        if m is not None:
            return m
    return None


def _assert_mask_ok(df: pd.DataFrame, col: str, vocab_len: int):
    """
    Optional: validate that a mask column (if present) is the right width and binary.
    Converts illegal→legal inside _to_mask_matrix already; this just checks shape/values.
    """
    if col not in df.columns:
        return
    m = _to_mask_matrix(df, col)
    if m is None:
        return
    assert m.shape[1] == vocab_len, f"{col} width {m.shape[1]} != vocab {vocab_len}"
    u = np.unique(m)
    # allow minor float noise from parquet round-trips
    ub = np.unique(np.round(u, 0))
    assert set(ub).issubset({0, 1}), f"{col} has non-binary values: {u}"

def _mae_zero_baseline(df: pd.DataFrame, mask: Optional[np.ndarray]) -> Tuple[float, Dict[str, float]]:
    """Return overall MAE(bb) if model predicts zeros, and percentiles of |y|."""
    Y = df[_ev_cols(df)].astype(np.float32).values  # [N,V]
    A = np.abs(Y)
    if mask is not None:
        # row-wise mean over legal actions only
        denom = mask.sum(axis=1)
        denom = np.where(denom <= 0, 1.0, denom)
        row_mae = (A * mask).sum(axis=1) / denom
    else:
        row_mae = A.mean(axis=1)
    mae = float(row_mae.mean())
    flat = A.ravel()
    pct = {
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(np.max(flat)),
    }
    return mae, pct

def _print_top_k_counts(name: str, s: pd.Series, k: int = 8):
    vc = s.value_counts(dropna=False)
    tot = int(vc.sum())
    print(f"  {name}: {tot} distinct" if len(vc) > k else f"  {name}:")
    for idx, cnt in vc.head(k).items():
        pct = cnt / tot if tot else 0
        print(f"    - {idx}: {cnt} ({pct:.3f})")
    if len(vc) > k:
        print(f"    ... (+{len(vc)-k} more)")

def _group_mae(df: pd.DataFrame, mask: Optional[np.ndarray], by: List[str], title: str):
    if not by:
        return
    print(f"\n— {title} —")
    evc = _ev_cols(df)
    Y = df[evc].astype(np.float32).values
    A = np.abs(Y)
    # Precompute row-wise denominators for masked mean
    if mask is not None:
        denom = mask.sum(axis=1)
        denom = np.where(denom <= 0, 1.0, denom)
        row_mae_vec = (A * mask).sum(axis=1) / denom
    else:
        row_mae_vec = A.mean(axis=1)
    tmp = df[by].copy()
    tmp["row_mae"] = row_mae_vec
    g = tmp.groupby(by, dropna=False)["row_mae"].mean().sort_values()
    for idx, val in g.items():
        print(f"  {idx}: {val:.3f} bb")

def _assert_mask_ok(df: pd.DataFrame, col: str, vocab_len: int):
    if col not in df.columns: return
    m = _to_mask_matrix(df, col)
    if m is None: return
    assert m.shape[1] == vocab_len, f"{col} width {m.shape[1]} != vocab {vocab_len}"
    u = np.unique(m)
    assert set(np.round(u, 0)).issubset({0, 1}), f"{col} has non-binary values: {u}"

def check_postflop_root(path: str):
    print("\n=== Checking POSTFLOP ROOT ===")
    df = pd.read_parquet(path)
    print(f"rows={len(df)} | cols={len(df.columns)}")

    # Basic feature sanity
    if "board_cluster_id" in df.columns:
        uniq = int(df["board_cluster_id"].nunique())
        print(f"  board_cluster_id unique={uniq}")
        _print_top_k_counts("board_cluster_id head", df["board_cluster_id"], k=8)
    if "size_frac" in df.columns:
        print(f"  size_frac unique={sorted(df['size_frac'].dropna().unique().tolist())[:8]} (expect [0.0] at root)")

    # Masked zero baseline
    mask = _coalesce_mask(df, "illegal_mask_root", "y_mask")
    mae, pct = _mae_zero_baseline(df, mask)
    print(f"  zero-baseline MAE: {mae:.3f} bb | |y| p50={pct['p50']:.2f} p90={pct['p90']:.2f} p95={pct['p95']:.2f} max={pct['max']:.2f}")

    # Group breakdowns
    by = [c for c in ["ctx", "hero_pos", "board_cluster_id", "stack_bb"] if c in df.columns]
    _group_mae(df, mask, ["ctx"], "Per-ctx MAE (zero baseline)")
    if "size_frac" in df.columns:
        _group_mae(df, mask, ["size_frac"], "Per-size_frac MAE (should be trivial at root)")
    if by:
        _group_mae(df, mask, by[:3], "Combined groups (first 3 keys)")

def check_postflop_facing(path: str):
    print("\n=== Checking POSTFLOP FACING ===")
    df = pd.read_parquet(path)
    print(f"rows={len(df)} | cols={len(df.columns)}")

    if "board_cluster_id" in df.columns:
        uniq = int(df["board_cluster_id"].nunique())
        print(f"  board_cluster_id unique={uniq}")
        _print_top_k_counts("board_cluster_id head", df["board_cluster_id"], k=8)
    if "size_frac" in df.columns:
        _print_top_k_counts("size_frac", df["size_frac"], k=8)

    mask = _coalesce_mask(df, "illegal_mask_facing", "y_mask")
    mae, pct = _mae_zero_baseline(df, mask)
    print(f"  zero-baseline MAE: {mae:.3f} bb | |y| p50={pct['p50']:.2f} p90={pct['p90']:.2f} p95={pct['p95']:.2f} max={pct['max']:.2f}")

    _group_mae(df, mask, ["ctx"], "Per-ctx MAE (zero baseline)")
    if "size_frac" in df.columns:
        _group_mae(df, mask, ["size_frac"], "Per-size_frac MAE")
        _group_mae(df, mask, ["ctx","size_frac"], "Per ctx × size_frac MAE")

def check_preflop(path: str):
    print("\n=== Checking PREFLOP ===")
    df = pd.read_parquet(path)
    print(f"rows={len(df)} | cols={len(df.columns)}")

    # Basic flags
    if "facing_flag" in df.columns:
        _print_top_k_counts("facing_flag", df["facing_flag"], k=8)
    if "free_check" in df.columns:
        _print_top_k_counts("free_check", df["free_check"], k=8)
    if "hero_pos" in df.columns:
        _print_top_k_counts("hero_pos", df["hero_pos"], k=10)

    # Mask: preflop builder writes no illegal-mask by default; we treat all actions as legal here.
    mask = _coalesce_mask(df, "y_mask") # or None
    mae, pct = _mae_zero_baseline(df, mask)
    print(f"  zero-baseline MAE: {mae:.3f} bb | |y| p50={pct['p50']:.2f} p90={pct['p90']:.2f} p95={pct['p95']:.2f} max={pct['max']:.2f}")

    # Groups
    keys = [k for k in ["facing_flag", "free_check", "hero_pos", "villain_pos"] if k in df.columns]
    for k in keys:
        _group_mae(df, mask, [k], f"Per-{k} MAE (zero baseline)")

def _safe_check(name: str, fn, path: str):
    """
    Run a checker and return (ok: bool, info: str).
    If it passes (no exception), also report rows/cols for a quick glance.
    """
    try:
        fn(path)
        # lightweight meta for the summary
        try:
            df = pd.read_parquet(path)
            info = f"{len(df)} rows, {len(df.columns)} cols"
        except Exception:
            info = "ok"
        return True, info
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def main():
    ap = argparse.ArgumentParser(description="Sanity checker for EV parquets.")
    ap.add_argument("--root", type=str, help="Path to postflop root parquet")
    ap.add_argument("--facing", type=str, help="Path to postflop facing parquet")
    ap.add_argument("--preflop", type=str, help="Path to preflop parquet")
    args = ap.parse_args()

    if not any([args.root, args.facing, args.preflop]):
        ap.error("Provide at least one of --root/--facing/--preflop")

    results = {}

    if args.root:
        results["Postflop Root"] = _safe_check("root", check_postflop_root, args.root)
    if args.facing:
        results["Postflop Facing"] = _safe_check("facing", check_postflop_facing, args.facing)
    if args.preflop:
        results["Preflop"] = _safe_check("preflop", check_preflop, args.preflop)

    # Consolidated summary
    print("\n=== SUMMARY ===")
    all_ok = True
    for label, (ok, info) in results.items():
        if ok:
            print(f"✅ {label}: {info}")
        else:
            print(f"❌ {label}: {info}")
            all_ok = False

    if all_ok:
        print("🎉 All requested EV parquet checks passed.")
        sys.exit(0)
    else:
        print("⚠️ Some checks failed. See messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
