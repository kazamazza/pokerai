#!/usr/bin/env python3
import argparse, math, json
from collections import Counter
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

EXPECTED_CTXS = {"VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"}
EXPECTED_ROLES = {"IP", "OOP"}
EXPECTED_SEATS = {"UTG","HJ","CO","BTN","SB","BB"}

REQ_X_COLS = ["hero_pos","ip_pos","oop_pos","ctx","street","board_cluster_id","stakes_id","hand_id"]
REQ_CONT_COLS = ["board_mask_52","pot_bb","stack_bb","size_frac"]

def infer_vocab_from_cols(cols: Sequence[str]) -> List[str]:
    toks = [c[3:] for c in cols if isinstance(c, str) and c.startswith("ev_")]
    return sorted(toks)

def has_board_mask_52(df: pd.DataFrame) -> bool:
    if "board_mask_52" in df.columns:
        return True
    # allow exploded bm0..bm51
    return all((f"bm{i}" in df.columns) for i in range(52))

def get_illegal_mask_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("illegal_mask_root","illegal_mask_facing"):
        if c in df.columns: return c
    return None

def summarize_head(df: pd.DataFrame, n: int = 3) -> str:
    return df.head(n).to_string(index=False)

def check_required(df: pd.DataFrame, cols: Sequence[str], label: str, errors: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        errors.append(f"[{label}] missing columns: {missing}")

def check_no_nans(df: pd.DataFrame, cols: Sequence[str], label: str, errors: List[str]):
    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        errors.append(f"[{label}] NaNs detected in: {bad}")

def check_cluster_diversity(df: pd.DataFrame, errors: List[str], warnings: List[str]):
    vc = df["board_cluster_id"].value_counts()
    uniq = len(vc)
    if uniq <= 1:
        errors.append(f"[cluster] board_cluster_id has {uniq} unique value(s) → likely broken clustering")
    else:
        top_prop = float(vc.iloc[0]) / float(len(df))
        if top_prop > 0.50:
            warnings.append(f"[cluster] Top cluster holds {top_prop:.1%} of rows (uniq={uniq}).")

def check_roles_and_ctx(df: pd.DataFrame, label: str, errors: List[str], warnings: List[str]):
    ip_ok = set(df["ip_pos"].unique()).issubset({"IP"})
    oop_ok = set(df["oop_pos"].unique()).issubset({"OOP"})
    if not ip_ok or not oop_ok:
        errors.append(f"[{label}] ip_pos/oop_pos must be role tokens 'IP'/'OOP'. "
                      f"Seen ip={df['ip_pos'].unique()[:5]}, oop={df['oop_pos'].unique()[:5]}")
    bad_ctx = set(df["ctx"].unique()) - EXPECTED_CTXS
    if bad_ctx:
        warnings.append(f"[{label}] unexpected ctx values: {sorted(bad_ctx)}")
    bad_seats = set(df["hero_pos"].unique()) - EXPECTED_SEATS
    if bad_seats:
        warnings.append(f"[{label}] unexpected hero_pos seats: {sorted(bad_seats)}")

def check_masks(df: pd.DataFrame, vocab: List[str], label: str, errors: List[str], warnings: List[str]):
    mcol = get_illegal_mask_col(df)
    if not mcol:
        warnings.append(f"[{label}] no illegal mask column present (ok if you chose not to emit it).")
        return
    # Sample a handful to avoid scanning huge frames
    sample = df[mcol].head(100)
    bad_len = [i for i, v in enumerate(sample.values) if not isinstance(v, (list, tuple, np.ndarray)) or len(v) != len(vocab)]
    if bad_len:
        errors.append(f"[{label}] {mcol} length != |vocab| on some rows (expected {len(vocab)}).")
    # Optional: DONK_* should be illegal at ROOT
    if mcol == "illegal_mask_root":
        donk_idx = [i for i,t in enumerate(vocab) if t.upper().startswith("DONK_")]
        if donk_idx:
            arr = np.array([list(r) for r in sample if isinstance(r,(list,tuple,np.ndarray)) and len(r)==len(vocab)])
            if arr.size:
                # proportion of DONK entries marked illegal (1)
                p_illegal = float(arr[:, donk_idx].mean())
                if p_illegal < 0.90:
                    warnings.append(f"[{label}] DONK_* not masked illegal at root enough (mean={p_illegal:.2f}).")

def check_root(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    errors, warnings = [], []
    label = "root"
    check_required(df, REQ_X_COLS, label, errors)
    check_required(df, REQ_CONT_COLS, label, errors)
    if errors: return errors, warnings

    check_no_nans(df, REQ_X_COLS + REQ_CONT_COLS, label, errors)
    check_roles_and_ctx(df, label, errors, warnings)

    # street must be 1
    if not set(df["street"].unique()).issubset({1}):
        errors.append("[root] street must be 1 for postflop root.")

    # size_frac must be all zeros at root
    nz = float((df["size_frac"].abs() > 1e-9).mean())
    if nz > 0.001:
        warnings.append(f"[root] {nz:.2%} rows have non-zero size_frac (should be 0.0 at root).")

    # cluster diversity
    check_cluster_diversity(df, errors, warnings)

    # board mask presence
    if not has_board_mask_52(df):
        warnings.append("[root] board_mask_52 (or bm0..bm51) not present.")

    # vocab & masks
    vocab = infer_vocab_from_cols(df.columns)
    if not vocab:
        errors.append("[root] no ev_* columns found.")
    else:
        check_masks(df, vocab, label, errors, warnings)

    # simple stat dump
    ctx_counts = df["ctx"].value_counts(normalize=True).round(3).to_dict()
    warnings.append(f"[root] rows={len(df):,}, ctx share={ctx_counts}")
    return errors, warnings

def check_facing(df: pd.DataFrame, expected_size_fracs: Optional[List[float]]) -> Tuple[List[str], List[str]]:
    errors, warnings = [], []
    label = "facing"
    check_required(df, REQ_X_COLS, label, errors)
    check_required(df, REQ_CONT_COLS, label, errors)
    if errors: return errors, warnings

    check_no_nans(df, REQ_X_COLS + REQ_CONT_COLS, label, errors)
    check_roles_and_ctx(df, label, errors, warnings)

    # street must be 1
    if not set(df["street"].unique()).issubset({1}):
        errors.append("[facing] street must be 1 for postflop facing.")

    # size_frac sanity
    uniq = sorted(set(np.round(df["size_frac"].astype(float), 4).tolist()))
    if any((x <= 0 or x > 1) for x in uniq):
        errors.append(f"[facing] size_frac must be in (0,1]; seen {uniq[:6]}...")
    if expected_size_fracs:
        exp = sorted(set([round(float(x), 2) for x in expected_size_fracs]))
        got = sorted(set([round(float(x), 2) for x in uniq]))
        if not set(got).issubset(set(exp)):
            warnings.append(f"[facing] size_frac values {got} not subset of expected {exp} (ok if you changed config).")

    # cluster diversity
    check_cluster_diversity(df, errors, warnings)

    # board mask presence
    if not has_board_mask_52(df):
        warnings.append("[facing] board_mask_52 (or bm0..bm51) not present.")

    # vocab & masks
    vocab = infer_vocab_from_cols(df.columns)
    if not vocab:
        errors.append("[facing] no ev_* columns found.")
    else:
        check_masks(df, vocab, label, errors, warnings)

    # cheap faced vs stack plausibility (raises shouldn’t all be overstack)
    if "stack_bb" in df.columns and "pot_bb" in df.columns and "size_frac" in df.columns:
        faced = (df["pot_bb"].astype(float) * df["size_frac"].astype(float)).values
        over = float((faced > df["stack_bb"].astype(float).values + 1e-9).mean())
        if over > 0.05:
            warnings.append(f"[facing] {over:.1%} rows have faced_bb > stack_bb (may be fine if stacks very small).")

    # simple stat dump
    ctx_counts = df["ctx"].value_counts(normalize=True).round(3).to_dict()
    sf_counts  = df["size_frac"].value_counts(normalize=True).round(3).to_dict()
    warnings.append(f"[facing] rows={len(df):,}, ctx share={ctx_counts}, size_frac share={sf_counts}")
    return errors, warnings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=False, help="Path to postflop root parquet")
    ap.add_argument("--facing", type=str, required=False, help="Path to postflop facing parquet")
    ap.add_argument("--size-fracs", type=float, nargs="*", default=None, help="Expected facing size_fracs (e.g. 0.33 0.66)")
    args = ap.parse_args()

    had_any = False

    if args.root:
        print("\n=== Checking POSTFLOP ROOT ===")
        df = pd.read_parquet(args.root)
        errs, warns = check_root(df)
        if errs:
            print("❌ Errors:"); [print("  -", e) for e in errs]
        if warns:
            print("⚠️  Warnings:"); [print("  -", w) for w in warns]
        if not errs: print("✅ Root parquet looks OK.")
        had_any = True

    if args.facing:
        print("\n=== Checking POSTFLOP FACING ===")
        df = pd.read_parquet(args.facing)
        errs, warns = check_facing(df, args.size_fracs)
        if errs:
            print("❌ Errors:"); [print("  -", e) for e in errs]
        if warns:
            print("⚠️  Warnings:"); [print("  -", w) for w in warns]
        if not errs: print("✅ Facing parquet looks OK.")
        had_any = True

    if not had_any:
        print("Provide --root and/or --facing paths to check.")

if __name__ == "__main__":
    main()