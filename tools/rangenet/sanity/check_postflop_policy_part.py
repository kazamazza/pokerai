#!/usr/bin/env python
# tools/rangenet/postflop/sanity/check_postflop_policy_part.py

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Core required columns
REQUIRED_COLS: List[str] = [
    "stack_bb", "pot_bb",
    "hero_pos", "ip_pos", "oop_pos",
    "street", "ctx",
    "actor",          # "ip" | "oop"
    "action",         # "CHECK" | "BET_33" | "DONK_33" | "RAISE_200" | ...
    "weight",
]

# Either "board_cluster" OR "board" must be present
BOARD_ALTERNATIVES = [["board_cluster"], ["board"]]

# Optional fields often present
OPTIONAL_COLS = [
    "bet_sizing_id",  # e.g. srp_hu.PFR_IP
    "bet_size_pct",   # numeric percentage (can be NaN for CHECK)
    "node_key",
]

# Patterns
ACTION_TOKEN_RE = re.compile(r"^(FOLD|CHECK|CALL|ALLIN|BET_\d+|DONK_\d+|RAISE_\d+)$")
PROB_COL_RE     = re.compile(r"^(FOLD|CHECK|CALL|ALLIN|BET_\d+|DONK_\d+|RAISE_\d+)$")

def fail(msg: str):
    raise SystemExit(f"❌ {msg}")

def ok(msg: str):
    print(f"✅ {msg}")

def warn(msg: str):
    print(f"⚠️ {msg}")

def top_counts(df: pd.DataFrame, col: str, k=10):
    if col in df.columns:
        vc = df[col].value_counts(dropna=False).head(k)
        print(f"\nby {col} (top-{k}):")
        for idx, cnt in vc.items():
            print(f"  {idx}: {cnt}")

def main():
    ap = argparse.ArgumentParser("Sanity check a postflop policy parquet part")
    ap.add_argument("--part", type=Path, required=True,
                    help="Path to a parquet part (e.g. data/datasets/postflop_policy_parts_test/part-00000.parquet)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="Tolerance for probability sum ≈ 1")
    args = ap.parse_args()

    p = args.part
    if not p.exists():
        fail(f"File not found: {p}")

    df = pd.read_parquet(p)
    print(f"=== FILE ===\npath: {p}\nrows: {len(df):,}\n")

    # ---------- schema checks ----------
    cols = set(df.columns)

    missing = [c for c in REQUIRED_COLS if c not in cols and c not in ("board_cluster", "board")]
    has_board_alt = any(all(c in cols for c in alt) for alt in BOARD_ALTERNATIVES)
    if not has_board_alt:
        missing.append("board_cluster|board")
    if missing:
        fail(f"Missing required columns: {missing}")
    ok("Required columns present")

    present_optional = [c for c in OPTIONAL_COLS if c in cols]
    print(f"optional columns present: {present_optional}")

    # ensure no range (y_*) columns
    y_cols = [c for c in df.columns if c.startswith("y_")]
    if y_cols:
        fail(f"Found unexpected range columns (y_*): count={len(y_cols)} e.g. {y_cols[:5]}")
    ok("No y_* columns present (policy dataset)")

    # ---------- nulls & basic types ----------
    critical = ["stack_bb", "pot_bb", "hero_pos", "ip_pos", "oop_pos", "street", "ctx", "actor", "action", "weight"]
    critical += [c for c in ("board_cluster", "board") if c in cols]
    nulls = {c: int(df[c].isna().sum()) for c in critical}
    bad_nulls = {k: v for k, v in nulls.items() if v > 0}
    if bad_nulls:
        warn(f"Null counts in critical columns: {bad_nulls}")
    else:
        ok("No nulls in critical columns")

    # numeric dtype sanity
    num_expect = [c for c in ["stack_bb", "pot_bb", "weight"] if c in df.columns]
    for c in num_expect:
        if not np.issubdtype(df[c].dtype, np.number):
            fail(f"Column {c} must be numeric, got {df[c].dtype}")

    # actor values
    bad_actor = df[~df["actor"].astype(str).isin(["ip", "oop"])]
    if not bad_actor.empty:
        fail(f"Unexpected actor values: {bad_actor['actor'].unique().tolist()}")
    ok("actor values look good (ip/oop)")

    # hero_pos coherence with actor
    pos_set = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}
    for pos_col in ["hero_pos", "ip_pos", "oop_pos"]:
        if pos_col in df.columns:
            bad_pos = df[~df[pos_col].astype(str).isin(pos_set)]
            if not bad_pos.empty:
                fail(f"Unexpected {pos_col} values: {bad_pos[pos_col].unique()[:10].tolist()}")
    if {"hero_pos", "ip_pos", "oop_pos", "actor"}.issubset(df.columns):
        mismatch = df[
            ((df["actor"] == "ip") & (df["hero_pos"] != df["ip_pos"])) |
            ((df["actor"] == "oop") & (df["hero_pos"] != df["oop_pos"]))
        ]
        if not mismatch.empty:
            warn(f"hero_pos not consistent with actor for {len(mismatch)} rows (showing up to 5):")
            print(mismatch[["actor","hero_pos","ip_pos","oop_pos"]].head(5).to_string(index=False))
        else:
            ok("hero_pos matches actor (ip/oop)")

    # street values (numeric 1/2/3 preferred)
    if not df["street"].isin([1, 2, 3]).all():
        uniq = sorted(df["street"].dropna().unique().tolist())
        warn(f"Non-numeric or unexpected street values observed: {uniq} (ensure normalization)")

    # action token formatting
    not_matched = df[~df["action"].astype(str).str.upper().str.fullmatch(ACTION_TOKEN_RE)]
    if not not_matched.empty:
        samp = not_matched["action"].astype(str).str.upper().unique()[:10]
        fail(f"Unmapped/invalid actions found (sample): {list(samp)}")
    ok("action tokens match expected pattern (FOLD/CHECK/CALL/ALLIN/BET_xx/DONK_xx/RAISE_xx)")

    # ---------- probability columns & semantics ----------
    prob_cols = [c for c in df.columns if PROB_COL_RE.fullmatch(str(c))]
    if not prob_cols:
        fail("No action-probability columns found (expected columns like CHECK, BET_33, RAISE_200, ALLIN, etc.)")
    print(f"found {len(prob_cols)} probability columns")

    # probs in [0,1]
    sub = df[prob_cols]
    non_finite = ~np.isfinite(sub.values)
    if non_finite.any():
        r, c = np.where(non_finite)
        fail(f"Non-finite values in probability columns at {len(r)} locations (e.g., row {r[0]}, col {prob_cols[c[0]]})")

    lt0 = (sub.values < -1e-9).any()
    gt1 = (sub.values > 1 + 1e-9).any()
    if lt0 or gt1:
        fail("Probability columns contain values outside [0,1]")

    # row sums ≈ 1 (allow small tolerance; also allow all-zeros row but warn)
    sums = sub.sum(axis=1).to_numpy()
    all_zero = np.isclose(sums, 0.0, atol=1e-12)
    if all_zero.any():
        warn(f"{int(all_zero.sum())} row(s) have all-zero probabilities (will be ignored by most trainers)")

    off = ~np.isclose(sums, 1.0, atol=args.tol) & ~all_zero
    if off.any():
        idxs = np.where(off)[0][:5]
        warn(f"{int(off.sum())} row(s) have prob-sum not ≈ 1 (tol={args.tol}). Examples:")
        print(df.loc[idxs, prob_cols + ["action"]].to_string(index=False))
    else:
        ok("Probability rows sum ≈ 1")

    # argmax(action_probs) == 'action'
    argmax_idx = sub.values.argmax(axis=1)
    argmax_name = np.array(prob_cols)[argmax_idx]
    mismatch = df[argmax_name != df["action"].astype(str).str.upper().values]
    if not mismatch.empty:
        warn(f"'action' != argmax(probabilities) for {len(mismatch)} row(s). Showing up to 5:")
        show_cols = ["action"] + prob_cols[:min(6, len(prob_cols))]
        print(mismatch[show_cols].head(5).to_string(index=False))
    else:
        ok("action column matches argmax of probability vector")

    # bet_size_pct coherence (optional)
    if "bet_size_pct" in df.columns:
        A = df["action"].astype(str).str.upper()
        pct = df["bet_size_pct"]
        # CHECK should generally have NaN bet_size_pct
        bad_check = A.eq("CHECK") & pct.notna()
        if bad_check.any():
            warn(f"{int(bad_check.sum())} CHECK rows have non-null bet_size_pct")

        # For actions with suffix (_NN), check rough agreement with bet_size_pct
        pat = A.str.extract(r".*_(\d+)$")[0]
        mask_num = pat.notna() & pct.notna()
        if mask_num.any():
            # Compare suffix integer to rounded pct
            suffix = pat[mask_num].astype(int)
            pct_rounded = pct[mask_num].round().astype(int)
            disagree = (suffix != pct_rounded)
            if disagree.any():
                warn(f"{int(disagree.sum())} row(s) where action suffix != rounded bet_size_pct (showing up to 5):")
                sample = df.loc[mask_num].loc[disagree].head(5)
                print(sample[["action","bet_size_pct"]].to_string(index=False))

    # weights > 0
    if (df["weight"] <= 0).any():
        fail("Non-positive weights found")
    ok("weights are positive")

    # ---------- quick distributions ----------
    top_counts(df, "actor")
    top_counts(df, "action")
    top_counts(df, "bet_sizing_id")
    top_counts(df, "street")
    top_counts(df, "ctx")

    if "board_cluster" in df.columns:
        bc = df["board_cluster"].dropna()
        if not bc.empty:
            print(f"\nboard_cluster: min={int(bc.min())}, max={int(bc.max())}, unique={bc.nunique()}")

    print("\n🎯 Sanity OK — this part looks usable for postflop policy training.")

if __name__ == "__main__":
    main()