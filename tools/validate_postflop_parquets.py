#!/usr/bin/env python3
import argparse, sys, math
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

ROOT_TOKENS = {
    # legal at root
    "CHECK",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_25","DONK_33","DONK_50","DONK_66","DONK_75","DONK_100",
}
FACING_TOKENS = {
    # legal when facing a bet
    "FOLD","CALL","ALLIN",
    "RAISE_150","RAISE_200","RAISE_250","RAISE_300","RAISE_400","RAISE_500",
}
# expected contexts in the “light” manifest (used as a default; can be relaxed with --no-ctx-expect)
CTX_EXPECT = {
    "VS_OPEN": 8,
    "VS_3BET": 10,
    "VS_4BET": 5,
    "LIMPED_SINGLE": 5,
}

def collist(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]

def nonzero_count(df: pd.DataFrame, cols: List[str], eps: float = 1e-9) -> int:
    if not cols: return 0
    return int((df[cols].sum(axis=1) > eps).sum())

def check_row_sums_unit(df: pd.DataFrame, action_cols: List[str], name: str, tol: float = 1e-5) -> List[str]:
    errs = []
    s = df[action_cols].sum(axis=1).to_numpy(dtype=float)
    bad = np.where(np.abs(s - 1.0) > tol)[0]
    if len(bad) > 0:
        # show up to first 5 examples
        idxs = ", ".join(str(int(i)) for i in bad[:5])
        errs.append(f"[{name}] {len(bad)} rows do not sum to 1 (tol={tol}). Examples idx: {idxs}")
    return errs

def check_illegal_mass(df: pd.DataFrame, legal: set, name: str, eps: float = 1e-9) -> List[str]:
    errs = []
    mass_cols = [c for c in df.columns if c.isupper() and (c not in legal)]
    if not mass_cols: return errs
    nz = df[mass_cols].sum()[(df[mass_cols].sum() > eps)]
    if len(nz) > 0:
        details = "; ".join([f"{k}={float(v):.4g}" for k, v in nz.items()][:8])
        errs.append(f"[{name}] Illegal action mass detected: {details}")
    return errs

def check_presence_by_ctx(df: pd.DataFrame, expect: Dict[str,int], name: str) -> List[str]:
    errs = []
    by = df.groupby(["ctx","ip_pos","oop_pos"]).size().reset_index(name="n")
    # collapse positions; just check counts per ctx
    got = by.groupby("ctx")["n"].sum().to_dict()
    for ctx, n_exp in expect.items():
        n_got = int(got.get(ctx, 0))
        if n_got < n_exp:
            errs.append(f"[{name}] Missing rows for ctx={ctx}: got {n_got}, expected ≥ {n_exp}")
    return errs

def check_facing_specific(df: pd.DataFrame) -> List[str]:
    errs = []
    # faced_size_pct present & non-null
    if "faced_size_pct" not in df.columns:
        errs.append("[FACING] Column faced_size_pct missing")
    else:
        nulls = int(df["faced_size_pct"].isna().sum())
        if nulls > 0:
            errs.append(f"[FACING] faced_size_pct has {nulls} nulls")

        # reasonable bucket set
        if "faced_size_pct" in df.columns:
            buckets = sorted(df["faced_size_pct"].dropna().unique().tolist())
            if not any(b in (25, 33, 50, 66, 75, 100) for b in buckets):
                errs.append(f"[FACING] Unexpected faced_size_pct buckets: {buckets}")

    # at least some raises OR all-in in the set (smoke sets can be sparse; require ≥1)
    raise_cols = [c for c in df.columns if c.startswith("RAISE_")]
    nz_raises = nonzero_count(df, raise_cols)
    any_allin = int((df["ALLIN"] > 1e-9).sum()) if "ALLIN" in df.columns else 0
    if (nz_raises + any_allin) == 0:
        errs.append("[FACING] No RAISE_* or ALLIN mass found in any row")

    # CALL or FOLD presence (should be very common)
    any_call = int((df["CALL"] > 1e-9).sum()) if "CALL" in df.columns else 0
    any_fold = int((df["FOLD"] > 1e-9).sum()) if "FOLD" in df.columns else 0
    if any_call == 0 and any_fold == 0:
        errs.append("[FACING] Neither CALL nor FOLD ever appears")
    return errs

def check_root_specific(df: pd.DataFrame) -> List[str]:
    errs = []
    # forbid CALL/RAISE/ALLIN mass
    forb_cols = ["CALL","ALLIN"] + [c for c in df.columns if c.startswith("RAISE_")]
    if any(c in df.columns for c in forb_cols):
        sub = [c for c in forb_cols if c in df.columns]
        nz = df[sub].sum()[(df[sub].sum() > 1e-9)]
        if len(nz) > 0:
            details = "; ".join([f"{k}={float(v):.4g}" for k,v in nz.items()][:8])
            errs.append(f"[ROOT] Found illegal CALL/RAISE/ALLIN mass: {details}")

    # ensure some CHECKs exist (at least 1 row)
    if "CHECK" in df.columns:
        if int((df["CHECK"] > 1e-9).sum()) == 0:
            errs.append("[ROOT] No CHECK mass found in any row")
    else:
        errs.append("[ROOT] CHECK column missing")
    return errs

def main():
    ap = argparse.ArgumentParser(description="Validate postflop root/facing parquet sanity.")
    ap.add_argument("--root", required=True, help="Path to root parquet")
    ap.add_argument("--facing", required=True, help="Path to facing parquet")
    ap.add_argument("--no-ctx-expect", action="store_true",
                    help="Do not enforce expected ctx counts (useful beyond the light manifest).")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any warning-level condition is met (more stringent).")
    ap.add_argument("--max-print", type=int, default=20, help="Max number of issues to print")
    args = ap.parse_args()

    issues: List[str] = []
    warnings: List[str] = []

    # ---- Load
    try:
        r = pd.read_parquet(args.root)
    except Exception as e:
        print(f"FAIL: Cannot read root parquet: {e}")
        sys.exit(2)
    try:
        f = pd.read_parquet(args.facing)
    except Exception as e:
        print(f"FAIL: Cannot read facing parquet: {e}")
        sys.exit(2)

    # ---- Identify action columns
    root_actions = [c for c in r.columns if c.isupper() and (c in ROOT_TOKENS)]
    facing_actions = [c for c in f.columns if c.isupper() and (c in FACING_TOKENS)]

    if not root_actions:
        issues.append("[ROOT] No recognized action columns found")
    if not facing_actions:
        issues.append("[FACING] No recognized action columns found")

    # ---- Row-sum checks (≈1.0)
    if root_actions:
        issues += check_row_sums_unit(r, root_actions, "ROOT")
    if facing_actions:
        issues += check_row_sums_unit(f, facing_actions, "FACING")

    # ---- Illegal mass checks
    issues += check_illegal_mass(r, legal=ROOT_TOKENS, name="ROOT")
    issues += check_illegal_mass(f, legal=FACING_TOKENS, name="FACING")

    # ---- Context coverage (only for the light manifest unless disabled)
    if not args.no_ctx_expect:
        issues += check_presence_by_ctx(r, CTX_EXPECT, "ROOT")
        issues += check_presence_by_ctx(f, CTX_EXPECT, "FACING")

    # ---- Side-specific assertions
    issues += check_root_specific(r)
    issues += check_facing_specific(f)

    # ---- Informative summaries (warnings)
    try:
        faced_dist = f["faced_size_pct"].value_counts().sort_index().to_dict()
        warnings.append(f"[FACING] faced_size_pct distribution: {faced_dist}")
    except Exception:
        pass

    # Non-critical: show token coverage counts
    def token_cov(df: pd.DataFrame, toks: List[str], label: str):
        if not toks: return
        nz = df[toks].gt(1e-9).sum().sort_index()
        warnings.append(f"[{label}] nonzero counts by token: " + ", ".join([f"{k}={int(v)}" for k,v in nz.items()]))

    token_cov(r, sorted(list(ROOT_TOKENS & set(r.columns))), "ROOT")
    token_cov(f, sorted(list(FACING_TOKENS & set(f.columns))), "FACING")

    # ---- Verdict
    if issues:
        print("❌ FAIL")
        for msg in issues[:args.max_print]:
            print(" -", msg)
        if len(issues) > args.max_print:
            print(f" ... and {len(issues)-args.max_print} more")
        # still print warnings after fails
        if warnings:
            print("\nNotes:")
            for w in warnings[:args.max_print]:
                print(" •", w)
        sys.exit(1)
    else:
        print("✅ PASS")
        for w in warnings[:args.max_print]:
            print(" •", w)
        sys.exit(0)

if __name__ == "__main__":
    main()