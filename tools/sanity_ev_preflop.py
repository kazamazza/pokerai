#!/usr/bin/env python3
# tools/sanity_ev_preflop.py
import argparse, sys, math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

REQ_X = ["hero_pos","villain_pos","stakes_id","facing_flag","free_check","hand_id"]
REQ_CONT = ["stack_bb","pot_bb","faced_frac"]

def _find_ev_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("ev_")])

def _share(series) -> Dict:
    total = float(series.shape[0]) or 1.0
    c = series.value_counts(dropna=False)
    return {str(k): round(float(v)/total, 3) for k, v in c.items()}

def main():
    ap = argparse.ArgumentParser(description="Preflop EV parquet sanity checks")
    ap.add_argument("--parquet", required=True, help="Path to preflop EV parquet")
    ap.add_argument("--stacks", type=float, nargs="*", default=[25,40,60,80,100,150])
    ap.add_argument("--open-bbs", type=float, nargs="*", default=[2.0,2.5,3.0,3.5],
                    help="Expected absolute open sizes (bb) for facing branch")
    ap.add_argument("--tol-ev", type=float, default=1e-3, help="Tolerance for EV=0 anchors")
    ap.add_argument("--max_oob_frac", type=float, default=1e-3, help="Max fraction allowed out-of-bounds EVs")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    n = len(df)
    print(f"=== Checking PREFLOP ===")
    print(f"rows={n:,}")

    # ---------- schema ----------
    missing = [c for c in REQ_X if c not in df.columns] + [c for c in REQ_CONT if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        sys.exit(1)
    ev_cols = _find_ev_cols(df)
    if not ev_cols:
        print("❌ No ev_* columns found")
        sys.exit(1)

    # ---------- NaN / inf ----------
    bad = []
    for c in REQ_X + REQ_CONT + ev_cols:
        if df[c].isna().any():
            bad.append(c)
    if bad:
        print(f"❌ NaNs present in columns: {bad}")
        sys.exit(1)
    # ---------- coverage ----------
    face_share = _share(df["facing_flag"])
    free_share = _share(df["free_check"])
    hero_share = _share(df["hero_pos"])
    print(f"facing_flag share={face_share}")
    print(f"free_check   share={free_share}")
    print(f"hero_pos     share={hero_share}")

    if not (("0" in face_share and "1" in face_share)):
        print("❌ Missing unopened or facing coverage (facing_flag must have 0 and 1)."); sys.exit(1)
    if not (("0" in free_share and "1" in free_share)):
        print("❌ Missing free_check coverage (need rows with 0 and 1)."); sys.exit(1)
    if "BB" not in hero_share:
        print("❌ 'BB' hero_pos not present (needed for free-check rows)."); sys.exit(1)

    # ---------- logical constraints ----------
    unopened = df[df["facing_flag"] == 0]
    facing   = df[df["facing_flag"] == 1]

    # free_check must be 1 only when unopened and hero_pos == "BB"
    invalid_free = df[
        ((df["free_check"] == 1) & (df["facing_flag"] != 0)) |
        ((df["free_check"] == 1) & (df["hero_pos"].astype(str).str.upper() != "BB"))
    ]
    if len(invalid_free) > 0:
        print(f"❌ free_check rows invalid (expect unopened & hero_pos=BB): {len(invalid_free)}"); sys.exit(1)

    # unopened faced_frac must be 0
    if not np.allclose(unopened["faced_frac"].values, 0.0):
        bad_cnt = int((unopened["faced_frac"].abs() > 1e-9).sum())
        print(f"❌ unopened rows with non-zero faced_frac: {bad_cnt}"); sys.exit(1)

    # facing faced_frac should be in (0, 1]; check rough range
    if not ((facing["faced_frac"] > 0).all() and (facing["faced_frac"] <= 1.0 + 1e-9).all()):
        print("❌ facing rows have faced_frac outside (0,1]"); sys.exit(1)

    # Optional: check open sizes align (faced_frac ≈ open_bb/stack)
    # build expected set {open_bb/stack for stacks, opens}
    expected = set()
    for s in args.stacks:
        for ob in args.open_bbs:
            expected.add(round(float(ob)/float(s), 4))
    fac_vals = set(np.round(facing["faced_frac"].values.astype(float), 4).tolist())
    miss_expected = sorted([x for x in expected if x not in fac_vals])
    if miss_expected:
        # warn only (some stacks may be absent depending on cfg)
        print(f"⚠️  Some expected faced_frac values absent (stack/open combos missing?): {miss_expected[:8]}{'...' if len(miss_expected)>8 else ''}")

    # ---------- EV anchors ----------
    def _mean_abs(col, mask):
        if col not in df.columns: return None
        return float(df.loc[mask, col].abs().mean()) if mask.any() else 0.0

    fold_abs = _mean_abs("ev_FOLD", df.index == df.index)  # all rows
    check_abs = _mean_abs("ev_CHECK", (df["free_check"] == 1))
    if fold_abs is not None and fold_abs > args.tol_ev:
        print(f"❌ ev_FOLD mean |value|={fold_abs:.4g} > tol ({args.tol_ev})"); sys.exit(1)
    if check_abs is not None and check_abs > args.tol_ev:
        print(f"❌ ev_CHECK (free_check==1) mean |value|={check_abs:.4g} > tol ({args.tol_ev})"); sys.exit(1)

    # ---------- EV numeric sanity ----------
    # Loose bound: EV ∈ [ -stack_bb , pot_bb + 2*stack_bb ]
    stk = df["stack_bb"].astype(float).to_numpy()
    pot = df["pot_bb"].astype(float).to_numpy()
    upper = pot + 2.0 * stk
    lower = -stk
    oob_total = 0
    for c in ev_cols:
        ev = df[c].astype(float).to_numpy()
        oob = ((ev < lower - 1e-6) | (ev > upper + 1e-6)).sum()
        if oob:
            frac = oob / float(n)
            oob_total += oob
            print(f"⚠️  {c}: {oob} / {n} ({frac:.5f}) outside loose bounds")
    if oob_total / float(n) > args.max_oob_frac:
        print("❌ Too many out-of-bounds EVs overall; investigate generator/units.")
        sys.exit(1)

    # ---------- distribution summaries ----------
    print("Summaries:")
    print(f"  unopened rows    = {len(unopened):,}")
    print(f"  facing rows      = {len(facing):,}")
    if "CALL" in [c[3:] for c in ev_cols]:
        print(f"  has CALL column  = yes")
    else:
        print(f"  has CALL column  = no (check your vocab)")

    # group balance hint: samples per (hero,villain,stack,facing_flag,free_check,faced_frac)
    key_cols = ["hero_pos","villain_pos","stack_bb","facing_flag","free_check","faced_frac"]
    grp = df.groupby(key_cols).size()
    if not grp.empty:
        mean_ct, std_ct = grp.mean(), grp.std(ddof=0)
        cv = float(std_ct / mean_ct) if mean_ct > 0 else 0.0
        print(f"  combo count mean={mean_ct:.1f}, std={std_ct:.1f}, CV={cv:.3f}")

    print("✅ Preflop parquet looks OK.")
    # gentle reminder of shares
    print(f"Context (unopened vs facing): facing_flag={face_share}, free_check={free_share}")

if __name__ == "__main__":
    main()