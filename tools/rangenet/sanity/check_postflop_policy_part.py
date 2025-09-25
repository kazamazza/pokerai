#!/usr/bin/env python3
import sys, numpy as np, pandas as pd

ACTION_COLS_ALL = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300","ALLIN",
]

BET_COLS   = ["BET_25","BET_33","BET_50","BET_66","BET_75","BET_100"]
RAISE_COLS = ["RAISE_150","RAISE_200","RAISE_300","ALLIN"]
DONK_COLS  = ["DONK_33"]  # extend if you add more donk sizes

REQ_COLS_META = [
    "hero_pos","ip_pos","oop_pos","ctx","street","actor","pot_bb","stack_bb","action"
]

EPS = 1e-9

def main():
    if len(sys.argv) != 2:
        print("usage: check_postflop_policy_part.py <parquet>")
        sys.exit(2)

    path = sys.argv[1]
    df = pd.read_parquet(path)
    n = len(df)
    print(f"file={path}  rows={n}")

    # ---- required cols ----
    missing = [c for c in REQ_COLS_META if c not in df.columns]
    if missing:
        print(f"❌ missing required metadata columns: {missing}")
        sys.exit(1)
    print("✅ required columns present")

    # ---- action columns presence ----
    present_actions = [c for c in ACTION_COLS_ALL if c in df.columns]
    print(f"✅ action columns: {present_actions}")
    if not present_actions:
        print("❌ no ACTION_VOCAB columns found")
        sys.exit(1)

    # ---- non-neg + row-sum≈1 checks on action cols ----
    A = df[present_actions].to_numpy(dtype=np.float64, copy=False)
    if (A < -1e-12).any():
        print("❌ negative probs detected")
    else:
        print("non-neg check: OK")

    sums = A.sum(axis=1)
    off = np.flatnonzero(np.abs(sums - 1.0) > 1e-6).size
    print(f"sum≈1 check: {off} rows off by > 1e-06")

    # ---- basic distributions ----
    def _top(col):
        if col in df.columns:
            vc = df[col].value_counts().head(5)
            print(f"\nby {col} (top):")
            for k, v in vc.items():
                print(f"  {k}: {v:,}")

    for k in ("ctx","street","bet_sizing_id","actor","action"):
        _top(k)

    # ---- specialized action diagnostics ----
    def _nonzero_counts(cols, name):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            print(f"\n{name}: (no columns present)")
            return {}
        nz = {c: int((df[c] > EPS).sum()) for c in cols}
        total_rows_with_any = int(((df[cols].sum(axis=1)) > EPS).sum())
        print(f"\n{name}:")
        for c in cols:
            print(f"  {c:10s} -> {nz[c]}")
        print(f"  rows with ANY {name.lower()} mass: {total_rows_with_any}/{n} "
              f"({total_rows_with_any/n:.1%})")
        return {"per_col": nz, "any_rows": total_rows_with_any}

    bet_stats   = _nonzero_counts(BET_COLS,   "BET BUCKETS")
    raise_stats = _nonzero_counts(RAISE_COLS, "RAISE BUCKETS")
    donk_stats  = _nonzero_counts(DONK_COLS,  "DONK BUCKETS")

    # ---- donk used only OOP? (if DONK columns exist) ----
    if any(c in df.columns for c in DONK_COLS):
        donk_sum = df[[c for c in DONK_COLS if c in df.columns]].sum(axis=1)
        has_donk = donk_sum > EPS
        if "actor" in df.columns:
            oop_with_donk = int(((df["actor"] == "oop") & has_donk).sum())
            ip_with_donk  = int(((df["actor"] == "ip")  & has_donk).sum())
            print(f"\nDONK usage by actor: OOP={oop_with_donk}, IP={ip_with_donk} "
                  f"(expect IP≈0)")
        else:
            print("\nDONK usage by actor: cannot check (no actor column)")

    # ---- small previews ----
    def _preview_rows_with_any(cols, title, limit=8):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return
        mask = (df[cols].sum(axis=1) > EPS)
        sub = df.loc[mask, REQ_COLS_META + cols].head(limit)
        if not sub.empty:
            print(f"\n=== preview rows with ANY {title} mass (up to {limit}) ===")
            print(sub.to_string(index=False))

    _preview_rows_with_any(RAISE_COLS, "RAISE")
    _preview_rows_with_any(DONK_COLS,  "DONK")
    _preview_rows_with_any(BET_COLS,   "BET")

    print("\n✅ sanity complete")

if __name__ == "__main__":
    main()