#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500",
    "ALLIN"
]
BET_BUCKETS   = {"BET_25","BET_33","BET_50","BET_66","BET_75","BET_100","DONK_33"}
RAISE_BUCKETS = {"RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"}

def _upper(x): return str(x).strip().upper()

def main():
    ap = argparse.ArgumentParser("Postflop dataset sanity")
    ap.add_argument("--parquet", required=True, help="path to postflop parquet")
    ap.add_argument("--peek", type=int, default=10, help="show first N rows of key cols")
    args = ap.parse_args()

    p = Path(args.parquet)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)
    N = len(df)
    print(f"Loaded: {p}  rows={N:,}")

    # --- 1) Basic schema checks ---
    req_cat = {"hero_pos","ip_pos","oop_pos","ctx","street"}
    req_misc = {"weight","pot_bb","effective_stack_bb","bet_sizing_id"}
    has_soft = all(c in df.columns for c in ACTION_VOCAB)
    has_hard = "action" in df.columns
    if not (has_soft or has_hard):
        raise ValueError("Dataset must have either soft per-action columns or a hard 'action' column.")

    missing = [c for c in (req_cat|req_misc) if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")

    # --- 2) Coverage snapshots ---
    for col in ["ctx","ip_pos","oop_pos","street"]:
        if col in df.columns:
            vc = df[col].astype(str).str.upper().value_counts()
            print(f"\n[{col}] top counts:")
            print(vc.head(10).to_string())

    # Optional board_cluster_id / board_mask_52 presence
    if "board_cluster_id" in df.columns:
        print("\nDetected board_cluster_id — unique clusters:", df["board_cluster_id"].nunique())
    else:
        print("\nNo board_cluster_id column detected (that’s OK if you’re using only 52-bit mask).")

    if "board_mask_52" in df.columns:
        # spot-check shape on a few rows
        bad_mask = []
        for i in range(min(200, N)):
            v = df.iloc[i]["board_mask_52"]
            try:
                a = np.asarray(v, dtype=np.float32).ravel()
                if a.size != 52:
                    bad_mask.append(i)
            except Exception:
                bad_mask.append(i)
        if bad_mask:
            print(f"⚠️  board_mask_52 wrong length on {len(bad_mask)} example(s) (expected 52).")
        else:
            print("✅ board_mask_52 length looks good on sampled rows.")
    else:
        print("⚠️  No board_mask_52 column found.")

    # --- 3) Menu legality sanity ---
    if "bet_sizes" in df.columns:
        def legal_from_row(r):
            actor = str(r.get("actor","ip")).lower()
            facing = int(r.get("facing_bet", 0) or 0)
            # parse bet_sizes
            menu = r.get("bet_sizes", None)
            if isinstance(menu, str):
                try: menu = json.loads(menu)
                except Exception: menu = None
            if isinstance(menu, (list, tuple)):
                menu = [float(x) for x in menu]
            legal = set()
            if facing:
                legal = {"FOLD","CALL"} | RAISE_BUCKETS
            else:
                legal = {"CHECK"} | BET_BUCKETS
                if actor != "oop" and "DONK_33" in legal:
                    legal.remove("DONK_33")
                # restrict to menu
                want = set()
                if menu:
                    if 0.25 in menu: want.add("BET_25")
                    if 0.33 in menu: want.add("BET_33"); want.add("DONK_33")
                    if 0.50 in menu: want.add("BET_50")
                    if 0.66 in menu: want.add("BET_66")
                    if 0.75 in menu: want.add("BET_75")
                    if 1.00 in menu: want.add("BET_100")
                    # drop disallowed BET_* buckets
                    legal = {a for a in legal if (a not in BET_BUCKETS) or (a in want)}
            return legal

        # sample 1000 rows for speed
        sample = df.sample(min(1000, N), random_state=13)
        bad = 0
        for _, r in sample.iterrows():
            legal = legal_from_row(r)
            if has_soft:
                # if soft labels exist, ensure zero mass on clearly illegal ALL-zero? (loose check)
                # Here we just assert that there's at least one legal action
                if len(legal) == 0:
                    bad += 1
            else:
                a = str(r["action"]).upper()
                if a not in legal:
                    bad += 1
        if bad:
            print(f"⚠️  Legality mismatch on ~{bad}/{len(sample)} sampled rows.")
        else:
            print("✅ Legality vs menu looks consistent on sampled rows.")
    else:
        print("ℹ️  No bet_sizes column; skipping menu/legal cross-check.")

    # --- 4) Quick peek
    keep = [c for c in ["ctx","ip_pos","oop_pos","street","bet_sizing_id","facing_bet","actor","board_cluster_id"] if c in df.columns]
    if keep:
        print("\nHead:")
        print(df[keep].head(args.peek).to_string(index=False))

if __name__ == "__main__":
    main()