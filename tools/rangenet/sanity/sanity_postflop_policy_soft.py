#!/usr/bin/env python3
import argparse, sys, json, re
import numpy as np
import pandas as pd

PARQUET_ENGINE = "pyarrow"

def load_action_vocab(args, df: pd.DataFrame) -> list[str]:
    # 1) explicit JSON
    if args.action_vocab_json:
        with open(args.action_vocab_json, "r") as fh:
            vocab = json.load(fh)
        return vocab

    # 2) infer from columns (soft labels)
    candidates = [c for c in df.columns
                  if re.match(r"^(FOLD|CHECK|CALL|BET_|DONK_|RAISE_|ALLIN)$", str(c))]
    if not candidates:
        raise RuntimeError("Could not infer ACTION_VOCAB.")
    fixed = [x for x in ["FOLD","CHECK","CALL"] if x in candidates]
    others = sorted([c for c in candidates if c not in fixed])
    return fixed + others

def sanity_postflop(df: pd.DataFrame, action_vocab: list[str], tol: float = 1e-6):
    n = len(df)
    print(f"\n=== POSTFLOP POLICY SANITY ===")
    print(f"rows: {n:,}")

    # ensure all action cols exist
    missing = [a for a in action_vocab if a not in df.columns]
    if missing:
        print(f"❌ Missing action columns: {missing}")
        return 2

    # soft label sums
    probs = df[action_vocab].to_numpy(dtype=np.float64, copy=False)
    sums = probs.sum(axis=1)
    bad = np.flatnonzero(np.abs(sums - 1.0) > tol).size
    print(f"row sums within ±{tol}: {n - bad:,} ok, {bad:,} bad  {'✅' if bad==0 else '⚠️'}")

    # check bounds
    negs = (probs < -1e-12).any(axis=1).sum()
    bigs = (probs > 1.0 + tol).any(axis=1).sum()
    print(f"values <0 rows: {negs}  values >1 rows: {bigs}  {'✅' if (negs==0 and bigs==0) else '⚠️'}")

    # totals by action
    totals = probs.sum(axis=0)
    print("\nTotals by action:")
    for a, t in zip(action_vocab, totals.tolist()):
        print(f"  {a:>10s} : {t:.6f}")
    missing_mass = [a for a, t in zip(action_vocab, totals.tolist()) if t <= 0]
    if missing_mass:
        print(f"⚠️ actions with zero total mass: {missing_mass}")

    # quick peeks
    for key in ("ctx", "street", "actor", "bet_sizing_id"):
        if key in df.columns:
            print(f"\nby {key} (top-5):")
            print(df[key].value_counts().head(5).to_string())

    print("\n✅ Sanity complete.")
    return 0

def main():
    ap = argparse.ArgumentParser("Postflop policy parquet sanity check (soft labels)")
    ap.add_argument("--parquet", required=True, help="Merged parquet path")
    ap.add_argument("--action-vocab-json", type=str, default=None,
                    help="Optional JSON file with explicit action vocab")
    ap.add_argument("--tol", type=float, default=1e-6, help="Tolerance on sum of probs")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, engine=PARQUET_ENGINE)
    vocab = load_action_vocab(args, df)
    sys.exit(sanity_postflop(df, vocab, tol=args.tol))

if __name__ == "__main__":
    main()