#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# Matches your training vocab
ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500",
    "ALLIN",
]

def find_part_files(parts_dir: Path) -> list[Path]:
    pats = ["part-*.parquet", "shard-*-part-*.parquet"]
    files = []
    for pat in pats:
        files.extend(sorted(parts_dir.glob(pat)))
    return files

def load_sample(parts_dir: Path, n: int) -> pd.DataFrame:
    files = find_part_files(parts_dir)
    if not files:
        raise FileNotFoundError(f"No part files found in {parts_dir}")

    rows = []
    for f in files:
        df = pd.read_parquet(f)
        rows.append(df)
        if sum(x.shape[0] for x in rows) >= n:
            break
    if not rows:
        raise RuntimeError("No rows loaded.")
    df = pd.concat(rows, ignore_index=True)
    return df.head(n).reset_index(drop=True)

def entropy_row(p: np.ndarray, eps=1e-12) -> float:
    q = np.clip(p, eps, 1.0)
    q = q / q.sum()
    return float(-(q * np.log2(q)).sum())

def main():
    ap = argparse.ArgumentParser("Sample + sanity-check postflop policy parts")
    ap.add_argument("--parts-dir", type=str, required=True,
                    help="Directory with part-*.parquet outputs")
    ap.add_argument("--sample-n", type=int, default=1000,
                    help="Rows to sample (across multiple parts)")
    args = ap.parse_args()

    parts_dir = Path(args.parts_dir)
    df = load_sample(parts_dir, args.sample_n)

    # Identify action columns present in the parts (robust to partial coverage)
    action_cols = [c for c in df.columns if c in ACTION_VOCAB]
    if not action_cols:
        raise RuntimeError("No ACTION_VOCAB columns found in parts.")
    A = len(action_cols)

    # Basic counts
    total = len(df)
    facing_rate = float(df.get("facing_bet", pd.Series([0]*total)).astype(int).mean())

    # Probability mass checks
    probs = df[action_cols].to_numpy(dtype=float)
    row_sums = probs.sum(axis=1)
    bad_mass = int(np.sum(np.abs(row_sums - 1.0) > 1e-3))
    zero_rows = int(np.sum(row_sums <= 1e-8))

    # Entropy + top1
    ents = np.array([entropy_row(p) for p in probs])
    top_idx = probs.argmax(axis=1)
    top_tokens = [action_cols[i] for i in top_idx]
    top_dist = pd.Series(top_tokens).value_counts(normalize=True).to_dict()

    # Coverage of actions
    coverage = (probs > 1e-6).any(axis=0)
    covered_actions = [c for c, ok in zip(action_cols, coverage) if ok]

    # Context / actor quick looks (if present)
    ctx_counts = (df["ctx"].value_counts().head(10).to_dict() if "ctx" in df.columns else {})
    actor_counts = (df["actor"].value_counts().to_dict() if "actor" in df.columns else {})

    # Print summary
    print("\n--- SAMPLE SUMMARY ---")
    print(f"rows_loaded          : {total}")
    print(f"facing_bet_rate      : {facing_rate:.3f}")
    print(f"row_sums!=1e-3_count : {bad_mass}")
    print(f"zero_mass_rows       : {zero_rows}")
    print(f"mean_entropy(bits)   : {ents.mean():.3f}")
    print(f"median_entropy(bits) : {np.median(ents):.3f}")

    print("\n--- TOP-1 ACTION DISTRIBUTION ---")
    for k, v in sorted(top_dist.items(), key=lambda kv: -kv[1])[:12]:
        print(f"{k:>10}: {v:.3f}")

    print("\n--- ACTION COVERAGE (any mass > 1e-6) ---")
    print(", ".join(covered_actions))

    if ctx_counts:
        print("\n--- CTX COUNTS (top 10) ---")
        for k, v in ctx_counts.items():
            print(f"{k:>15}: {v}")

    if actor_counts:
        print("\n--- ACTOR COUNTS ---")
        for k, v in actor_counts.items():
            print(f"{k:>5}: {v}")

    # Show a couple of example rows (truncated)
    print("\n--- EXAMPLES (2 rows) ---")
    cols_show = ["ctx","ip_pos","oop_pos","facing_bet","action","board","pot_bb","effective_stack_bb"]
    cols_show = [c for c in cols_show if c in df.columns]
    ex = df.loc[[0, min(1, total-1)], cols_show + action_cols].copy()
    # round probs for readability
    for c in action_cols:
        ex[c] = ex[c].map(lambda x: float(f"{x:.3f}"))
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(ex)

    # Quick invariants to fail fast in CI/console
    problems = []
    if facing_rate < 0.05:
        problems.append(f"Low facing_bet rate ({facing_rate:.3f})")
    if bad_mass > 0:
        problems.append(f"{bad_mass} rows with probs not summing ≈ 1")
    if len(covered_actions) < 5:
        problems.append("Very low action coverage (<5 actions)")

    print("\n--- VERDICT ---")
    if problems:
        print("❌ Issues detected:")
        for p in problems:
            print("  -", p)
    else:
        print("✅ Looks sane.")

if __name__ == "__main__":
    main()