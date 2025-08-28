#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict

import pandas as pd

# If you placed helpers in a module, you can import; otherwise keep local POS set
POS_SET = {"UTG","HJ","CO","BTN","SB","BB"}

REQUIRED_COLS = {
    "stack_bb", "hero_pos", "sequence",
    "abs_path", "file_sha1", "sig"
}
OPTIONAL_COLS = {"opener_pos","opener_action","n_files","filename_stem","rel_path"}

def _ok_path(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False

def _parse_seq(s: str) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []

def validate_manifest(path: Path, sample: int | None = None) -> int:
    df = pd.read_parquet(path)
    print(f"Loaded manifest: {path}  rows={len(df)}  cols={len(df.columns)}")

    # ——— Schema
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return 1
    extra = sorted(set(df.columns) - REQUIRED_COLS - OPTIONAL_COLS)
    if extra:
        print(f"ℹ️ Extra columns present: {extra}")

    # ——— Basic null/NA checks
    bad_nulls = []
    for c in ["stack_bb","hero_pos","sequence","abs_path"]:
        if df[c].isna().any():
            bad_nulls.append(c)
    if bad_nulls:
        print(f"❌ Nulls found in required columns: {bad_nulls}")
        return 1

    # ——— File existence
    bad_paths = df[~df["abs_path"].map(_ok_path)]
    if not bad_paths.empty:
        print(f"❌ {len(bad_paths)} abs_path entries do not exist (first 5 shown):")
        print(bad_paths[["abs_path"]].head(5).to_string(index=False))
        return 1

    # ——— Parse sequences and sanity tokens
    seqs = df["sequence"].map(_parse_seq)
    if (seqs.map(len) == 0).any():
        n_bad = int((seqs.map(len) == 0).sum())
        print(f"❌ {n_bad} rows have unparsable/empty sequence JSON")
        return 1

    # quick token checks
    bad_pos_token_rows = 0
    action_counter = Counter()
    opener_counter = Counter()
    for seq in seqs:
        # every entry should be dict with pos and maybe action
        for e in seq:
            pos = e.get("pos")
            act = e.get("action")
            if pos not in POS_SET:
                bad_pos_token_rows += 1
                break
            if act:
                action_counter[act] += 1
        # opener (first non-Fold if present)
        opener = next(( (e.get("pos"), e.get("action"))
                        for e in seq
                        if e.get("pos") in POS_SET and e.get("action") and e.get("action") != "FOLD"), None)
        if opener:
            opener_counter[opener] += 1

    if bad_pos_token_rows:
        print(f"❌ {bad_pos_token_rows} rows contain non-canonical position tokens in sequence")
        return 1

    # ——— Duplicates by (stack, hero, sequence)
    grp_keys = ["stack_bb","hero_pos","sequence"]
    dup_counts = df.groupby(grp_keys).size().reset_index(name="n").query("n > 1")
    if not dup_counts.empty:
        n_dups = int(dup_counts["n"].sum() - len(dup_counts))
        print(f"⚠️ Duplicate rows by (stack_bb,hero_pos,sequence): groups={len(dup_counts)} "
              f"extra_rows={n_dups} (showing first 3)")
        print(dup_counts.head(3).to_string(index=False))

    # ——— Coverage summaries
    print("\nCoverage by stack:")
    cov_stack = df.groupby("stack_bb").size().reset_index(name="rows")
    print(cov_stack.to_string(index=False))

    print("\nCoverage by hero_pos:")
    cov_hero = df.groupby("hero_pos").size().reset_index(name="rows").sort_values("hero_pos")
    print(cov_hero.to_string(index=False))

    # opener matrix (if columns exist)
    if "opener_pos" in df.columns and "opener_action" in df.columns:
        print("\nOpener counts (pos, action) top-10:")
        top = df.groupby(["opener_pos","opener_action"]).size().reset_index(name="n")\
                .sort_values("n", ascending=False).head(10)
        print(top.to_string(index=False))
    else:
        # fallback to what we computed from sequences
        print("\nOpener counts (derived from sequence) top-10:")
        for (pos,act), cnt in opener_counter.most_common(10):
            print(f"  {pos:>3} {act:<10} x{cnt}")

    print("\nTop actions (from sequences) top-10:")
    for act, cnt in action_counter.most_common(10):
        print(f"  {act:<10} x{cnt}")

    # ——— Optional sampling
    if sample:
        sample = max(1, min(sample, len(df)))
        print(f"\nSamples ({sample}):")
        for _, r in df.sample(sample, random_state=42).iterrows():
            one = {
                "stack_bb": r["stack_bb"],
                "hero_pos": r["hero_pos"],
                "opener_pos": r.get("opener_pos"),
                "opener_action": r.get("opener_action"),
                "stem": r.get("filename_stem"),
                "rel_path": r.get("rel_path"),
            }
            print("  -", one)

    print("\n✅ Manifest looks structurally sound.")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Sanity-check Monker manifest parquet")
    ap.add_argument("--manifest", type=str, default="data/artifacts/monker_manifest.parquet")
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()
    rc = validate_manifest(Path(args.manifest), sample=args.sample)
    raise SystemExit(rc)

if __name__ == "__main__":
    main()