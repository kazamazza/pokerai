#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

POS_NAMES = {"UTG","LJ","HJ","CO","BTN","SB","BB","EP","MP","BU"}
OPENING_ACTIONS = {"OPEN","RAISE","ALL_IN","LIMP","CALL"}  # treat any non-fold as opener

def norm_pos(s: str) -> str:
    alias = {"EP":"UTG","MP":"HJ","BU":"BTN"}
    return alias.get(s, s)

def parse_seq(seq_json: str):
    try:
        seq = json.loads(seq_json)
        return seq if isinstance(seq, list) else []
    except Exception:
        return []

def first_non_fold_actor(seq):
    for e in seq:
        a = e.get("action")
        if a and a != "FOLD" and a in OPENING_ACTIONS:
            return e.get("pos"), a
    return None, None

def is_srp_open_call(seq, ip_pos: str, oop_pos: str) -> bool:
    if not seq: return False
    pos0, act0 = seq[0].get("pos"), seq[0].get("action")
    if pos0 != ip_pos or act0 not in ("OPEN","RAISE","ALL_IN","LIMP","CALL"):
        return False
    raised = False
    for step in seq[1:]:
        act = step.get("action")
        pos = step.get("pos")
        if act in ("RAISE","ALL_IN","3BET","4BET","5BET"):
            raised = True
        if pos == oop_pos:
            return (act == "CALL") and (not raised)
    return False

def validate(path: Path, sample: int|None = 5) -> int:
    df = pd.read_parquet(path)
    print(f"Loaded: {path}  shape={df.shape}")

    required = {"stack_bb","hero_pos","sequence","abs_path","sig"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ missing columns: {missing}")
        return 1

    # Normalize hero_pos and basic cats
    df["hero_pos"] = df["hero_pos"].fillna("").map(norm_pos)
    bad_pos = df[~df["hero_pos"].isin(POS_NAMES)].shape[0]
    if bad_pos:
        print(f"⚠️ hero_pos outside known set: {bad_pos}")

    # Parse sequences + opener sanity
    seqs = df["sequence"].astype(str).map(parse_seq)
    opener_pos = []
    opener_act = []
    bad_json = 0
    bad_opener = 0
    disagree = 0
    for i, seq in enumerate(seqs):
        if not seq:
            bad_json += 1; opener_pos.append(None); opener_act.append(None); continue
        p, a = first_non_fold_actor(seq)
        if p is None:
            bad_opener += 1
        opener_pos.append(p); opener_act.append(a)

        # if manifest has columns, cross-check
        if "opener_pos" in df.columns and df.at[i,"opener_pos"] and p and df.at[i,"opener_pos"] != p:
            disagree += 1
        if "opener_action" in df.columns and df.at[i,"opener_action"] and a and df.at[i,"opener_action"] != a:
            disagree += 1

    print(f"JSON parse failures: {bad_json}  |  no opener found: {bad_opener}  |  opener disagreements: {disagree}")

    # Duplicates
    dup_abs = df["abs_path"].duplicated().sum()
    dup_sig = df["sig"].duplicated().sum()
    if dup_abs or dup_sig:
        print(f"⚠️ duplicates → abs_path: {dup_abs}, sig: {dup_sig}")

    # Coverage by (stack, hero)
    cov = (
        df.groupby(["stack_bb","hero_pos"], dropna=False)["sig"]
          .count().rename("files").reset_index()
    )
    print("\nCoverage by (stack_bb, hero_pos):")
    for _, r in cov.sort_values(["stack_bb","hero_pos"]).iterrows():
        print(f"  {int(r['stack_bb']) if pd.notna(r['stack_bb']) else 'NA':>4}bb  {r['hero_pos'] or 'NA':>3}: {int(r['files'])}")

    # Quick SRP pair sniff
    want_pairs = [("BTN","BB"),("CO","BB"),("SB","BB"),("BTN","SB"),("HJ","BB")]
    pairs = defaultdict(int)
    for seq in seqs:
        if not seq: continue
        # list of unique positions in order
        poss = []
        for s in seq:
            p = s.get("pos");
            if p in POS_NAMES and (not poss or poss[-1]!=p):
                poss.append(p)
        for ip, oop in want_pairs:
            if is_srp_open_call(seq, ip, oop):
                pairs[(ip,oop)] += 1

    print("\nSRP OPEN/CALL coverage (want pairs):")
    for p in want_pairs:
        n = pairs.get(p, 0)
        print(f"  {p[0]}v{p[1]}: {n} file(s)")

    # Tiny sample
    if sample:
        print("\nSamples:")
        for _, r in df.sample(min(sample, len(df)), random_state=42).iterrows():
            seq = parse_seq(r["sequence"])[:6]
            print({
                "stack_bb": r["stack_bb"],
                "hero_pos": r["hero_pos"],
                "opener_pos": r.get("opener_pos"),
                "opener_action": r.get("opener_action"),
                "stem": r.get("filename_stem"),
                "seq_head": seq,
            })

    # Verdict
    hard_fail = bool(missing or bad_json > 0)
    if hard_fail:
        print("\n❌ Manifest failed hard checks.")
        return 1
    print("\n✅ Manifest looks structurally sound.")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/artifacts/monker_manifest.parquet"))
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()
    raise SystemExit(validate(args.manifest, sample=args.sample))

if __name__ == "__main__":
    main()