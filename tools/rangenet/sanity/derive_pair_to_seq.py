#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
from collections import Counter, defaultdict

POS_NAMES = {"UTG","LJ","HJ","CO","BTN","SB","BB","EP","MP","BU"}

def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "sequence" not in df.columns:
        raise SystemExit("manifest missing 'sequence' column")
    # sequence is JSON string as produced by your builder
    return df

def parse_seq(seq_json: str):
    try:
        seq = json.loads(seq_json)
        # expected: [{"pos":"UTG","action":"OPEN"}, {"pos":"HJ","action":"FOLD"}, ...]
        return seq if isinstance(seq, list) else []
    except Exception:
        return []

def is_srp_open_call(seq, ip_pos: str, oop_pos: str) -> bool:
    """
    Single-raised pot we want: ip_pos opens/raises, and the first action by oop_pos is CALL,
    with no intervening re-raise before oop acts.
    """
    if not seq:
        return False

    # opener must be ip_pos and first action must be OPEN/RAISE
    if not (seq[0].get("pos") == ip_pos and seq[0].get("action") in ("OPEN", "RAISE")):
        return False

    # ensure oop's first action is CALL and no re-raise occurs before that
    re_raised = False
    for step in seq[1:]:
        pos = step.get("pos")
        act = step.get("action")
        if act in ("RAISE", "ALL_IN", "3BET", "4BET", "5BET"):
            re_raised = True
        if pos == oop_pos:
            return (act == "CALL") and (not re_raised)
    return False

def stem_for_pair(seq, ip_pos: str, oop_pos: str) -> str | None:
    """
    Produce a stable stem *for this pair*, e.g. "BTN_Open_BB_Call" or "UTG_Raise_CO_Call".
    """
    if not seq:
        return None

    ip_act = None
    oop_act = None

    # opener must be first and be ip_pos
    if seq[0].get("pos") == ip_pos and seq[0].get("action") in ("OPEN", "RAISE"):
        ip_act = seq[0].get("action")

    # find first action by oop_pos
    for step in seq[1:]:
        if step.get("pos") == oop_pos:
            oop_act = step.get("action")
            break

    if ip_act and oop_act:
        return f"{ip_pos}_{ip_act.title()}_{oop_pos}_{oop_act.title()}"
    return None


def stem_from_seq(seq):
    # Recreate filename-style token prefix e.g. "BTN_Open_BB_Call" (first actions of the key actors)
    # We’ll keep it short/explicit for stability.
    if not seq:
        return None
    toks = []
    for step in seq:
        p = step.get("pos")
        a = step.get("action")
        if p in POS_NAMES and isinstance(a, str):
            toks.append(f"{p}_{a.title()}")
        # stop once both IP OPEN and OOP CALL have appeared, to keep stem minimal
    return "_".join(toks[:2])  # "BTN_Open" then "BB_Call" expected

def derive_pair_to_seq(manifest_path: str | Path):
    df = load_manifest(Path(manifest_path))
    # We’ll try all ordered pairs (ip, oop) present in the manifest
    # by looking at sequences to detect SRP patterns.
    pairs = defaultdict(Counter)  # (ip,oop) -> Counter(stem -> count)

    for _, r in df.iterrows():
        seq = parse_seq(r["sequence"])
        if len(seq) < 2:
            continue
        poss = [s.get("pos") for s in seq if s.get("pos") in POS_NAMES]
        uniq = list(dict.fromkeys(poss))  # preserve order
        for i_pos in uniq:
            for o_pos in uniq:
                if i_pos == o_pos:
                    continue
                if is_srp_open_call(seq, i_pos, o_pos):
                    stem = stem_for_pair(seq, i_pos, o_pos)
                    if stem:
                        pairs[(i_pos, o_pos)][stem] += 1

    # Choose majority stem per pair
    final_map = {}
    conflicts = {}
    for (ip, oop), ctr in pairs.items():
        if not ctr:
            continue
        stem, cnt = ctr.most_common(1)[0]
        final_map[(ip, oop)] = stem
        if len(ctr) > 1:
            conflicts[(ip, oop)] = dict(ctr.most_common(5))

    # Pretty print
    print("✅ Derived PAIR_TO_SEQ (SRP OPEN/CALL):")
    for (ip, oop), stem in sorted(final_map.items()):
        print(f'  ("{ip}","{oop}"): "{stem}"')

    if conflicts:
        print("\n⚠️  Conflicts (multiple candidate stems):")
        for k, ctr in conflicts.items():
            print(f"  {k}: {ctr}")

    # Show uncovered pairs that you might want
    want_pairs = [("BTN","BB"), ("CO","BB"), ("SB","BB"), ("BTN","SB"), ("HJ","BB")]
    missing = [p for p in want_pairs if p not in final_map]
    if missing:
        print("\n⚠️  Missing desired pairs:", ", ".join(f"{a}v{b}" for a,b in missing))

    return final_map

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/artifacts/monker_manifest.parquet")
    args = ap.parse_args()
    derive_pair_to_seq(args.manifest)

if __name__ == "__main__":
    main()