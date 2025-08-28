#!/usr/bin/env python3
import argparse, json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd

# --- basic position normalization ---
POS_ALIASES = {
    "BU": "BTN",
    "EP": "UTG",
    "MP": "HJ",
}
CANON_POS: Set[str] = {"UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"}

def norm_pos(p: Optional[str]) -> Optional[str]:
    if not isinstance(p, str):
        return None
    p = p.strip().upper()
    p = POS_ALIASES.get(p, p)
    return p if p in CANON_POS else p

# --- sequence utilities ---
def parse_seq(seq_json: str) -> List[Dict[str, str]]:
    try:
        seq = json.loads(seq_json)
        return seq if isinstance(seq, list) else []
    except Exception:
        return []

def is_srp_open_call(seq: List[Dict[str, str]], ip_pos: str, oop_pos: str) -> bool:
    """
    Single-raised pot pattern:
      - Find the first aggressive open (OPEN/RAISE/ALL_IN) anywhere in the sequence.
      - It must be made by ip_pos.
      - Before oop_pos acts, there must be no re-raise (RAISE/ALL_IN/3BET/...).
      - The first action by oop_pos after the open must be CALL.
    """
    if not seq:
        return False

    OPENERS = {"OPEN", "RAISE", "ALL_IN"}  # vendor-normalized actions
    RERAISE = {"RAISE", "ALL_IN", "3BET", "4BET", "5BET"}

    # 1) locate first open/raise
    first_open_idx = None
    for i, step in enumerate(seq):
        act = step.get("action")
        if act in OPENERS:
            first_open_idx = i
            break
    if first_open_idx is None:
        return False

    opener_pos = seq[first_open_idx].get("pos")
    if opener_pos != ip_pos:
        return False

    # 2) walk forward until oop_pos acts; reject if any re-raise occurs before oop acts
    for j in range(first_open_idx + 1, len(seq)):
        pos = seq[j].get("pos")
        act = seq[j].get("action")
        if act in RERAISE:
            return False
        if pos == oop_pos:
            return act == "CALL"

    return False  # oop never acted


def stem_for_pair(seq: List[Dict[str, str]], ip_pos: str, oop_pos: str) -> Optional[str]:
    """
    Build a readable stem like 'BTN_Raise_BB_Call' using
    the first open/raise and the first action by oop after it.
    """
    if not seq:
        return None

    OPENERS = {"OPEN", "RAISE", "ALL_IN"}

    # find first open/raise
    first_open_idx = None
    for i, step in enumerate(seq):
        if step.get("action") in OPENERS:
            first_open_idx = i
            break
    if first_open_idx is None:
        return None

    ip_act = None
    if seq[first_open_idx].get("pos") == ip_pos:
        ip_act = seq[first_open_idx].get("action")

    oop_act = None
    for j in range(first_open_idx + 1, len(seq)):
        if seq[j].get("pos") == oop_pos and seq[j].get("action"):
            oop_act = seq[j]["action"]
            break

    if ip_act and oop_act:
        return f"{ip_pos}_{ip_act.title()}_{oop_pos}_{oop_act.title()}"
    return None

def derive_pair_to_stem(df: pd.DataFrame) -> Tuple[Dict[Tuple[str,str], str], Dict[Tuple[str,str], Dict[str,int]]]:
    """
    Majority stem per (ip,oop) among rows that match SRP open/call.
    Returns (final_map, conflicts)
    """
    pairs = defaultdict(Counter)  # (ip,oop) -> Counter(stem -> count)
    for _, r in df.iterrows():
        seq = parse_seq(r["sequence"])
        # collect unique positions present in sequence in order
        poss = [norm_pos(s.get("pos")) for s in seq if norm_pos(s.get("pos")) in CANON_POS]
        if not poss:
            continue
        uniq = list(dict.fromkeys(poss))
        for ip in uniq:
            for oop in uniq:
                if ip == oop:
                    continue
                if is_srp_open_call(seq, ip, oop):
                    stem = stem_for_pair(seq, ip, oop)
                    if stem:
                        pairs[(ip, oop)][stem] += 1

    final_map = {}
    conflicts = {}
    for key, ctr in pairs.items():
        if not ctr:
            continue
        stem, _ = ctr.most_common(1)[0]
        final_map[key] = stem
        if len(ctr) > 1:
            conflicts[key] = dict(ctr.most_common(5))
    return final_map, conflicts

def coverage_table(df: pd.DataFrame, stacks: List[int], pairs: List[Tuple[str,str]], stem_map: Dict[Tuple[str,str], str]) -> pd.DataFrame:
    """
    For each (stack, pair), mark ip_ok and oop_ok based on the presence of a matching SRP-open/call file
    under hero_pos == ip (for IP) and hero_pos == oop (for OOP).
    """
    rows = []
    # pre-parse sequences to speed up
    parsed = [parse_seq(s) for s in df["sequence"]]
    df = df.copy()
    df["_seq"] = parsed
    df["hero_pos"] = df["hero_pos"].apply(norm_pos)
    df["stack_bb"] = df["stack_bb"].astype("Int64")  # may have nulls

    for stack in stacks:
        df_s = df[df["stack_bb"] == stack]
        for (ip, oop) in pairs:
            stem = stem_map.get((ip,oop))
            ip_ok = False
            oop_ok = False
            if stem:
                # IP availability: any row at this stack with hero_pos == ip and SRP open/call for (ip,oop)
                mask_ip = (df_s["hero_pos"] == ip)
                for _, r in df_s[mask_ip].iterrows():
                    if is_srp_open_call(r["_seq"], ip, oop):
                        ip_ok = True
                        break
                # OOP availability: hero_pos == oop and SRP open/call for (ip,oop)
                mask_oop = (df_s["hero_pos"] == oop)
                for _, r in df_s[mask_oop].iterrows():
                    if is_srp_open_call(r["_seq"], ip, oop):
                        oop_ok = True
                        break
            rows.append({
                "stack_bb": stack,
                "pair": f"{ip}v{oop}",
                "ip_ok": ip_ok,
                "oop_ok": oop_ok,
                "both_ok": ip_ok and oop_ok,
                "stem": stem or "(none)",
            })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Preflop coverage report from monker_manifest.parquet")
    ap.add_argument("--manifest", type=str, default="data/artifacts/monker_manifest.parquet")
    ap.add_argument("--stacks", type=int, nargs="*", default=[12,15,18])
    ap.add_argument("--pairs", type=str, nargs="*", default=["BTNvBB","COvBB","SBvBB","BTNvSB"])
    ap.add_argument("--show-conflicts", action="store_true")
    args = ap.parse_args()

    path = Path(args.manifest)
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")

    df = pd.read_parquet(path)

    # Normalize positions up front
    for col in ("hero_pos","opener_pos"):
        if col in df.columns:
            df[col] = df[col].apply(norm_pos)

    # Derive majority stems for SRP open/call
    stem_map, conflicts = derive_pair_to_stem(df)

    # Parse pairs input
    wanted_pairs: List[Tuple[str,str]] = []
    for p in args.pairs:
        if "v" not in p:
            continue
        a, b = p.split("v", 1)
        a, b = norm_pos(a), norm_pos(b)
        if a in CANON_POS and b in CANON_POS and a != b:
            wanted_pairs.append((a,b))

    # Build coverage
    cov = coverage_table(df, stacks=args.stacks, pairs=wanted_pairs, stem_map=stem_map)

    # Print summary
    print(f"Loaded manifest: {path}  rows={len(df)}")
    print("\nStem map (majority SRP OPEN/CALL) for requested pairs:")
    for (ip,oop) in wanted_pairs:
        s = stem_map.get((ip,oop))
        print(f"  {ip}v{oop}: {s or '(none)'}")

    if args.show_conflicts and conflicts:
        print("\n⚠️  Conflicts (multiple stems seen). Majority chosen; inspect if unexpected:")
        for k, ctr in conflicts.items():
            print(f"  {k}: {ctr}")

    # Coverage grid
    print("\nCoverage:")
    piv = cov.pivot(index="pair", columns="stack_bb", values="both_ok").fillna(False)
    print(piv.replace({True:"✔", False:"✘"}).to_string())

    # Totals
    total = len(cov)
    both = int(cov["both_ok"].sum())
    ip_only = int((cov["ip_ok"] & ~cov["oop_ok"]).sum())
    oop_only = int((~cov["ip_ok"] & cov["oop_ok"]).sum())
    none = total - both - ip_only - oop_only
    print(f"\nTotals: pairs×stacks={total}  both_ok={both}  ip_only={ip_only}  oop_only={oop_only}  none={none}")
    print(f"Overall both-sides coverage: {both/total:.1%}")

if __name__ == "__main__":
    main()