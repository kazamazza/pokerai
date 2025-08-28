#!/usr/bin/env python3
from pathlib import Path
import json, re, sys
from collections import Counter, defaultdict

POS = {"UTG","LJ","HJ","CO","BTN","SB","BB","EP","MP","BU"}
ALIAS = {"BU":"BTN","EP":"UTG","MP":"HJ","LJ":"HJ"}  # tweak if needed

def norm_pos(p:str)->str:
    p = p.upper()
    return ALIAS.get(p,p)

def scan(root: Path):
    files = list(root.rglob("*.txt"))
    print(f"Found {len(files)} files under {root}")
    tokens = Counter()
    pos_tokens = Counter()
    action_tokens = Counter()
    opener_first = Counter()
    hero_vs_opener = Counter()
    stems = Counter()

    def toks_of(stem:str):
        return [t for t in stem.split("_") if t]

    def classify(tok:str):
        # Heuristic: POS if in POS set (with aliasing); else action-ish
        if norm_pos(tok) in POS:
            return "POS", norm_pos(tok)
        return "ACT", tok

    for p in files:
        parts = p.parts
        try:
            stack = next(int(pp[:-2]) for pp in parts if pp.lower().endswith("bb"))
        except StopIteration:
            stack = None
        hero = norm_pos(p.parent.name)

        stem = p.stem
        toks = toks_of(stem)
        tokens.update(toks)

        # parse alternating POS / ACT pairs
        seq = []
        i = 0
        while i < len(toks):
            kind, val = classify(toks[i])
            if kind != "POS":
                # skip garbage token and continue
                i += 1
                continue
            act = None
            if i+1 < len(toks):
                k2,v2 = classify(toks[i+1])
                if k2 != "POS":
                    act = v2
                    i += 2
                else:
                    i += 1
            else:
                i += 1
            pos_tokens.update([val])
            if act: action_tokens.update([act])
            seq.append((val, act))

        # opener = first non-FOLD with some action token
        opener = None
        for pos,act in seq:
            if act and act.upper() not in {"FOLD"}:
                opener = (pos, act.upper())
                break

        # stem for first two relevant acts (for SRP mapping hints)
        short = []
        for pos,act in seq:
            if pos and act:
                short.append(f"{pos}_{act}")
            if len(short) >= 2:
                break
        if short:
            stems["_".join(short)] += 1

        if opener:
            opener_first[(opener[0], opener[1])] += 1
            hero_vs_opener[(hero, opener[0])] += 1

    # Print taxonomy
    print("\n=== Token taxonomy ===")
    print(f"Distinct POS tokens: {len(pos_tokens)} → {sorted(pos_tokens.keys())}")
    print(f"Top actions: {action_tokens.most_common(20)}")

    print("\n=== Who opens first (by (pos,action)) ===")
    for (pos,act),n in opener_first.most_common(20):
        print(f"  {pos:>3} {act:<10}  x{n}")

    print("\n=== Hero-folder vs opener-pos (matrix head) ===")
    for (hero,op),n in hero_vs_opener.most_common(30):
        print(f"  hero={hero:>3}  opener={op:>3}  x{n}")

    print("\n=== Common stems (first two POS_ACT tokens) ===")
    for s,n in stems.most_common(30):
        print(f"  {s}  x{n}")

if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/vendor/monker")
    scan(root)