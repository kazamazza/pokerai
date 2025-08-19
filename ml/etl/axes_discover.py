# ml/etl/axes_discover.py
from __future__ import annotations
import json, gzip
from pathlib import Path

IN_PATH  = Path("data/preflop/preflop.hu.v1.jsonl.gz")
OUT_PATH = Path("ml/config/axes.auto.json")

# derive action_context from filename sequence (HU subset)
def derive_context(action_seq):
    # simple rules:
    # - if first act is OPEN/RAISE/LIMP from X and our hero is later, the later seat responds → VS_OPEN
    # - presence of '3BET'/'4BET' in facing branch → VS_3BET / VS_4BET
    acts = [a["act"] for a in action_seq]
    if "4BET" in acts: return "VS_4BET"
    if "3BET" in acts: return "VS_3BET"
    if "OPEN" in acts or "RAISE" in acts or "LIMP" in acts: return "VS_OPEN"
    # If this file encodes the *opener* itself (some packs do), tag OPEN
    if acts and acts[0] in {"OPEN","RAISE","LIMP","AI"}: return "OPEN"
    return "VS_OPEN"

def run():
    stacks, positions, contexts = set(), set(), set()
    # optional: also count frequencies
    with gzip.open(IN_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            m = rec["meta"]
            if m.get("is_multiway"):       # v1 skip, you already filtered earlier; safe guard
                continue
            stacks.add(int(m["stack_bb"]))
            positions.add(m["hero_position"])
            ctx = derive_context(m["action_sequence"])
            contexts.add(ctx)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as out:
        json.dump({
            "stacks": sorted(stacks),
            "positions": sorted(positions),
            "action_contexts": sorted(contexts)
        }, out, indent=2)
    print(f"✅ wrote {OUT_PATH}")

if __name__ == "__main__":
    run()