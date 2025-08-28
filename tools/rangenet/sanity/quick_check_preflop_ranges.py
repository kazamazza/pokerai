# tools/rangenet/quick_check_preflop_ranges.py
import argparse, random
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.utils.preflop_ranges import get_ranges_for_pair
from ml.utils.config import load_model_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rangenet/postflop")
    ap.add_argument("--stacks", type=int, nargs="*", default=[12,15,18])
    ap.add_argument("--pairs", nargs="*", default=["BTNvBB","COvBB","SBvBB","BTNvSB"])
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    for s in args.stacks:
        for pair in args.pairs:
            ip, oop = pair.split("v")
            ip_rng, oop_rng = get_ranges_for_pair(stack_bb=s, ip=ip, oop=oop, cfg=cfg)
            def ok(r): return isinstance(r, str) and ("..." not in r) and len(r) > 10 and "," in r
            print(f"{pair} @ {s}bb")
            print("  IP ok:", ok(ip_rng), "sample:", ip_rng[:80], "…")
            print("  OOP ok:", ok(oop_rng), "sample:", oop_rng[:80], "…")

if __name__ == "__main__":
    main()