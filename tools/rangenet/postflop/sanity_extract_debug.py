# tools/rangenet/postflop/sanity_extract_debug.py
# -*- coding: utf-8 -*-
"""
Sanity runner: loops over data/debug_samples, parses each solve, and prints
the non-zero action probs aligned to ACTION_VOCAB.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.solver_policy_parser import parse_solver_file, default_pot_for_filename
from ml.models.policy_consts import ACTION_VOCAB


BET_SIZE_MENUS = {
    "srp_hu.PFR_IP":      [0.33, 0.66],
    "srp_hu.Caller_OOP":  [0.33],
    "3bet_hu.Aggressor_IP":  [0.33, 0.66],
    "3bet_hu.Aggressor_OOP": [0.33, 0.66],
    "4bet_hu.Aggressor_IP":  [0.33],
    "4bet_hu.Aggressor_OOP": [0.33],
    "limped_single.SB_IP": [0.33],
    "limped_multi.Any":    [0.33],
}

def _menu_id_from_filename(base: str) -> str:
    stem = os.path.splitext(os.path.splitext(base)[0])[0]  # handle .json.gz
    return stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/debug_samples")
    ap.add_argument("--stack-bb", type=float, default=100.0)
    ap.add_argument("--pot-override", type=float, default=None)
    ap.add_argument("--max-depth", type=int, default=18)
    args = ap.parse_args()

    # accept file or directory
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(
            glob.glob(os.path.join(args.input, "*.json")) +
            glob.glob(os.path.join(args.input, "*.json.gz"))
        )
    if not paths:
        print(f"[err] no solves found under {args.input}"); return

    dump = {}
    print("\n=== Parsed action distributions (non-zero only) ===")
    for p in paths:
        base = os.path.basename(p)
        pot_bb = args.pot_override if args.pot_override is not None else default_pot_for_filename(base)
        menu_id = _menu_id_from_filename(base)
        menu_pcts = BET_SIZE_MENUS.get(menu_id, [0.33])

        probs, meta = parse_solver_file(
            p, pot_bb=pot_bb, stack_bb=args.stack_bb, menu_pcts=menu_pcts, max_depth=args.max_depth
        )

        ordered = [(k, probs.get(k, 0.0)) for k in ACTION_VOCAB if k in probs]
        pretty = ", ".join(f"{k}:{v:.3f}" for k, v in ordered if v > 0)
        print(f"- {base} | pot={pot_bb:.1f} | stack={args.stack_bb:.1f} | menu={menu_id} {menu_pcts} → {pretty}")

        if meta.get("raises_seen"):
            samples = []
            for m in meta["raises_seen"][:6]:
                rb = f"{m['ratio_rb']:.2f}" if m.get("ratio_rb") is not None else "nan"
                rp = f"{m['ratio_pot']:.2f}" if m.get("ratio_pot") is not None else "nan"
                samples.append(
                    f"{m['label']} (to={m['raise_to_bb']:.1f}, facing={m['facing_bet_bb']:.1f}, rb={rb}, potr={rp})"
                )
            print("    raises_seen:", "; ".join(samples))

        dump[base] = {"pot": pot_bb, "stack": args.stack_bb, "menu_id": menu_id, "menu_pcts": menu_pcts,
                      "probs": probs, "meta": meta}

    with open("data/debug_samples_parsed.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)
    print("\n[OK] wrote data/debug_samples_parsed.json")

if __name__ == "__main__":
    main()