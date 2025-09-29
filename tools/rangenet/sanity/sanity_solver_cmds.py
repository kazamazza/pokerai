#!/usr/bin/env python3
"""
Sanity check for solver command text emitted per manifest row.

Validates (per row):
- set_board is comma-separated correctly
- set_range_ip / set_range_oop exist (non-empty)
- FLOP bet family present per role (bet vs donk where applicable)
- FLOP raise ladder present for both roles with 150/200/300 (>100)
- FLOP allin present for both roles (if enabled)
- No raise lines on turn/river (we keep those trees lean)
- Limped and caller-OOP cases: donk rules apply only where intended

Exits non-zero if any hard errors are found.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.config.bet_menus import build_contextual_bet_sizes, _parse_menu_id
from ml.range.solvers.command_text import build_command_text

# ----- import your project helpers -----
# --------- regex helpers ----------
RE_SET_BOARD = re.compile(r"^set_board\s+([2-9TJQKA][cdhs](?:,[2-9TJQKA][cdhs]){2,})$", re.I)
RE_SET_RANGE = re.compile(r"^set_range_(ip|oop)\s+(.+)$", re.I)
RE_SET_BS    = re.compile(r"^set_bet_sizes\s+(ip|oop),(flop|turn|river),(bet|donk|raise|allin)(?:,(.*))?$", re.I)

RE_RAISE_LIST = re.compile(r"(?:^|,) *(1[5-9]0|[2-9]\d{2,}) *(?:,|$)")  # >=150, accepts 150/200/300/...
REQ_RAISES = {"150","200","300"}  # we require these to be present


def _lines(cmd: str) -> list[str]:
    return [ln.strip() for ln in cmd.splitlines() if ln.strip()]


def _fail(msgs: list[str], row_ix: int, msg: str) -> None:
    msgs.append(f"[row {row_ix}] {msg}")


def check_one(row: Dict[str, Any], row_ix: int, hard_errors: list[str], soft_warnings: list[str]) -> None:
    menu_id = str(row.get("bet_sizing_id", "") or "")
    pot_bb  = float(row.get("pot_bb", 0.0))
    eff_bb  = float(row.get("effective_stack_bb", row.get("stack_bb", 0.0)))
    board   = str(row.get("board", ""))
    r_ip    = str(row.get("range_ip", "AA"))
    r_oop   = str(row.get("range_oop", "KK"))

    # Build bet sizes + command
    bet_sizes = build_contextual_bet_sizes(menu_id)
    cmd = build_command_text(
        pot_bb=pot_bb or 50.0,
        effective_stack_bb=eff_bb or 100.0,
        board=board or "QsJh2h",
        range_ip=r_ip,
        range_oop=r_oop,
        bet_sizes=bet_sizes,
        dump_path="output_result.json",
    )

    ls = _lines(cmd)

    # ---- board line
    sb = [l for l in ls if l.lower().startswith("set_board ")]
    if not sb:
        _fail(hard_errors, row_ix, "missing set_board")
    else:
        m = RE_SET_BOARD.match(sb[0])
        if not m:
            _fail(hard_errors, row_ix, f"bad set_board format: {sb[0]}")

    # ---- ranges
    have_ip = any(RE_SET_RANGE.match(l) and RE_SET_RANGE.match(l).group(1).lower()=="ip" for l in ls)
    have_oop= any(RE_SET_RANGE.match(l) and RE_SET_RANGE.match(l).group(1).lower()=="oop" for l in ls)
    if not have_ip:  _fail(hard_errors, row_ix, "missing set_range_ip")
    if not have_oop: _fail(hard_errors, row_ix, "missing set_range_oop")

    # Parse all bet-size directives
    by_key = {}  # (role, street, kind) -> payload string (sizes or None)
    for l in ls:
        m = RE_SET_BS.match(l)
        if not m:
            continue
        role, street, kind, payload = m.groups()
        key = (role.lower(), street.lower(), kind.lower())
        by_key[key] = (payload or "").strip()

    # Must have flop raise ladder + allin for both roles
    for role in ("ip","oop"):
        # raise present
        k = (role, "flop", "raise")
        if k not in by_key:
            _fail(hard_errors, row_ix, f"missing flop raise for {role}")
        else:
            sizes = {s.strip() for s in by_key[k].split(",") if s.strip()}
            missing = [r for r in REQ_RAISES if r not in sizes]
            if missing:
                _fail(hard_errors, row_ix, f"{role} flop raise missing {missing}, got {sorted(sizes)}")
            # sanity: all > 100
            if not all(s.isdigit() and int(s) > 100 for s in sizes):
                _fail(hard_errors, row_ix, f"{role} flop raise has invalid values: {by_key[k]}")

        # allin present
        if (role, "flop", "allin") not in by_key:
            _fail(hard_errors, row_ix, f"missing flop allin for {role}")

    # Role-driven bet vs donk expectations
    group, role_name = _parse_menu_id(menu_id)
    caller_oop = role_name.endswith("Caller_OOP") or role_name == "Caller_OOP"

    # IP should have bet (not donk)
    if ( "ip","flop","bet") not in by_key:
        _fail(hard_errors, row_ix, "IP missing flop bet sizes")
    if ("ip","flop","donk") in by_key:
        soft_warnings.append(f"[row {row_ix}] IP has donk line (unexpected)")

    # OOP: donk only when caller_OOP or limped_single
    have_oop_donk = ("oop","flop","donk") in by_key
    if caller_oop or group.startswith("limped_single"):
        if not have_oop_donk:
            soft_warnings.append(f"[row {row_ix}] OOP expected donk (caller_OOP/limped_single) but missing")
    else:
        if have_oop_donk:
            soft_warnings.append(f"[row {row_ix}] OOP has donk in non-caller/limped context")

    # No raises on turn/river
    for st in ("turn","river"):
        for role in ("ip","oop"):
            if (role, st, "raise") in by_key:
                _fail(hard_errors, row_ix, f"unexpected {st} raise line for {role}")

    # Sizes must not be empty where present (bet/donk/raise)
    for (role, street, kind), payload in by_key.items():
        if kind in ("bet","donk","raise") and payload == "":
            _fail(hard_errors, row_ix, f"empty sizes list: {(role,street,kind)}")


def main():
    ap = argparse.ArgumentParser("Sanity check solver command text before launching solves")
    ap.add_argument("--manifest", required=True, help="Path to manifest parquet with s3_key, bet_sizing_id, pot_bb, effective_stack_bb, board, ranges, etc.")
    ap.add_argument("--max-rows", type=int, default=2000, help="Sample up to this many rows (head)")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows).copy()

    hard_errors: list[str] = []
    soft_warnings: list[str] = []

    for i, r in enumerate(df.to_dict("records")):
        try:
            check_one(r, i, hard_errors, soft_warnings)
        except Exception as e:
            hard_errors.append(f"[row {i}] exception: {e}")

    print("\n=== Solver Command Sanity ===")
    print(f"rows checked: {len(df)}")
    print(f"hard errors : {len(hard_errors)}")
    print(f"warnings    : {len(soft_warnings)}")

    if soft_warnings:
        print("\n-- warnings (first 20) --")
        for w in soft_warnings[:20]:
            print(w)

    if hard_errors:
        print("\n-- hard errors (first 50) --")
        for e in hard_errors[:50]:
            print(e)
        sys.exit(2)

    print("\n✅ OK — menus look consistent for all checked rows.")
    sys.exit(0)


if __name__ == "__main__":
    main()