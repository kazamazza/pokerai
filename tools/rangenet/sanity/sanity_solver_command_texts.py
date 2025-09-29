#!/usr/bin/env python3
"""
Build-and-verify solver command text for each manifest row.

What we enforce per row:
- set_board is comma-separated like "Qs,Jh,2h"
- set_range_ip / set_range_oop present (non-empty strings)
- FLOP: both roles have raise ladder 150/200/300 and an allin line
- FLOP: IP has 'bet' (not donk). OOP has 'donk' only in caller_OOP / limped_single
- Turn/River: no 'raise' lines (we keep those trees lean)
- Sizes on bet/donk/raise are non-empty and match menu expectations
- Pot and stack are positive; command ends with 'dump_result ...'

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

from ml.config.bet_menus import build_contextual_bet_sizes, _parse_menu_id, BET_SIZE_MENUS
from ml.range.solvers.command_text import build_command_text

# ---------- regexes ----------
RE_SET_BOARD  = re.compile(r"^set_board\s+([2-9TJQKA][cdhs](?:,[2-9TJQKA][cdhs]){2,})$", re.I)
RE_SET_RANGE  = re.compile(r"^set_range_(ip|oop)\s+(.+)$", re.I)
RE_SET_BS     = re.compile(r"^set_bet_sizes\s+(ip|oop),(flop|turn|river),(bet|donk|raise|allin)(?:,(.*))?$", re.I)
RE_DUMP       = re.compile(r"^dump_result\s+(\S+)$", re.I)

REQUIRED_RAISES = {"150", "200", "300"}  # raise-to as % of facing bet

def _lines(cmd: str) -> list[str]:
    return [ln.strip() for ln in cmd.splitlines() if ln.strip()]

def _fail(hard_errors: list[str], row_ix: int, msg: str) -> None:
    hard_errors.append(f"[row {row_ix}] {msg}")

def _warn(soft_warnings: list[str], row_ix: int, msg: str) -> None:
    soft_warnings.append(f"[row {row_ix}] {msg}")

def _as_int_set(csv: str) -> set[str]:
    return {t.strip() for t in (csv or "").split(",") if t.strip()}

def _sizes_from_menu_id(menu_id: str) -> list[int]:
    # Mirrors build_contextual_bet_sizes’ size derivation (0.33 -> 33 etc)

    sizes = BET_SIZE_MENUS.get(menu_id or "", None)
    if not sizes:
        sizes = [0.33]
    return sorted({int(round(x * 100)) for x in sizes})

def check_one(row: Dict[str, Any], row_ix: int, hard: list[str], warn: list[str]) -> None:
    menu_id = str(row.get("bet_sizing_id", "") or "")
    pot_bb  = float(row.get("pot_bb", 0.0) or 0.0)
    eff_bb  = float(row.get("effective_stack_bb", row.get("stack_bb", 0.0)) or 0.0)
    board   = str(row.get("board", "") or "QsJh2h")
    rng_ip  = str(row.get("range_ip", "") or "AA")
    rng_oop = str(row.get("range_oop", "") or "KK")

    if pot_bb <= 0 or eff_bb <= 0:
        _fail(hard, row_ix, f"non-positive pot/stack (pot={pot_bb}, stack={eff_bb})")

    bet_sizes = build_contextual_bet_sizes(menu_id)
    cmd = build_command_text(
        pot_bb=pot_bb,
        effective_stack_bb=eff_bb,
        board=board,
        range_ip=rng_ip,
        range_oop=rng_oop,
        bet_sizes=bet_sizes,
        dump_path="output_result.json",
    )
    ls = _lines(cmd)

    # --- board line ---
    sb = [l for l in ls if l.lower().startswith("set_board ")]
    if not sb or not RE_SET_BOARD.match(sb[0]):
        _fail(hard, row_ix, f"bad or missing set_board: {sb[0] if sb else '<none>'}")

    # --- ranges ---
    have_ip  = any(RE_SET_RANGE.match(l) and RE_SET_RANGE.match(l).group(1).lower() == "ip"  for l in ls)
    have_oop = any(RE_SET_RANGE.match(l) and RE_SET_RANGE.match(l).group(1).lower() == "oop" for l in ls)
    if not have_ip:  _fail(hard, row_ix, "missing set_range_ip")
    if not have_oop: _fail(hard, row_ix, "missing set_range_oop")

    # --- collect all bet-size directives ---
    lines_by_key: Dict[tuple[str,str,str], str] = {}
    for l in ls:
        m = RE_SET_BS.match(l)
        if not m:
            continue
        role, street, kind, payload = m.groups()
        key = (role.lower(), street.lower(), kind.lower())
        lines_by_key[key] = (payload or "").strip()

    # --- flop raises and allin for both roles ---
    for role in ("ip", "oop"):
        k_raise = (role, "flop", "raise")
        if k_raise not in lines_by_key:
            _fail(hard, row_ix, f"missing flop raise for {role}")
        else:
            got = _as_int_set(lines_by_key[k_raise])
            missing = [x for x in sorted(REQUIRED_RAISES) if x not in got]
            if missing:
                _fail(hard, row_ix, f"{role} flop raise missing {missing}; got {sorted(got)}")
            # sanity: ensure they’re > 100
            if not all(s.isdigit() and int(s) > 100 for s in got):
                _fail(hard, row_ix, f"{role} flop raise has invalid sizes: {lines_by_key[k_raise]}")

        k_allin = (role, "flop", "allin")
        if k_allin not in lines_by_key:
            _fail(hard, row_ix, f"missing flop allin for {role}")

    # --- bet vs donk correctness ---
    group, role_name = _parse_menu_id(menu_id)
    caller_oop = role_name.endswith("Caller_OOP") or role_name == "Caller_OOP"

    # IP must have bet (not donk)
    if ("ip", "flop", "bet") not in lines_by_key:
        _fail(hard, row_ix, "IP missing flop bet sizes")
    if ("ip", "flop", "donk") in lines_by_key:
        _warn(warn, row_ix, "IP has donk on flop (unexpected)")

    # OOP: donk only when caller_OOP or limped_single.*
    have_oop_donk = ("oop", "flop", "donk") in lines_by_key
    if caller_oop or group.startswith("limped_single"):
        if not have_oop_donk:
            _warn(warn, row_ix, "OOP expected donk (caller_OOP/limped_single) but missing")
    else:
        if have_oop_donk:
            _warn(warn, row_ix, "OOP has donk in non-caller/limped context")

    # --- sizes must match menu expectation for 'bet' or 'donk' families ---
    expected_pct = {str(x) for x in _sizes_from_menu_id(menu_id)}
    for fam in ("bet", "donk"):
        for role in ("ip", "oop"):
            k = (role, "flop", fam)
            if k in lines_by_key:  # if present, it must match expected menu
                got = _as_int_set(lines_by_key[k])
                if not got:
                    _fail(hard, row_ix, f"empty {role} flop {fam} sizes")
                elif got != expected_pct:
                    _fail(hard, row_ix, f"{role} flop {fam} sizes {sorted(got)} != expected {sorted(expected_pct)}")

    # --- no raises on turn/river ---
    for st in ("turn", "river"):
        for role in ("ip", "oop"):
            if (role, st, "raise") in lines_by_key:
                _fail(hard, row_ix, f"unexpected {st} raise for {role}")

    # --- command ends with a dump_result path ---
    dumps = [l for l in ls if RE_DUMP.match(l)]
    if not dumps:
        _fail(hard, row_ix, "missing dump_result line")

def main():
    ap = argparse.ArgumentParser("Sanity check: build + validate solver command text for each manifest row")
    ap.add_argument("--manifest", required=True, help="Path to manifest parquet")
    ap.add_argument("--max-rows", type=int, default=5000, help="Sample head(N) rows for speed")
    ap.add_argument("--fail-on-warn", action="store_true", help="Treat warnings as errors")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows).copy()

    hard: list[str] = []
    warn: list[str] = []
    for i, r in enumerate(df.to_dict("records")):
        try:
            check_one(r, i, hard, warn)
        except Exception as e:
            _fail(hard, i, f"exception during build/validate: {e}")

    print("\n=== Solver Command Text Sanity ===")
    print(f"rows checked: {len(df)}")
    print(f"hard errors : {len(hard)}")
    print(f"warnings    : {len(warn)}")

    if warn:
        print("\n-- warnings (first 30) --")
        for w in warn[:30]:
            print(w)

    if hard or (warn and args.fail_on_warn):
        print("\n-- hard errors (first 60) --")
        for e in hard[:60]:
            print(e)
        sys.exit(2)

    print("\n✅ OK — all command texts look consistent.")
    sys.exit(0)

if __name__ == "__main__":
    main()