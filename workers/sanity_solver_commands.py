# workers/sanity_solver_commands.py
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# ---- adjust these imports to your project ----
from ml.etl.rangenet.postflop.build_rangenet_postflop_manifest import load_yaml
from workers.stake_params import load_stake_params, StakeSolverParams
from workers.submit_solver_jobs_pilot import pick_one_per_scenario_family, parse_bet_sizes  # reuse your helpers
from workers.rangenet_postflop_solver_worker import _build_solver_command_text_for_job, \
    _oop_root_kind  # your real builder


# ----------------------------
# Parsing helpers (commands.txt)
# ----------------------------

BET_LINE_RE = re.compile(
    r"^set_bet_sizes\s+(?P<role>ip|oop),(?P<street>flop|turn|river),(?P<kind>donk|bet|raise|allin)(?:,(?P<csv>.*))?$"
)

STREET_BY_ID = {1: "flop", 2: "turn", 3: "river"}

def _street_name(street_id: int) -> str:
    return STREET_BY_ID.get(int(street_id), "flop")

def _extract_bet_lines(cmd: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in cmd.splitlines():
        line = line.strip()
        if not line.startswith("set_bet_sizes "):
            continue
        m = BET_LINE_RE.match(line)
        if not m:
            out.append({"raw": line, "parse_ok": False})
            continue
        csv = m.group("csv")
        vals = []
        if csv:
            vals = [x.strip() for x in csv.split(",") if x.strip() != ""]
        out.append({
            "raw": line,
            "parse_ok": True,
            "role": m.group("role"),
            "street": m.group("street"),
            "kind": m.group("kind"),
            "vals": vals,
        })
    return out

def _find_lines(lines: List[Dict[str, Any]], *, role: str, street: str, kind: str) -> List[Dict[str, Any]]:
    return [x for x in lines if x.get("parse_ok") and x["role"] == role and x["street"] == street and x["kind"] == kind]

def _as_floats(vals: List[str]) -> List[float]:
    out = []
    for v in vals:
        out.append(float(v))
    return out


# ----------------------------
# Contract rules per scenario
# ----------------------------

@dataclass(frozen=True)
class ScenarioRule:
    # For the current street, which menus are allowed?
    allow_ip_bet: bool
    allow_oop_bet: bool
    allow_oop_donk: bool

def rule_for_bet_sizing_id(bet_sizing_id: str, *, oop_root_kind: str) -> ScenarioRule:
    """
    Uses your existing mapping/intent.
    We assume:
      - Non-limp: IP always has bet menu (the bet size being solved)
      - OOP has either donk OR bet menu depending on oop_root_kind
      - Limp: only IP has bet menu, OOP has none (donk/bet disabled)
    """
    bid = (bet_sizing_id or "").strip()

    is_limp = bid.startswith("limped_single") or bid.startswith("limped_multi")
    if is_limp:
        return ScenarioRule(allow_ip_bet=True, allow_oop_bet=False, allow_oop_donk=False)

    # non-limp
    if oop_root_kind == "bet":
        return ScenarioRule(allow_ip_bet=True, allow_oop_bet=True, allow_oop_donk=False)
    else:
        # donk
        return ScenarioRule(allow_ip_bet=True, allow_oop_bet=False, allow_oop_donk=True)


# ----------------------------
# Assertions
# ----------------------------

class CommandSanityError(RuntimeError):
    pass

def assert_command_sanity(
    *,
    cmd: str,
    bet_sizing_id: str,
    street_id: int,
    size_pct: int,
    raise_mults: Sequence[float],
    oop_root_kind: str,
) -> None:
    sn = _street_name(street_id)
    lines = _extract_bet_lines(cmd)

    # 0) Must contain core lines
    must_have = ["set_pot", "set_effective_stack", "set_board", "set_range_ip", "set_range_oop", "build_tree", "start_solve", "dump_result"]
    for token in must_have:
        if token not in cmd:
            raise CommandSanityError(f"missing required command token: {token}")

    # 1) Every set_bet_sizes line must parse
    bad = [x["raw"] for x in lines if not x.get("parse_ok")]
    if bad:
        raise CommandSanityError(f"unparseable set_bet_sizes lines: {bad[:5]}")

    # 2) Raise lines must be multipliers, not percents/fractions
    # Expect raise lines for both roles and all streets (your generator currently sets them everywhere).
    expected_raise = [float(x) for x in raise_mults]
    for role in ("ip", "oop"):
        for st in ("flop", "turn", "river"):
            raise_lines = _find_lines(lines, role=role, street=st, kind="raise")
            if not raise_lines:
                raise CommandSanityError(f"missing raise menu for {role},{st}")
            vals = _as_floats(raise_lines[0]["vals"])
            if vals != expected_raise:
                raise CommandSanityError(f"raise menu mismatch for {role},{st}: got={vals} expected={expected_raise}")

    # 3) Bet/donk sizing must exist ONLY where allowed, and ONLY on current street, with the correct size
    rule = rule_for_bet_sizing_id(bet_sizing_id, oop_root_kind=oop_root_kind)

    # helper to check no bet/donk set on non-current streets (very important)
    def ensure_no_menu_on_other_streets(role: str, kind: str) -> None:
        for st in ("flop", "turn", "river"):
            if st == sn:
                continue
            if _find_lines(lines, role=role, street=st, kind=kind):
                raise CommandSanityError(f"unexpected {kind} menu on non-current street: {role},{st}")

    # IP bet
    ip_bet = _find_lines(lines, role="ip", street=sn, kind="bet")
    if rule.allow_ip_bet:
        if not ip_bet:
            raise CommandSanityError(f"expected ip,{sn},bet menu but missing")
        vals = ip_bet[0]["vals"]
        if len(vals) != 1 or int(float(vals[0])) != int(size_pct):
            raise CommandSanityError(f"ip,{sn},bet wrong size: got={vals} expected=[{size_pct}]")
        ensure_no_menu_on_other_streets("ip", "bet")
    else:
        if ip_bet:
            raise CommandSanityError(f"unexpected ip,{sn},bet menu present for bet_sizing_id={bet_sizing_id}")

    # OOP bet
    oop_bet = _find_lines(lines, role="oop", street=sn, kind="bet")
    if rule.allow_oop_bet:
        if not oop_bet:
            raise CommandSanityError(f"expected oop,{sn},bet menu but missing (oop_root_kind=bet)")
        vals = oop_bet[0]["vals"]
        if len(vals) != 1 or int(float(vals[0])) != int(size_pct):
            raise CommandSanityError(f"oop,{sn},bet wrong size: got={vals} expected=[{size_pct}]")
        ensure_no_menu_on_other_streets("oop", "bet")
    else:
        if oop_bet:
            raise CommandSanityError(f"unexpected oop,{sn},bet menu present (oop_root_kind={oop_root_kind})")

    # OOP donk
    oop_donk = _find_lines(lines, role="oop", street=sn, kind="donk")
    if rule.allow_oop_donk:
        if not oop_donk:
            raise CommandSanityError(f"expected oop,{sn},donk menu but missing (oop_root_kind=donk)")
        vals = oop_donk[0]["vals"]
        if len(vals) != 1 or int(float(vals[0])) != int(size_pct):
            raise CommandSanityError(f"oop,{sn},donk wrong size: got={vals} expected=[{size_pct}]")
        ensure_no_menu_on_other_streets("oop", "donk")
    else:
        if oop_donk:
            raise CommandSanityError(f"unexpected oop,{sn},donk menu present (oop_root_kind={oop_root_kind})")

    # 4) Limp: explicitly ensure no OOP donk/bet on current street
    if bet_sizing_id.startswith("limped_"):
        if _find_lines(lines, role="oop", street=sn, kind="bet"):
            raise CommandSanityError("limp contract violated: oop bet menu present")
        if _find_lines(lines, role="oop", street=sn, kind="donk"):
            raise CommandSanityError("limp contract violated: oop donk menu present")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser("Sanity check commands.txt generation vs YAML intent")
    ap.add_argument("--manifest", required=True, help="postflop manifest parquet")
    ap.add_argument("--solver-yaml", required=True)
    ap.add_argument("--stake-key", default="Stakes.NL10")
    ap.add_argument("--streets", default="1,2,3")
    ap.add_argument("--per-group", type=int, default=10, help="rows per (street, bet_sizing_id)")
    ap.add_argument("--limit", type=int, default=0, help="optional cap of manifest rows before grouping")
    ap.add_argument("--write-out", default="", help="optional dir to write commands for failures")
    args = ap.parse_args()

    streets = [int(x) for x in args.streets.split(",") if x.strip()]
    df = pd.read_parquet(args.manifest)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    # load stake params and raise_mult
    solver_yaml = load_yaml(args.solver_yaml)
    stake_params: StakeSolverParams = load_stake_params(solver_yaml, args.stake_key)
    raise_mults = list(stake_params.raise_mult)

    pick_df = pick_one_per_scenario_family(df, streets=streets, per_group=max(1, int(args.per_group)))
    if pick_df.empty:
        print("⚠️ nothing selected")
        return

    out_dir = Path(args.write_out).resolve() if args.write_out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    bad = 0

    # iterate + expand sizes
    for row in pick_df.itertuples(index=False):
        sizes = parse_bet_sizes(getattr(row, "bet_sizes", None))
        if not sizes:
            continue

        for size_pct in sizes:
            total += 1
            params: Dict[str, Any] = {
                "street": int(getattr(row, "street")),
                "pot_bb": float(getattr(row, "pot_bb")),
                "effective_stack_bb": float(getattr(row, "effective_stack_bb")),
                "board": str(getattr(row, "board")),
                "range_ip": getattr(row, "range_ip"),
                "range_oop": getattr(row, "range_oop"),
                "bet_sizing_id": str(getattr(row, "bet_sizing_id")),
                "size_pct": int(size_pct),

                # profiles come from solver.yaml (already inside stake_params)
                "accuracy": float(getattr(row, "accuracy", 0.02)),
                "max_iter": int(getattr(row, "max_iter", 4000)),
                "allin_threshold": float(getattr(row, "allin_threshold", 0.67)),
            }

            # IMPORTANT: your builder determines limp oop_kind itself (bet vs donk)
            cmd = _build_solver_command_text_for_job(
                params=params,
                stake_params=stake_params,
                dump_path=Path("output_result.json"),
            )

            # figure oop_root_kind for rule purposes:
            # - limp forced to "bet" in your patched builder
            # - otherwise use your existing mapping if you have it inside the builder module
            bet_sizing_id = params["bet_sizing_id"]
            oop_root_kind = _oop_root_kind(bet_sizing_id)

            # If you intentionally override limp to "bet" in your patched builder,
            # then your sanity rules must match that reality:
            oop_root_kind = _oop_root_kind(bet_sizing_id)

            # If you intentionally override limp to "bet" in your patched builder,
            # then your sanity rules must match that reality:
            is_limp = bet_sizing_id.startswith("limped_single") or bet_sizing_id.startswith("limped_multi")

            if is_limp:
                # limp has special contract; root kind is irrelevant (you override menus anyway)
                oop_root_kind = "limp"  # sentinel value for logging/debug
            else:
                oop_root_kind = _oop_root_kind(bet_sizing_id)

            try:
                assert_command_sanity(
                    cmd=cmd,
                    bet_sizing_id=bet_sizing_id,
                    street_id=int(params["street"]),
                    size_pct=int(params["size_pct"]),
                    raise_mults=raise_mults,
                    oop_root_kind=oop_root_kind,
                )
                ok += 1
            except Exception as e:
                bad += 1
                sha1 = getattr(row, "sha1", "no_sha1")
                print(f"❌ FAIL sha1={sha1} street={params['street']} bet_sizing_id={bet_sizing_id} size={size_pct}: {e}")
                if out_dir:
                    p = out_dir / f"{sha1}_street{params['street']}_{bet_sizing_id}_size{size_pct}.txt"
                    p.write_text(cmd, encoding="utf-8")

    print("\n====================")
    print("COMMAND SANITY SUMMARY")
    print("====================")
    print("total:", total)
    print("ok   :", ok)
    print("fail :", bad)

    if bad > 0:
        raise SystemExit(2)

if __name__ == "__main__":
    main()