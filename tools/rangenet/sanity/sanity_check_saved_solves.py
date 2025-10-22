# tools/rangenet/sanity/sanity_check_saved_solves.py
from __future__ import annotations
import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.config.bet_menus import build_contextual_bet_sizes
from ml.config.solver import STAKE_CFG
from ml.core.types import Stakes
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor

# --- Assumes these are importable in your project context ---
# from your_module import TexasSolverExtractor, STAKE_CFG, Stakes, build_contextual_bet_sizes
# If they live near this file, adjust imports accordingly:

# -------- filename parsing --------
FN_RE = re.compile(
    r"^(?P<ctx>[A-Z0-9_]+)_"          # context (VS_OPEN, LIMPED_SINGLE, etc.)
    r"(?P<ip>[A-Z]{2,3})v(?P<oop>[A-Z]{2,3})_"  # positions
    r"(?P<menu>.+)_"                  # menu id (can contain underscores)
    r"(?P<pot>[0-9]+(?:\.[0-9]+)?)bb\.json(?:\.gz)?$",  # pot
    re.IGNORECASE
)

def parse_filename(p: Path) -> Tuple[str, str, str, str, float]:
    m = FN_RE.match(p.name)
    if not m:
        raise ValueError(f"unrecognized file name pattern: {p.name}")
    ctx   = m.group("ctx")
    ip    = m.group("ip")
    oop   = m.group("oop")
    menu  = m.group("menu")
    pot   = float(m.group("pot"))
    return ctx, ip, oop, menu, pot

# -------- minimal vocab derivation + bucketing --------
def _pct_token(prefix: str, frac: float) -> str:
    return f"{prefix}_{int(round(frac * 100))}"

def _raise_token(mult: float) -> str:
    return f"RAISE_{int(round(mult * 100))}"

def derive_vocab_from_menu(menu_id: Optional[str], *, stake: Stakes) -> Tuple[Set[str], Set[str], Dict]:
    cfg = build_contextual_bet_sizes(menu_id, stake=stake)
    flop = cfg["flop"]
    sizes = sorted(set((flop["ip"].get("bet") or []) + (flop["oop"].get("bet") or [])))
    stake_cfg = STAKE_CFG[stake]
    raise_mult = stake_cfg["raise_mult"]
    enable_allin = bool(stake_cfg.get("flop_allin", True))
    allow_donk = "donk" in (flop["oop"] or {})
    root_tokens = {"CHECK"} | { _pct_token("BET", s) for s in sizes }
    if allow_donk:
        root_tokens |= { _pct_token("DONK", s) for s in sizes }
    facing_tokens = {"CALL", "FOLD"} | { _raise_token(m) for m in raise_mult }
    if enable_allin:
        facing_tokens.add("ALLIN")
    dbg = dict(sizes=sizes, raise_mult=raise_mult, allow_donk=allow_donk, enable_allin=enable_allin)
    return root_tokens, facing_tokens, dbg

def validate_seen_vs_expected(root_mix: Dict[str,float], facing_mix: Dict[str,float],
                              menu_id: str, stake: Stakes) -> Tuple[bool, str]:
    exp_root, exp_face, dbg = derive_vocab_from_menu(menu_id, stake=stake)
    seen_root = {k for k,v in root_mix.items() if v > 1e-8}
    seen_face = {k for k,v in facing_mix.items() if v > 1e-8}
    ok_root   = any(t in seen_root for t in exp_root)
    ok_face   = any(t in seen_face for t in exp_face)
    if ok_root and ok_face:
        return True, f"sizes={dbg['sizes']} raises={dbg['raise_mult']} donk={dbg['allow_donk']} allin={dbg['enable_allin']}"
    miss_root = exp_root - seen_root
    miss_face = exp_face - seen_face
    msg = []
    if not ok_root:  msg.append(f"root missing any-of expected; seen={sorted(seen_root)} exp={sorted(exp_root)}")
    if not ok_face:  msg.append(f"facing missing any-of expected; seen={sorted(seen_face)} exp={sorted(exp_face)}")
    return False, " | ".join(msg)

# -------- smoke runner --------
def load_json_any(p: Path) -> Dict:
    if p.suffix == ".gz" or p.name.endswith(".json.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(p.read_text(encoding="utf-8"))

def stakes_from_str(s: str) -> Stakes:
    s = (s or "NL10").upper()
    for enum_val in Stakes:
        if enum_val.name.upper() == s:  # if Enum named NL10/NL25
            return enum_val
    # or map like your earlier helper:
    if s == "NL25": return Stakes.NL25
    if s == "NL10": return Stakes.NL10
    if s == "NL5":  return Stakes.NL5
    if s == "NL2":  return Stakes.NL2
    return Stakes.NL10

def topk(d: Dict[str,float], k=3) -> str:
    return ", ".join([f"{k}:{v:.3f}" for k,v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]])

def main():
    ap = argparse.ArgumentParser(description="Sanity-check saved solver JSONs using TexasSolverExtractor.")
    ap.add_argument("--glob", default="data/ts_smoke/*.json", help="Glob for solved JSONs")
    ap.add_argument("--stake", default="NL10", help="Stake tier (e.g., NL10)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    stake = stakes_from_str(args.stake)
    files = sorted([Path(p) for p in Path().glob(args.glob)])
    if not files:
        print(f"No files matched: {args.glob}")
        sys.exit(2)

    x = TexasSolverExtractor()
    failures: List[Tuple[str,str]] = []
    checked = 0

    for fp in files:
        try:
            ctx, ip, oop, menu_id, pot_bb = parse_filename(fp)
        except Exception as e:
            failures.append((fp.name, f"filename_parse: {e}"))
            continue

        # Build sizes/raises from stake config to guide bucketing
        menu_cfg = build_contextual_bet_sizes(menu_id, stake=stake)
        bet_sizes = list(sorted(set((menu_cfg["flop"]["ip"].get("bet") or []) + (menu_cfg["flop"]["oop"].get("bet") or []))))
        raise_mults = STAKE_CFG[stake]["raise_mult"]
        # Load payload to pick effective_stack_bb (fallback if absent)
        try:
            payload = load_json_any(fp)
            stack_bb = float(payload.get("stack_bb") or payload.get("effective_stack_bb") or 60.0)
            board = str(payload.get("board") or "QsJh2h")
        except Exception:
            stack_bb = 60.0
            board = "QsJh2h"

        ex = x.extract(
            str(fp),
            ctx=ctx,
            ip_pos=ip,
            oop_pos=oop,
            board=board,
            pot_bb=pot_bb,
            stack_bb=stack_bb,
            bet_sizing_id=menu_id,
            bet_sizes=bet_sizes,
            raise_mults=raise_mults,
        )

        if not ex.ok:
            failures.append((fp.name, f"extract_failed: {ex.reason}"))
            continue

        ok, why = validate_seen_vs_expected(ex.root_mix, ex.facing_mix, menu_id, stake)
        if ok:
            checked += 1
            print(f"✅ {fp.name} | {ctx} {ip}v{oop} | {why}")
            if args.verbose:
                print(f"    root:   {topk(ex.root_mix)}")
                print(f"    facing: {topk(ex.facing_mix)}  via={ex.meta.get('facing_path')}")
        else:
            failures.append((fp.name, f"vocab_mismatch: {why}"))
            print(f"❌ {fp.name} | {ctx} {ip}v{oop} | {why}")

    print(f"\nSUMMARY: checked={checked} failed={len(failures)}")
    if failures:
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()