from typing import Dict, List, Tuple

from utils.cluster_helpers import load_preflop_json_from_s3, _load_vs_open_doc_any

# --- Which actions count as "range" for each context/role ---
OPEN_OPENER_ACTIONS = {"open"}  # opener's raising range from action=OPEN

# defender facing an open (from action=VS_OPEN):
# include common synonyms you might emit in your preflop generator
VS_OPEN_DEFENDER_ACTIONS = {
    "call", "flat", "overcall", "cold_call",
    "defend",
    "3bet", "4bet", "jam"
}

def _collect_actions(doc: dict, wanted: set[str]) -> list[str]:
    actions = doc.get("actions") or {}
    combos = []
    for k, v in actions.items():
        if k.lower() in wanted:
            combos.extend(v)
    # de-dupe, keep order
    seen = set()
    out = []
    for c in combos:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _is_valid_open_pair(ip: str, oop: str) -> bool:
    """Basic sanity for SRP OPEN. Opener can be UTG/MP/CO/BTN/SB. BB doesn't open into BTN/CO/MP/UTG."""
    ipU, oopU = ip.upper(), oop.upper()
    if ipU == "BB":                    # BB doesn't first-in open facing BTN in a normal SRP pairing
        return False
    if ipU == "SB" and oopU not in {"BB"}:
        return False                   # SB open should face BB
    return True

def _load_open_opener_file(ip: str, oop: str, stack_bb: int, profile: str, exploit: str,
                           multiway: str, pop: str) -> dict:
    return load_preflop_json_from_s3(
        ip, oop, stack_bb, profile, exploit, multiway, pop, action_context="OPEN"
    )

def _load_vs_open_defender_file(
    ip: str, oop: str, stack_bb: int,
    profile: str, exploit: str, multiway: str, pop: str
) -> dict:
    """
    Load the defender file for 'hero facing OPEN' with the SAME IP_vs_OOP order
    you used for filenames. Do NOT reverse here.
    """
    return load_preflop_json_from_s3(
        ip, oop, stack_bb, profile, exploit, multiway, pop, action_context="VS_OPEN"
    )

def extract_ip_oop_ranges_for_open(ip: str, oop: str, stack_bb: int,
                                   villain_profile: str, exploit_setting: str,
                                   multiway_context: str, population_type: str) -> tuple[list[str], list[str]]:
    """
    For SRP OPEN:
      IP range  = opener's OPEN file (ip_vs_oop)
      OOP range = defender's VS_OPEN file (defender_vs_opener). We try both orderings to be safe.
    """
    # Opener (IP)
    opener_doc = load_preflop_json_from_s3(
        ip, oop, stack_bb, villain_profile, exploit_setting, multiway_context, population_type, action_context="OPEN"
    )
    ip_range = _collect_actions(opener_doc, OPEN_OPENER_ACTIONS)

    # Defender (OOP) — roles reversed for VS_OPEN
    defender_doc, vs_key = _load_vs_open_doc_any(
        ip_defender=oop,          # defender is OOP
        oop_opener=ip,            # opener is IP
        stack_bb=stack_bb,
        profile=villain_profile,
        exploit=exploit_setting,
        multiway=multiway_context,
        pop=population_type
    )
    oop_range = _collect_actions(defender_doc, VS_OPEN_DEFENDER_ACTIONS)

    # Helpful breadcrumbs
    print(f"    • OPEN used: action=OPEN/{ip}_vs_{oop}_{stack_bb}bb.json.gz")
    print(f"    • VS_OPEN used: {vs_key}")
    print(f"    • IP_range={len(ip_range)} OOP_range={len(oop_range)}")
    return ip_range, oop_range