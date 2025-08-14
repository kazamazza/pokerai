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

# ---- tiny loader wrappers (reuse your existing load_preflop_json_from_s3) ----
def _load_pf(ip: str, oop: str, stack_bb: int, profile: str, exploit: str, multiway: str, pop: str, action: str) -> dict:
    return load_preflop_json_from_s3(ip, oop, stack_bb, profile, exploit, multiway, pop, action)

# Which buckets to collect from each file for each context/role.
# NOTE: keep names lowercase; _collect_actions lowercases keys.
CONTEXT_RECIPES = {
    # Single-raised pot (SRP): opener is IP in your clustering stage
    "OPEN": [
        # (action_file_used, buckets_we_want, pair_orientation)
        ("OPEN",    ("open",),                           "ip_oop"),  # opener's range
        ("VS_OPEN", ("call", "3bet", "4bet", "jam"),     "oop_ip"),  # defender's range (all defending lines)
    ],

    # Facing a limp (hero can ISO or over-limp). Opponent is the limper.
    "VS_LIMP": [
        ("VS_LIMP", ("iso", "limp"),                     "ip_oop"),  # hero's options vs limp
        ("VS_ISO",  ("fold", "call", "3bet", "jam"),     "oop_ip"),  # limper's defend vs ISO
    ],

    # Hero (IP) was the opener, faces a 3bet by OOP.
    # Hero's file is VS_3BET; OOP's 3betting range is in VS_OPEN (their "3bet" bucket).
    "VS_3BET": [
        ("VS_3BET", ("fold", "call", "4bet", "jam"),     "ip_oop"),  # hero's responses
        ("VS_OPEN", ("3bet",),                           "oop_ip"),  # defender's 3betting range
    ],

    # Hero (IP) now faces a 4bet. OOP’s 4bet range typically appears in their VS_3BET file.
    "VS_4BET": [
        ("VS_4BET", ("fold", "call", "jam"),             "ip_oop"),  # hero's responses to 4bet
        ("VS_3BET", ("4bet", "jam"),                     "oop_ip"),  # defender's 4bet/jam buckets
    ],

    # Optional: if you also do "VS_ISO" as a primary context
    "VS_ISO": [
        ("VS_ISO",  ("fold", "call", "3bet", "jam"),     "ip_oop"),
        ("VS_LIMP", ("limp",),                           "oop_ip"),  # original limper range
    ],
}

def extract_ip_oop_ranges(
    action_context: str,
    ip: str, oop: str, stack_bb: int,
    villain_profile: str, exploit_setting: str,
    multiway_context: str, population_type: str,
) -> tuple[list[str], list[str]]:
    """
    Generic extractor: loads the correct two preflop files (possibly role-reversed)
    and collects the requested buckets to produce (ip_range, oop_range).
    """
    ctx = action_context.upper()
    if ctx not in CONTEXT_RECIPES:
        raise ValueError(f"Unsupported action_context: {action_context}")

    steps = CONTEXT_RECIPES[ctx]

    # Step A: get IP range
    fileA, bucketsA, orientA = steps[0]
    if orientA == "ip_oop":
        docA = _load_pf(ip, oop, stack_bb, villain_profile, exploit_setting, multiway_context, population_type, fileA)
    else:
        docA = _load_pf(oop, ip, stack_bb, villain_profile, exploit_setting, multiway_context, population_type, fileA)
    ip_range = _collect_actions(docA, set(bucketsA))

    # Step B: get OOP range
    fileB, bucketsB, orientB = steps[1]
    if orientB == "ip_oop":
        docB = _load_pf(ip, oop, stack_bb, villain_profile, exploit_setting, multiway_context, population_type, fileB)
    else:
        docB = _load_pf(oop, ip, stack_bb, villain_profile, exploit_setting, multiway_context, population_type, fileB)
    oop_range = _collect_actions(docB, set(bucketsB))

    return ip_range, oop_range