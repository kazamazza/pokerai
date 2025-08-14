import json
import os
import re
from itertools import product
from typing import Dict

from features.types import STACK_BUCKETS, VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, ACTION_CONTEXTS, \
    POPULATION_TYPES, POSITIONS
from utils.canonicalize_range_string import canonicalize_range_string
from utils.range_utils import get_stack_bucket_label

OUTPUT_PATH = "data/villain_range_map.json"


def build_base_range(position: str, stack_label: str) -> str:
    """
    Assign a GTO-like positional base range based on position and stack depth.
    """
    templates = {
        "UTG": {
            "ultra_short": "66+,ATs+,AJo+",
            "short":       "55+,A9s+,ATo+,KQo",
            "mid":         "44+,A8s+,KTs+,AJo+",
            "deep":        "33+,A7s+,KTs+,QTs+,ATo+,KQo",
            "standard":    "33+,A7s+,KTs+,QTs+,ATo+,KQo",
            "deepstack":   "22+,A5s+,KTs+,QTs+,JTs,ATo+,KQo",
            "very_deep":   "22+,A5s+,KTs+,QTs+,JTs,ATo+,KQo"
        },
        "MP": {
            "ultra_short": "55+,ATs+,AJo+",
            "short":       "44+,A9s+,KTs+,ATo+,KQo",
            "mid":         "33+,A8s+,KTs+,QTs+,AJo+",
            "deep":        "22+,A7s+,KTs+,QTs+,ATo+,KQo",
            "standard":    "22+,A6s+,KTs+,QTs+,JTs,ATo+,KQo",
            "deepstack":   "22+,A5s+,KTs+,QTs+,JTs,ATo+,KQo",
            "very_deep":   "22+,A4s+,KTs+,QTs+,JTs,ATo+,KQo"
        },
        "CO": {
            "ultra_short": "44+,A8s+,KTs+,ATo+",
            "short":       "33+,A7s+,KTs+,QTs+,A9o+,KQo",
            "mid":         "22+,A6s+,KTs+,QTs+,JTs,A9o+,KTo+",
            "deep":        "22+,A5s+,K9s+,QTs+,JTs,A8o+,KTo+",
            "standard":    "22+,A5s+,K9s+,Q9s+,JTs,T9s,A8o+,KTo+",
            "deepstack":   "22+,A4s+,K8s+,Q9s+,J9s+,T9s,A8o+,KTo+",
            "very_deep":   "22+,A4s+,K8s+,Q9s+,J9s+,T9s,98s,A8o+,KTo+"
        },
        "BTN": {
            "ultra_short": "22+,A6s+,K9s+,Q9s+,A8o+,KTo+",
            "short":       "22+,A4s+,K8s+,Q8s+,J9s+,A7o+,KTo+",
            "mid":         "22+,A2s+,K8s+,Q8s+,J9s+,T9s,A2o+,K9o+",
            "deep":        "22+,A2s+,K7s+,Q8s+,J8s+,T8s+,A2o+,K9o+,QTo+",
            "standard":    "22+,A2s+,K7s+,Q8s+,J8s+,T8s+,98s,A2o+,K9o+,QTo+",
            "deepstack":   "22+,A2s+,K6s+,Q7s+,J8s+,T8s+,98s,87s,A2o+,K9o+,Q9o+",
            "very_deep":   "22+,A2s+,K6s+,Q7s+,J7s+,T7s+,97s+,87s,A2o+,K9o+,Q9o+,JTo"
        },
        "SB": {
            "ultra_short": "22+,A2s+,K9s+,Q9s+,J9s+,A8o+,KTo+",
            "short":       "22+,A2s+,K7s+,Q8s+,J8s+,T8s+,A7o+,K9o+,Q9o+",
            "mid":         "22+,A2s+,K6s+,Q7s+,J8s+,T8s+,A6o+,K8o+,Q9o+",
            "deep":        "22+,A2s+,K5s+,Q7s+,J7s+,T8s+,98s,A5o+,K8o+,Q8o+",
            "standard":    "22+,A2s+,K5s+,Q7s+,J7s+,T8s+,97s+,87s,A5o+,K8o+,Q8o+",
            "deepstack":   "22+,A2s+,K4s+,Q6s+,J7s+,T7s+,97s+,87s,A4o+,K8o+,Q8o+",
            "very_deep":   "22+,A2s+,K4s+,Q6s+,J7s+,T7s+,97s+,86s+,A4o+,K8o+,Q8o+"
        },
        "BB": {
            "ultra_short": "22+,A2s+,K9s+,Q9s+,J9s+,A9o+,KTo+,QJo",
            "short":       "22+,A2s+,K8s+,Q8s+,J8s+,T8s+,A8o+,K9o+,Q9o+",
            "mid":         "22+,A2s+,K7s+,Q7s+,J7s+,T8s+,97s+,A7o+,K9o+,Q8o+",
            "deep":        "22+,A2s+,K6s+,Q6s+,J7s+,T7s+,97s+,87s,A6o+,K9o+,Q8o+",
            "standard":    "22+,A2s+,K6s+,Q6s+,J7s+,T7s+,97s+,86s+,A6o+,K9o+,Q8o+",
            "deepstack":   "22+,A2s+,K5s+,Q6s+,J6s+,T7s+,96s+,86s+,A5o+,K9o+,Q8o+",
            "very_deep":   "22+,A2s+,K5s+,Q6s+,J6s+,T6s+,96s+,86s+,A5o+,K9o+,Q8o+"
        },
    }

    return templates.get(position, {}).get(stack_label, "66+,ATs+,AJo+")


def apply_profile_adjustments(base: str, profile: str) -> str:
    if profile == "NIT":
        return "QQ+,AKs,AKo"
    elif profile == "TAG":
        return base
    elif profile == "LAG":
        return base + ",K9s+,QJo+"
    elif profile == "MANIAC":
        return "22+,A2s+,K2s+,Q2s+,J2s+,A2o+,K2o+"
    elif profile == "FISH":
        return base + ",Q9o+,J9o+,T9o"
    return base  # GTO


def apply_exploit_adjustments(base: str, exploit: str) -> str:
    if exploit == "EXPLOIT_LIGHT":
        return base + ",65s,54s"
    elif exploit == "EXPLOIT_HEAVY":
        return base + ",64s,53s,43s"
    return base


def apply_multiway_adjustments(base: str, multiway: str) -> str:
    if multiway == "3WAY":
        return base.replace("KJo", "").replace("QJo", "")
    elif multiway == "4WAY_PLUS":
        return base.replace("KJo", "").replace("QJo", "").replace("A9o", "")
    return base


def apply_action_context_adjustments(base: str, position: str, action_context: str) -> str:
    # Example dummy logic — replace with real GTO/solver outputs
    if action_context == "OPEN":
        return base
    elif action_context == "VS_OPEN":
        # Narrow calling/3betting range vs open
        return "99+,AQs+,AKo"
    elif action_context == "VS_3BET":
        # Defend tighter vs 3bet
        return "JJ+,AKs,AKo"
    elif action_context == "VS_LIMP":
        # Widen vs limp
        return base + ",Q9o+,J9o+,T9o"
    return base


def apply_population_adjustments(base: str, pop: str) -> str:
    if pop == "RECREATIONAL":
        return base + ",J9s,T8s"
    return base

def build_range_string(
    position: str,
    stack: int,
    profile: str,
    exploit: str,
    multiway: str,
    action: str,
    population: str
) -> str:
    stack_label = get_stack_bucket_label(stack)

    base = build_base_range(position, stack_label)
    base = apply_profile_adjustments(base, profile)
    base = apply_exploit_adjustments(base, exploit)
    base = apply_multiway_adjustments(base, multiway)
    base = apply_population_adjustments(base, population)
    base = apply_action_context_adjustments(base, position, action)

    return canonicalize_range_string(base)


def generate_villain_range_map():
    result: Dict[str, str] = {}

    for (
        position,
        stack,
        profile,
        exploit,
        multiway,
        action,
        population
    ) in product(
        POSITIONS,
        STACK_BUCKETS,
        VILLAIN_PROFILES,
        EXPLOIT_SETTINGS,
        MULTIWAY_CONTEXTS,
        ACTION_CONTEXTS,
        POPULATION_TYPES
    ):
        key = f"{position}|{stack}|{profile}|{exploit}|{multiway}|{action}|{population}"
        range_str = build_range_string(position, stack, profile, exploit, multiway, action, population)
        result[key] = range_str

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Saved villain range map to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_villain_range_map()
