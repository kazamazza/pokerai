import json
import os

from utils.combos import get_169_combo_list
from utils.equity import compute_hand_vs_range_equity
from utils.range_utils import expand_range_syntax


def get_range_string_for_config(position, stack_bb, profile, exploit, multiway, action, population) -> str:
    """
    Compute hero's preflop range string based on full configuration and equity simulation.
    """
    all_combos = get_169_combo_list()

    villain_range_str = get_villain_range_string(
        position=position,
        stack_bb=stack_bb,
        villain_profile=profile,
        exploit_setting=exploit,
        multiway_context=multiway,
        action_context=action,
        population_type=population
    )
    villain_combos = expand_range_syntax(villain_range_str)

    if not villain_combos:
        raise ValueError(f"❌ No combos in villain range: {villain_range_str}")

    threshold = get_equity_threshold(profile, exploit, multiway)
    selected_combos = []

    for combo in all_combos:
        equity = compute_hand_vs_range_equity(combo, villain_combos)
        if equity >= threshold:
            selected_combos.append(combo)

    return ",".join(selected_combos)


# Optional: cache the ranges after first load
_range_cache = None

def get_villain_range_string(
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    action_context: str,
    population_type: str,
    range_path: str = "data/preflop_range_templates.json"
) -> str:
    """
    Retrieve the villain's range string based on full configuration.
    Looks up the range from a precomputed JSON file.

    Args:
        position (str): Hero position (e.g. 'UTG')
        stack_bb (int): Stack depth in big blinds (e.g. 50)
        villain_profile (str): e.g. 'TAG', 'LAG', etc.
        exploit_setting (str): e.g. 'GTO', 'EXPLOIT_LIGHT'
        multiway_context (str): e.g. 'HU', '3WAY'
        action_context (str): e.g. 'VS_OPEN', 'VS_3BET'
        population_type (str): e.g. 'REGULAR', 'RECREATIONAL'
        range_path (str): Path to the range template JSON file

    Returns:
        str: Range string in shorthand format (e.g. '22+,A5s+,KTs+,...')

    Raises:
        KeyError: if key is not found in the JSON file
        FileNotFoundError: if the template file is missing
    """
    global _range_cache

    if _range_cache is None:
        if not os.path.exists(range_path):
            raise FileNotFoundError(f"❌ Range file not found: {range_path}")
        with open(range_path, "r") as f:
            _range_cache = json.load(f)

    key = f"{position}|{stack_bb}|{villain_profile}|{exploit_setting}|{multiway_context}|{action_context}|{population_type}"

    try:
        return _range_cache[key]
    except KeyError:
        raise KeyError(f"❌ No range found for key: {key}")

def get_equity_threshold(villain_profile, exploit_setting, multiway_context) -> float:
    """
    Dynamic threshold used to include hands based on simulation equity.
    """
    base = 0.50

    if exploit_setting == "EXPLOIT_LIGHT":
        base -= 0.02
    elif exploit_setting == "EXPLOIT_HEAVY":
        base -= 0.05

    if multiway_context == "3WAY":
        base += 0.02
    elif multiway_context == "4WAY_PLUS":
        base += 0.05

    if villain_profile == "MANIAC":
        base -= 0.02
    elif villain_profile == "NIT":
        base += 0.02

    return max(0.40, min(base, 0.70))