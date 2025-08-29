# remapper.py
from typing import Tuple

# canonical 6-max positions in order of distance from BTN
POS6_ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
POS6_INDEX = {p: i for i, p in enumerate(POS6_ORDER)}

def remap_pair_6max(ip: str, oop: str, available_pairs: set[Tuple[str,str]]) -> Tuple[str, str, bool]:
    """
    Try to remap a 6-max (ip, oop) pair to the closest available pair.
    - Always preserves OOP.
    - Only shifts IP to nearest neighbour(s) in POS6_ORDER.
    Returns (ip_mapped, oop, substituted_flag).
    """

    ip = ip.upper(); oop = oop.upper()
    # if exact pair exists, no remap
    if (ip, oop) in available_pairs:
        return ip, oop, False

    if ip not in POS6_INDEX or oop not in POS6_INDEX:
        return ip, oop, False  # unknown positions, nothing to do

    target_idx = POS6_INDEX[ip]
    # search outward from target index: ±1, ±2, ...
    for delta in range(1, len(POS6_ORDER)):
        for direction in (-1, +1):
            cand_idx = target_idx + direction * delta
            if 0 <= cand_idx < len(POS6_ORDER):
                cand_ip = POS6_ORDER[cand_idx]
                if (cand_ip, oop) in available_pairs:
                    return cand_ip, oop, True

    # nothing found
    return ip, oop, False