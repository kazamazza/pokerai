# preflop/validators.py
from typing import Callable, Tuple

# Canonical preflop seat order (earlier opens into later)
SEAT_ORDER = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
IDX = {p: i for i, p in enumerate(SEAT_ORDER)}

def _later(pos_a: str, pos_b: str) -> bool:
    """True if pos_a acts later preflop than pos_b."""
    return IDX[pos_a] > IDX[pos_b]

# ---------- Context validators ----------

def valid_open(ip: str, oop: str) -> bool:
    """
    IP is the opener, OOP is the defender.
    Opener must be earlier in order than defender.
    Special cases:
      - SB can only open into BB (no SB_vs_BTN etc).
      - BB never opens into later seats in SRP.
    """
    ipU, oopU = ip.upper(), oop.upper()

    # SB opens → only vs BB
    if ipU == "SB":
        return oopU == "BB"

    # BB doesn't first-in open into anyone in SRP producer
    if ipU == "BB":
        return False

    # Everyone else: opener must be earlier than defender
    return _later(oopU, ipU)


def valid_vs_open(ip: str, oop: str) -> bool:
    """
    IP is the **defender** facing an open by OOP.
    So OOP must be earlier, IP later (reverse of OPEN file).
      e.g. BTN_vs_CO is valid; CO_vs_BTN is not.
    SB can defend vs BB open is nonsensical (BB doesn’t open SRP).
    BB can defend vs everyone except SB open (there is no SB open into BB here).
    """
    ipU, oopU = ip.upper(), oop.upper()

    if oopU == "SB":
        # a first-in SB 'open' is allowed, but its defender is BB only.
        return ipU == "BB"
    if oopU == "BB":
        return False

    return _later(ipU, oopU)


def valid_vs_3bet(ip: str, oop: str) -> bool:
    """
    Treat as a post-open defense/attack pairing.
    Use the same seating direction as VS_OPEN (defender later than opener)
    because most of your VS_3BET files encode 'hero facing 3bet'.
    This keeps only the common/tough spots.
    """
    return valid_vs_open(ip, oop)


def valid_vs_4bet(ip: str, oop: str) -> bool:
    """Same directional logic as VS_3BET; keep it consistent."""
    return valid_vs_open(ip, oop)


def valid_vs_limp(ip: str, oop: str) -> bool:
    """
    The simple/common low‑stakes case we’re supporting:
      - SB limps, BB acts → encode as IP=BB (acting), OOP=SB (limper)
    You can widen later (e.g., over‑limp spots), but keep v1 strict.
    """
    ipU, oopU = ip.upper(), oop.upper()
    return ipU == "BB" and oopU == "SB"


def valid_vs_iso(ip: str, oop: str) -> bool:
    """
    If you generate VS_ISO, constrain to BTN/CO isolating a limper in EP/MP/CO.
    For now, keep it tight: BTN isolates CO/MP/UTG; CO isolates MP/UTG.
    (Only if you actually produce these charts.)
    """
    ipU, oopU = ip.upper(), oop.upper()
    if ipU == "BTN" and oopU in {"CO", "MP", "UTG"}:
        return True
    if ipU == "CO" and oopU in {"MP", "UTG"}:
        return True
    return False


# Map each action context → validator
VALIDATORS: dict[str, Callable[[str, str], bool]] = {
    "OPEN":     valid_open,
    "VS_OPEN":  valid_vs_open,
    "VS_3BET":  valid_vs_3bet,
    "VS_4BET":  valid_vs_4bet,
    "VS_LIMP":  valid_vs_limp,
    "VS_ISO":   valid_vs_iso,
}