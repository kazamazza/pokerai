# ml/etl/popnet_normalize.py
from typing import List, Dict, Iterable
import math

POS2ID = {"SB":0,"BB":1,"UTG":2,"HJ":3,"CO":4,"BTN":5}
STREET2ID = {"preflop":0,"flop":1,"turn":2,"river":3}
ACT2ID = {"fold":0,"check":1,"call":2,"bet":3,"raise":4,"allin":5}

def _safe(actor: str, position_by_player: Dict[str,str]) -> int:
    pos = position_by_player.get(actor)
    return POS2ID.get(pos, -1)  # -1 if unknown; you can skip those rows if you like

def _log_clip(x: float, max_bb: float = 100.0) -> float:
    # cap extreme values and stabilize
    if x is None:
        return 0.0
    x = max(0.0, min(float(x), max_bb))
    return math.log1p(x)

def normalize_hand_to_rows(hand: Dict) -> Iterable[Dict]:
    """
    Turns one parsed hand into a sequence of supervised rows.
    Each row = one decision event by a player.
    """
    bb = float(hand.get("min_bet", 0.0) or 0.0)
    pos_by = hand.get("position_by_player", {})
    for ev in hand.get("actions", []):
        actor = ev["actor"]
        act   = ev["act"]
        if act not in ACT2ID:
            continue
        pos_id = _safe(actor, pos_by)
        if pos_id < 0:
            continue  # skip unknown position (should be rare)
        feat = {
            "stake_bb": bb if bb > 0 else 0.1,
            "position_id": pos_id,
            "street_id": STREET2ID.get(ev["street"], 0),
            "stack_bb": None,                # optional: fill later when you track stacks
            "pot_bb": hand.get("pot_bb", 0.0),
            "amount_bb": _log_clip(ev.get("amount_bb", 0.0)),
        }
        lab = {"action_id": ACT2ID[act]}
        yield {"x": feat, "y": lab}