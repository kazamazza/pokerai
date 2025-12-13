# preflop_seq.py (or near PolicyInfer helpers)

RAISE_ALIASES = {"RAISE", "OPEN", "ISO", "3BET", "4BET"}
LIMP_ALIASES  = {"LIMP"}
CALL_ALIASES  = {"CALL", "FLAT"}
CHECK_ALIASES = {"CHECK"}

def _act(obj) -> str:
    # supports dicts or dataclass-ish objects
    if obj is None: return ""
    a = getattr(obj, "action", None) or getattr(obj, "type", None)
    if a is None and isinstance(obj, dict):
        a = obj.get("action") or obj.get("type")
    return (str(a) if a else "").upper()

def _is_limped_single(hero_pos: str, actions_hist) -> bool:
    # SB limps, no raises, hero=BB yet to act ⇒ limped single
    H = (hero_pos or "").upper()
    if H != "BB" or not actions_hist:
        return False
    seen_raise = any(_act(e) in RAISE_ALIASES for e in actions_hist)
    if seen_raise:
        return False
    # require at least one LIMP before hero
    return any(_act(e) in LIMP_ALIASES for e in actions_hist)

def _raise_count(actions_hist) -> int:
    return sum(1 for e in (actions_hist or []) if _act(e) in RAISE_ALIASES)

def infer_preflop_action_seq(actions_hist, hero_pos: str) -> dict:
    """
    Maps history to sidecar’s discrete tokens:
      action_seq_1 ∈ {LIMP, RAISE}
      action_seq_2 ∈ {3BET, CALL, CHECK}
      action_seq_3 ∈ {4BET, CALL, NONE}
    """
    if _is_limped_single(hero_pos, actions_hist):
        return {"action_seq_1": "LIMP",  "action_seq_2": "CHECK", "action_seq_3": "NONE"}

    r = _raise_count(actions_hist)
    if r == 0:
        return {"action_seq_1": "RAISE", "action_seq_2": "CHECK", "action_seq_3": "NONE"}   # unopened SRP
    if r == 1:
        return {"action_seq_1": "RAISE", "action_seq_2": "CALL",  "action_seq_3": "NONE"}   # vs open
    if r == 2:
        return {"action_seq_1": "RAISE", "action_seq_2": "3BET",  "action_seq_3": "NONE"}   # vs 3bet
    return {"action_seq_1": "RAISE", "action_seq_2": "3BET",  "action_seq_3": "4BET"}      # 4bet+