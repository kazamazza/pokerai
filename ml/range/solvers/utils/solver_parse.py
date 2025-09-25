from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# -------- core accessors --------

def root_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("root", payload) if isinstance(payload, dict) else {}

def get_children(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the children mapping for a node.
    Accepts either 'childrens' (TexasSolver) or 'children' (some variants).
    Always returns a dict (or {} if missing/invalid).
    """
    if not isinstance(node, dict):
        return {}
    ch = node.get("childrens")
    if not isinstance(ch, dict):
        ch = node.get("children")
    return ch if isinstance(ch, dict) else {}

def actions_and_mix(node: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """
    Return (actions, mix) where actions are strings and mix sums to 1 (if available).
    Falls back to [] for mix if no per-hand strategy is present.
    """
    acts = list(node.get("actions") or [])
    strat = node.get("strategy") or {}

    # Some dumps mirror actions under strategy.actions
    s_acts = list((strat.get("actions") or []))
    if not acts and s_acts:
        acts = s_acts

    # Aggregate per-hand strategy → action mass
    strat_map = strat.get("strategy") or {}
    k = len(acts)
    if not k or not isinstance(strat_map, dict) or not strat_map:
        return acts, []

    mass = [0.0] * k
    nrows = 0
    for probs in strat_map.values():
        if not isinstance(probs, list):
            continue
        L = min(len(probs), k)
        if L <= 0:
            continue
        for i in range(L):
            v = probs[i]
            if v is not None and v >= 0:
                mass[i] += float(v)
        nrows += 1

    if nrows == 0:
        return acts, []

    s = sum(mass)
    if s > 0:
        mass = [m / s for m in mass]
    return acts, mass

# -------- helpers for BET aggregation --------

def _iter_bet_children_with_weight(node: Dict[str, Any]):
    """
    Yield (bet_label, child_node, root_weight) for every BET child at this node.
    root_weight is the strategy mass on that bet at the current node (0 if unknown).
    """
    acts, mix = actions_and_mix(node)
    ch = get_children(node)
    have_mix = bool(mix) and len(mix) == len(acts)

    # First pass: use actions[] indices (preferred)
    for i, lab in enumerate(acts):
        up = str(lab).upper()
        if up.startswith("BET"):
            child = ch.get(lab) or ch.get(str(lab))
            if isinstance(child, dict):
                w = float(mix[i]) if have_mix else 0.0
                yield str(lab), child, w

    # Fallback: scan dict keys in case some BET child isn't listed in actions[]
    for k, v in ch.items():
        up = str(k).upper()
        if up.startswith("BET") and isinstance(v, dict):
            # avoid double-yield if it was already yielded via actions[]
            # (string compare with acts)
            if not any(str(k) == str(a) for a in acts if str(a).upper().startswith("BET")):
                yield str(k), v, 0.0

def _normalize(vec: Dict[str, float]) -> Dict[str, float]:
    s = sum(vec.values())
    if s > 0:
        inv = 1.0 / s
        for k in list(vec.keys()):
            vec[k] *= inv
    return vec

# -------- high-level extractors --------

def extract_ip_root_decision(payload: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """
    IP at root: return (labels, mix) over CHECK and BET sizes at the root node.
    """
    root = root_node(payload)
    acts, mix = actions_and_mix(root)
    # Normalize defensively if solver emitted near-zero error
    if mix and abs(sum(mix) - 1.0) > 1e-6:
        s = sum(mix)
        if s > 0:
            mix = [m / s for m in mix]
    return acts, mix

def extract_oop_facing_bet(payload: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """
    OOP vs a bet at root: aggregate ALL bet-children.
    For each BET child c with root mass w_root, accumulate w_root * c.strategy over
    the child actions (CALL/FOLD/RAISE*/ALLIN). Return merged, normalized (labels, mix).
    """
    root = root_node(payload)

    # Accumulate mass per label across bet-children
    acc: Dict[str, float] = defaultdict(float)
    total_root_mass = 0.0
    any_child = False

    for bet_label, child, w_root in _iter_bet_children_with_weight(root):
        any_child = True
        # If root had no mix, treat each discovered BET child equally
        # (split uniformly). This keeps us from discarding branches blindly.
        if w_root <= 0.0:
            w_root = 1.0  # temporary; we’ll renorm over children below
        acts, mix = actions_and_mix(child)
        if not acts or not mix:
            continue
        # ensure child mix sums to 1
        s_child = sum(mix)
        if s_child > 0 and abs(s_child - 1.0) > 1e-6:
            mix = [m / s_child for m in mix]

        total_root_mass += w_root
        for i, a in enumerate(acts):
            acc[str(a).upper()] += w_root * float(mix[i])

    if not any_child or not acc:
        return [], []

    # If we fabricated equal weights (root had no usable mix), make them uniform
    # by dividing by the number of used bet-children.
    if total_root_mass <= 0.0:
        # Count how many bet-children actually contributed
        n_children = 0
        for _ in _iter_bet_children_with_weight(root):
            n_children += 1
        if n_children > 0:
            scale = 1.0 / n_children
            for k in list(acc.keys()):
                acc[k] *= scale

    acc = _normalize(acc)

    labels = list(acc.keys())
    mix = [acc[k] for k in labels]
    return labels, mix