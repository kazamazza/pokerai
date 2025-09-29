import math
from typing import Any, Dict, List, Optional, Tuple, Mapping


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

def actions_and_mix(node: Mapping[str, Any]) -> Tuple[List[str], List[float]]:
    """
    Extract (actions, mix) from a solver node.

    - `actions`: list of action labels as strings
    - `mix`: normalized probabilities (sums ~1.0) or [] if unavailable
    """
    # Actions may be directly on node or inside node["strategy"]["actions"]
    acts: List[str] = list(node.get("actions") or [])
    strat = node.get("strategy") or {}

    if not acts and isinstance(strat, dict):
        s_acts = strat.get("actions") or []
        if s_acts:
            acts = list(s_acts)

    if not acts:
        return [], []

    k = len(acts)

    # Strategy probabilities are usually under node["strategy"]["strategy"]
    strat_map = strat.get("strategy") if isinstance(strat, dict) else None
    if not isinstance(strat_map, dict) or not strat_map:
        return acts, []

    mass = [0.0] * k
    nrows = 0

    for probs in strat_map.values():
        if not isinstance(probs, (list, tuple)):
            continue
        L = min(len(probs), k)
        if L <= 0:
            continue
        for i in range(L):
            v = probs[i]
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv) and fv >= 0.0:
                mass[i] += fv
        nrows += 1

    if nrows == 0:
        return acts, []

    s = sum(mass)
    if s > 1e-12:
        mass = [m / s for m in mass]
    else:
        return acts, []

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


def collect_oop_actions_recursive(
    node: Mapping[str, Any],
    weight: float,
    *,
    pot_bb: float,
    facing_bet_bb: float,
    stack_bb: float | None,
    ACTION_VOCAB: List[str],
    VOCAB_INDEX: Dict[str, int],
    actions_and_mix,           # fn(node) -> (List[str], List[float])
    get_children,              # fn(node) -> Dict[str, Any]
    _resolve_child,            # fn(children, label) -> Mapping[str, Any] | {}
    _has_any,                  # fn(s, *needles) -> bool
    parse_raise_to_bb,         # fn(label, pot_bb, bet_size_bb) -> float | None
    bucket_raise_label,        # fn(label, pot_bb, facing_bet_bb, stack_bb) -> str
) -> List[Tuple[str, float]]:
    """
    DFS from an OOP action node, accumulating soft mass into policy buckets.
    IMPORTANT: when we see a RAISE, we recurse into that child and update facing_bet_bb
               to the *raise-to* size, so re-raises are bucketed correctly.
    Returns list of (bucket_label, mass).
    """
    out: List[Tuple[str, float]] = []
    acts, mix = actions_and_mix(node)
    if not acts or not mix:
        return out

    for a, p in zip(acts, mix):
        w = float(weight) * float(p)
        if w <= 0:
            continue
        up = str(a).strip().upper()

        if up.startswith("CALL"):
            out.append(("CALL", w))
        elif up.startswith("FOLD"):
            out.append(("FOLD", w))
        elif _has_any(up, "ALLIN", "ALL-IN", "JAM"):
            if "ALLIN" in VOCAB_INDEX:
                out.append(("ALLIN", w))
        elif up.startswith("RAISE") or _has_any(up, "RE-RAISE", "RERAISE", "MIN-RAISE", "MINRAISE"):
            # bucket this raise vs current facing bet
            bucket = bucket_raise_label(
                up, pot_bb=pot_bb, facing_bet_bb=facing_bet_bb, stack_bb=stack_bb
            )
            if bucket in VOCAB_INDEX:
                out.append((bucket, w))

            # Recurse into the raise child with updated "facing bet" = raise_to
            ch = _resolve_child(get_children(node), a)
            if ch and isinstance(ch, Mapping):
                raise_to_bb = parse_raise_to_bb(up, pot_bb=pot_bb, bet_size_bb=facing_bet_bb) or facing_bet_bb
                out.extend(
                    collect_oop_actions_recursive(
                        ch, w,
                        pot_bb=pot_bb,
                        facing_bet_bb=raise_to_bb,
                        stack_bb=stack_bb,
                        ACTION_VOCAB=ACTION_VOCAB,
                        VOCAB_INDEX=VOCAB_INDEX,
                        actions_and_mix=actions_and_mix,
                        get_children=get_children,
                        _resolve_child=_resolve_child,
                        _has_any=_has_any,
                        parse_raise_to_bb=parse_raise_to_bb,
                        bucket_raise_label=bucket_raise_label,
                    )
                )
        # else: ignore anything not relevant (e.g., INFO nodes)

    return out