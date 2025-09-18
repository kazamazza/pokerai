from __future__ import annotations
import re, math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------- 169 grid helpers (replace with your canonical indexer if available) ---------
_RANKS = "AKQJT98765432"
def _is_suited(h: str) -> bool: return h.endswith("s")
def _is_offsuit(h: str) -> bool: return h.endswith("o")
def _card_ranks(h: str) -> Tuple[str, str]:
    # Accept AA, AKs, KQo, 9Ts, etc.
    h = h.strip()
    if len(h) == 2:  # pairs like "AA"
        return h[0], h[1]
    if len(h) == 3:  # AKs / AKo
        return h[0], h[1]
    # Very loose fallback (e.g. "AhKd"): collapse to rank-only like "AKo" by suit relation
    if len(h) == 4:
        r1, s1, r2, s2 = h[0], h[1], h[2], h[3]
        if r1 == r2: return r1, r2
        if s1 == s2: return r1, r2  # suited
        return r1, r2
    return "", ""

def hand_to_index_169(hand: str) -> Optional[int]:
    """
    Map text like 'AA','AKs','KQo','22' to [0..168].
    Uses a standard 13x13 upper-tri layout (pairs on diagonal).
    """
    h = hand.strip()
    if not h: return None

    # Normalize to rank-rank + suit tag
    suited = None
    if len(h) == 2:  # pairs
        r1, r2 = h[0], h[1]
        suited = None
    elif len(h) == 3:  # AKs / AKo
        r1, r2 = h[0], h[1]
        suited = h[2].lower()
    else:
        # Try collapsing AhKd → AKo / AhKh → AKs depending on suit relation
        if len(h) == 4:
            r1, s1, r2, s2 = h[0], h[1], h[2], h[3]
            suited = "s" if s1 == s2 else "o"
        else:
            return None

    if r1 not in _RANKS or r2 not in _RANKS: return None
    i = _RANKS.index(r1)
    j = _RANKS.index(r2)

    # 13x13 index rules (row-major):
    # pairs on diagonal i==j
    # for non-pairs: suited hands live in lower triangle (i>j), offsuit in upper (i<j)
    if i == j:
        row, col = i, j
    else:
        if suited == "s":
            # suited: put higher rank first (i < j means r1 higher in our ordering)
            if i > j: i, j = j, i
            row, col = j, i  # lower triangle
        elif suited == "o":
            if i < j: i, j = j, i
            row, col = i, j  # upper triangle
        else:
            # unknown suitedness -> cannot place reliably
            return None

    idx = row * 13 + col
    if 0 <= idx < 169: return idx
    return None

# --------- node / action helpers ----------
def _root(js: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("root", "tree", "nodes", "graph"):
        if isinstance(js.get(k), dict):
            # "nodes" shape may be mapping of id->node; try "root" id
            if k == "nodes" and "root" in js[k] and isinstance(js[k]["root"], dict):
                return js[k]["root"]
            if k in ("root", "tree"):
                return js[k]
    # last resort: assume top-level is a node
    return js

def _children_map(node: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("childrens", "children", "edges"):
        if isinstance(node.get(k), dict):
            return node[k]
    return {}

def _strategy_map_on_node(node: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    """
    Return a dict: hand -> list of action probs (in node's action order), if present.
    Accepts node['strategy']['strategy'] or node['strategy'] directly.
    """
    strat = node.get("strategy")
    if isinstance(strat, dict) and "strategy" in strat and isinstance(strat["strategy"], dict):
        return strat["strategy"]
    if isinstance(strat, dict):
        # sometimes it's directly the map
        if all(isinstance(v, (list, tuple)) for v in strat.values()):
            return strat  # type: ignore
    return None

def _action_list_on_node(node: Dict[str, Any]) -> List[str]:
    """
    If node has explicit action order, try to read it; otherwise derive from child names.
    """
    for k in ("actions", "action_labels", "action_list"):
        if isinstance(node.get(k), list) and all(isinstance(x, str) for x in node[k]):
            return node[k]
    # derive from children keys (stable sort)
    ch = _children_map(node)
    return list(ch.keys())

def _fuzzy_find_action_index(action_names: List[str], target_prefix: str) -> Optional[int]:
    pat = re.compile(rf"^{re.escape(target_prefix)}\b", flags=re.IGNORECASE)
    for i, nm in enumerate(action_names):
        if isinstance(nm, str) and pat.match(nm.strip()):
            return i
    # fallback: look for substring
    pat2 = re.compile(re.escape(target_prefix), flags=re.IGNORECASE)
    for i, nm in enumerate(action_names):
        if isinstance(nm, str) and pat2.search(nm):
            return i
    return None

def _vector_from_node_strategy_for_action(node: Dict[str, Any], action_prefix: str) -> List[float]:
    """
    Parent-centric: node's strategy is a map hand -> vector over actions.
    We pick the probability for the action that matches action_prefix.
    """
    strat_map = _strategy_map_on_node(node)
    if strat_map is None:
        return [0.0] * 169
    actions = _action_list_on_node(node)
    a_idx = _fuzzy_find_action_index(actions, action_prefix)
    if a_idx is None:
        return [0.0] * 169

    vec = [0.0] * 169
    for h, probs in strat_map.items():
        try:
            idx = hand_to_index_169(h)
            if idx is None: continue
            p = float(probs[a_idx]) if (isinstance(probs, (list, tuple)) and len(probs) > a_idx) else 0.0
            if not math.isnan(p):
                vec[idx] = p
        except Exception:
            continue
    return vec

def _vector_from_child_node_strategy(child_node: Dict[str, Any]) -> List[float]:
    """
    Child-centric fallback: some dumps place a per-hand vector on the *child*.
    We take the 2nd component if it's a 2-action node (bet/continue), else last.
    """
    strat_map = _strategy_map_on_node(child_node)
    if strat_map is None:
        return [0.0] * 169

    # choose a component index heuristically
    comp_idx = 1  # prefer the 'take-this-edge' prob if two choices
    vec = [0.0] * 169
    for h, probs in strat_map.items():
        try:
            idx = hand_to_index_169(h)
            if idx is None: continue
            if isinstance(probs, (list, tuple)) and len(probs) > 0:
                j = comp_idx if comp_idx < len(probs) else (len(probs) - 1)
                p = float(probs[j])
                if not math.isnan(p):
                    vec[idx] = p
        except Exception:
            continue
    return vec

def _find_child(node: Dict[str, Any], action_prefix: str) -> Optional[Dict[str, Any]]:
    ch = _children_map(node)
    if not ch: return None
    # strong match: key starts with "BET", "CHECK", ...
    for k, v in ch.items():
        if isinstance(k, str) and re.match(rf"^{re.escape(action_prefix)}\b", k, flags=re.IGNORECASE):
            return v if isinstance(v, dict) else None
    # weak match: contains substring
    for k, v in ch.items():
        if isinstance(k, str) and action_prefix.lower() in k.lower():
            return v if isinstance(v, dict) else None
    return None

# --------- public extractors ----------
def extract_root_bet_vector169(js: Dict[str, Any]) -> List[float]:
    """
    Probability of taking a 'BET' from the *root node* for each hand.
    """
    root = _root(js)
    # Preferred: parent strategy at root, pick BET component
    v = _vector_from_node_strategy_for_action(root, "BET")
    if any(x > 0 for x in v):
        return v
    # Fallback: child-centric (root -> BET child has per-hand map)
    bet_child = _find_child(root, "BET")
    if bet_child:
        v2 = _vector_from_child_node_strategy(bet_child)
        if any(x > 0 for x in v2):
            return v2
    return [0.0] * 169

def extract_root_donk_vector169(js: Dict[str, Any]) -> List[float]:
    """
    Alias for 'BET at root' (used when root actor is OOP and can donk).
    """
    return extract_root_bet_vector169(js)

def extract_ip_cbet_after_oop_check_vector169(js: Dict[str, Any]) -> List[float]:
    """
    Follow root 'CHECK' (by OOP), then extract 'BET' on the next node (IP c-bet).
    """
    root = _root(js)

    # Step 1: find the CHECK child of root
    chk = _find_child(root, "CHECK")
    if not chk:
        # parent-centric root may encode a prob for CHECK; but we need the *next* node.
        # If no explicit child, we cannot traverse -> return zeros.
        return [0.0] * 169

    # Step 2: at that child node (now IP to act), extract BET
    v = _vector_from_node_strategy_for_action(chk, "BET")
    if any(x > 0 for x in v):
        return v

    bet_child = _find_child(chk, "BET")
    if bet_child:
        v2 = _vector_from_child_node_strategy(bet_child)
        if any(x > 0 for x in v2):
            return v2

    return [0.0] * 169

# --------- high-level, action-conditioned extractor (169) ---------
def extract_action_vector_169(
    js: Dict[str, Any],
    *,
    actor: str,              # 'ip' | 'oop' (whose strategy we want at this point)
    node_key: str = "root",  # which node to read; default root
    action_prefix: str = "BET"  # 'BET' | 'DONK' (DONK == OOP bet at root)
) -> Optional[np.ndarray]:
    """
    Returns a 169-length numpy vector with the probability of taking `action_prefix`
    for each hand at `node_key`, from the point-of-view of `actor`.

    Root heuristics:
      - OOP DONK at root → use root BET vector (donk == OOP bet into IP).
      - IP BET at root   → assume OOP checked first, then IP c-bets (CHECK→BET).
      - OOP BET at root  → use root BET vector.

    Non-root:
      - Try to read the node directly (parent-centric & child-centric fallbacks).
    """
    actor = (actor or "").lower()
    action_prefix = (action_prefix or "BET").upper()

    # Non-root path: read node directly when possible
    nodes = js.get("nodes")
    if node_key and node_key != "root" and isinstance(nodes, dict) and node_key in nodes:
        node = nodes[node_key]
        v = _vector_from_node_strategy_for_action(node, action_prefix)
        if any(x > 0 for x in v):
            return np.asarray(v, dtype=np.float32)
        child = _find_child(node, action_prefix)
        if child:
            v2 = _vector_from_child_node_strategy(child)
            if any(x > 0 for x in v2):
                return np.asarray(v2, dtype=np.float32)
        return None  # no signal here

    # Root path: apply simple, robust heuristics
    if action_prefix == "DONK":
        # OOP donk at root is just "BET at root" from OOP
        v = extract_root_bet_vector169(js)
        return np.asarray(v, dtype=np.float32) if any(x > 0 for x in v) else None

    if action_prefix == "BET":
        if actor == "ip":
            # IP c-bet after OOP checks
            v = extract_ip_cbet_after_oop_check_vector169(js)
            if any(x > 0 for x in v):
                return np.asarray(v, dtype=np.float32)
            # fallback: some dumps encode parent-centric root BET even for IP
            v2 = extract_root_bet_vector169(js)
            return np.asarray(v2, dtype=np.float32) if any(x > 0 for x in v2) else None
        else:
            # OOP betting at root (e.g., single-raised probes or weird trees)
            v = extract_root_bet_vector169(js)
            return np.asarray(v, dtype=np.float32) if any(x > 0 for x in v) else None

    # Unknown action tag → try generic root BET as the safest fallback
    v = extract_root_bet_vector169(js)
    return np.asarray(v, dtype=np.float32) if any(x > 0 for x in v) else None