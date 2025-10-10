import json, gzip, re
from typing import Any, Dict, List, Tuple, Sequence, Optional

# --- vocab (unchanged) ---
ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500",
    "ALLIN",
]

FOLD_RE  = re.compile(r"\bfold\b", re.IGNORECASE)
CALL_RE  = re.compile(r"\bcall\b", re.IGNORECASE)
CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)
BET_RE   = re.compile(r"\bbet\b", re.IGNORECASE)
RAISE_RE = re.compile(r"\braise\b", re.IGNORECASE)
ALLIN_RE = re.compile(r"\ball[-\s]*in\b|\bjam\b", re.IGNORECASE)

Json = Dict[str, Any]

# ======================================================================
# Public entrypoint
# ======================================================================

def parse_solver_simple(
    path: str,
    *,
    facing_bet: bool = False,
    **_,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Schema-tolerant parser for TexasSolver JSON(.gz).
    Two modes:
      - facing_bet=False: extract strategy at root (first-to-act)
      - facing_bet=True : follow first BET edge (or CHECK->BET), descend through chance nodes,
                          then extract strategy at the first action node.
    Returns (probs_map, meta).
    """
    data = _open_json_any(path)
    root = data.get("root", data)
    if not isinstance(root, dict):
        return {}, {"error": "malformed_root"}

    if not facing_bet:
        acts, mix, where = _find_any_strategy(root)
        if not acts:
            return {}, {"error": "no_strategy_found_anywhere"}
        probs = _map_to_vocab(acts, mix)
        return probs, {"level": "root", "found_at": where, "actions": acts}

    # Facing bet: locate BET node (root->BET or root->CHECK->BET or shallow DFS),
    # then descend through chance nodes to first action node that holds strategy.
    bet_node, path_labels = _find_first_bet_node(root)
    if bet_node is None:
        return {}, {"error": "no_bet_edge_found"}

    action_node, descend_path = _descend_to_first_action_node(bet_node, prefix_path=path_labels)
    if action_node is None:
        return {}, {"error": "no_action_node_under_bet", "via": " -> ".join(path_labels)}

    acts, mix, where = _find_any_strategy(action_node)
    if not acts:
        return {}, {"error": "no_strategy_at_bet_action_node", "via": " -> ".join(descend_path)}
    probs = _map_to_vocab(acts, mix)
    return probs, {"level": "facing_bet", "via": " -> ".join(descend_path), "found_at": where, "actions": acts}

# ======================================================================
# Strategy discovery
# ======================================================================

def _find_any_strategy(node: Json) -> Tuple[List[str], List[float], str]:
    """
    Recursively search 'node' for any recognizable strategy encoding.
    Returns (actions, mix, where_string).
    Order:
      1) node itself
      2) node.data
      3) read per-child weights if present (prob/weight/frequency)
      4) recurse into children
    """
    acts, mix = _extract_strategy_any(node)
    if acts:
        return acts, mix, "node.strategy"

    data = node.get("data")
    if isinstance(data, dict):
        acts, mix = _extract_strategy_any(data)
        if acts:
            return acts, mix, "node.data.strategy"

    # Per-child weights (rare, but support it)
    children = _normalize_children(node)
    if children:
        acts_c, mix_c = [], []
        for lbl, ch in children.items():
            p = _read_child_prob(ch)
            if p is not None:
                acts_c.append(str(lbl))
                mix_c.append(p)
        if acts_c and sum(mix_c) > 0:
            acts_c, mix_c = _renorm(acts_c, mix_c)
            return acts_c, mix_c, "children.weights"

    # Recurse
    if children:
        for lbl, ch in children.items():
            a2, m2, w2 = _find_any_strategy(ch)
            if a2:
                return a2, m2, f"descend[{lbl}].{w2}"

    return [], [], ""

def _extract_strategy_any(node: dict) -> tuple[list[str], list[float]]:
    """
    Returns (actions, mix) for a node, supporting:
      A) {"actions":[...], "strategy":[...]}                      # flat
      B) {"strategy":{"actions":[...], "strategy": {"AhAd":[...], ...}}}  # combo table
      C) {"strategy": {"CALL":0.4,"FOLD":0.6}}                    # map form
      D) fallback: spread evenly over child labels
    """
    s = node.get("strategy")

    # ---- B) dict with actions + per-combo vectors (most powerful) ----
    if isinstance(s, dict) and "actions" in s and "strategy" in s:
        actions = [str(a) for a in s["actions"]]
        strat = s["strategy"]
        k = len(actions)
        if isinstance(strat, dict):
            acc = [0.0] * k
            n = 0
            for v in strat.values():
                if isinstance(v, list) and len(v) == k:
                    for i, x in enumerate(v):
                        acc[i] += float(x)
                    n += 1
            if n > 0:
                mix = [a / n for a in acc]
                return _renorm(actions, mix)[0], _renorm(actions, mix)[1]

        # allow a flat list sitting under s["strategy"]
        if isinstance(strat, list) and len(strat) == k:
            mix = [float(x) for x in strat]
            return _renorm(actions, mix)

    # ---- C) simple map {"CALL": 0.4, "FOLD": 0.6} ----
    if isinstance(s, dict) and s and all(isinstance(v, (int, float)) for v in s.values()):
        actions = [str(a) for a in s.keys()]
        mix = [float(v) for v in s.values()]
        return _renorm(actions, mix)

    # ---- A) flat arrays directly on node ----
    if isinstance(node.get("actions"), list) and isinstance(node.get("strategy"), list):
        actions = [str(a) for a in node["actions"]]
        mix = [float(x) for x in node["strategy"]]
        return _renorm(actions, mix)

    # ---- D) fallback: spread evenly over children labels ----
    children = _normalize_children(node)
    acts = list(children.keys())
    if acts:
        p = 1.0 / len(acts)
        return acts, [p] * len(acts)

    return [], []

# ======================================================================
# Navigation helpers
# ======================================================================

def _normalize_children(node: dict) -> dict[str, dict]:
    """
    TexasSolver uses 'childrens'. Also accept 'children' list/dict forms.
    Return a dict: {label -> child_node}.
    """
    for key in ("childrens", "children"):
        ch = node.get(key)
        if isinstance(ch, dict):
            # keys already labels
            return {str(k): v for k, v in ch.items() if isinstance(v, dict)}
        if isinstance(ch, list):
            out = {}
            for c in ch:
                if isinstance(c, dict):
                    label = c.get("label") or c.get("action") or str(len(out))
                    out[str(label)] = c
            return out
    return {}

def _find_first_bet_node(root: Json) -> Tuple[Optional[Json], List[str]]:
    """Find node reached by a BET edge: root->BET, or root->CHECK->BET, else shallow DFS."""
    kids = _normalize_children(root)

    # direct BET under root
    for lbl, ch in kids.items():
        if "bet" in lbl.lower():
            return ch, [lbl]

    # CHECK -> BET
    for lbl, ch in kids.items():
        if "check" in lbl.lower():
            kids2 = _normalize_children(ch)
            for lbl2, ch2 in kids2.items():
                if "bet" in lbl2.lower():
                    return ch2, [lbl, lbl2]

    # fallback: shallow DFS
    stack = [([], root)]
    visited = set()
    while stack:
        path, node = stack.pop()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        for lbl, ch in _normalize_children(node).items():
            new_path = path + [lbl]
            if "bet" in lbl.lower():
                return ch, new_path
            stack.append((new_path, ch))

    return None, []

def _descend_to_first_action_node(start: Json, prefix_path: List[str]) -> Tuple[Optional[Json], List[str]]:
    """
    After finding a BET edge, TexasSolver often has chains of 'chance_node' dealing cards
    before the next 'action_node'. Walk down until the first node that contains any strategy.
    """
    path = list(prefix_path)
    node = start
    visited = set()

    while isinstance(node, dict):
        nid = id(node)
        if nid in visited:
            break
        visited.add(nid)

        # if this node already has usable strategy, stop
        acts, mix = _extract_strategy_any(node)
        if acts:
            return node, path

        # else step into the single child if this is a chance fan-out (often many children)
        kids = _normalize_children(node)
        if not kids:
            break

        # prefer diving toward action-bearing branches: if any child has 'actions'/'strategy' keys
        next_label = None
        for lbl, ch in kids.items():
            if isinstance(ch, dict) and ("strategy" in ch or "actions" in ch):
                next_label = lbl; break
        if next_label is None:
            # else pick a deterministic first label (to keep behavior stable)
            next_label = sorted(kids.keys())[0]

        node = kids[next_label]
        path.append(next_label)

    return None, path

def _read_child_prob(node: Json) -> Optional[float]:
    """Try to read a weight/prob/frequency stored on a child node."""
    for key in ("prob", "p", "weight", "frequency", "freq", "w"):
        v = node.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    data = node.get("data")
    if isinstance(data, dict):
        for key in ("prob", "p", "weight", "frequency", "freq", "w"):
            v = data.get(key)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
    return None

# ======================================================================
# Mapping to your vocab
# ======================================================================

def _map_to_vocab(actions: List[str], probs: List[float]) -> Dict[str, float]:
    mapped = {a: 0.0 for a in ACTION_VOCAB}
    for lab, p in zip(actions, probs):
        a = str(lab or "").lower().strip()
        if not a:
            continue

        if FOLD_RE.search(a):  mapped["FOLD"]   += p; continue
        if CHECK_RE.search(a): mapped["CHECK"]  += p; continue
        if CALL_RE.search(a):  mapped["CALL"]   += p; continue
        if ALLIN_RE.search(a): mapped["ALLIN"]  += p; continue

        if BET_RE.search(a):
            pct = _extract_size_pct(a)
            tok = _bucket_to_vocab(pct, "BET")
            mapped[tok] += p; continue

        if RAISE_RE.search(a):
            pct = _extract_size_pct(a)
            tok = _bucket_to_vocab(pct, "RAISE")
            mapped[tok] += p; continue

    s = sum(mapped.values()) or 1.0
    for k in mapped:
        mapped[k] /= s
    return mapped

def _extract_size_pct(text: str) -> Optional[float]:
    """
    Tries to recover a size as a % of pot from a label.
    Handles:
      - 'BET 25.000000' (no '%' sign)
      - 'bet 33%'
      - 'raise 400%' (treat as 4x pot)
      - '2x pot'
    """
    t = text.lower()

    # '2x pot' → 200%
    m = re.search(r"(\d+(?:\.\d+)?)\s*x\s*pot", t)
    if m:
        try: return float(m.group(1)) * 100.0
        except Exception: pass

    # explicit percent
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if m:
        try: return float(m.group(1))
        except Exception: pass

    # bare number after BET/RAISE → assume it's % (TexasSolver often prints 'BET 25.000000')
    m = re.search(r"\b(?:bet|raise)\s+(\d+(?:\.\d+)?)", t)
    if m:
        try:
            val = float(m.group(1))
            # Heuristic: if it's <= 600 treat as % bucket; otherwise ignore
            if 0.0 < val <= 600.0:
                return val
        except Exception:
            pass

    return None

def _bucket_to_vocab(pct: Optional[float], prefix: str) -> str:
    if prefix == "BET":
        defaults = "BET_50"; buckets = [25, 33, 50, 66, 75, 100]
    else:
        defaults = "RAISE_300"; buckets = [150, 200, 300, 400, 500]
    if pct is None:
        return defaults
    nearest = min(buckets, key=lambda t: abs(t - pct))
    tok = f"{prefix}_{int(nearest)}"
    return tok if tok in ACTION_VOCAB else defaults

# ======================================================================
# Utils
# ======================================================================

def _open_json_any(path: str) -> Json:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _renorm(acts: List[str], mix: List[float]) -> Tuple[List[str], List[float]]:
    mix = [0.0 if v is None else float(v) for v in mix]
    s = sum(mix)
    if s <= 0:
        if not acts:
            return [], []
        mix = [1.0 / len(acts)] * len(acts)
    else:
        mix = [v / s for v in mix]
    return acts, mix