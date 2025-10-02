import gzip
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ml.models.policy_consts import VOCAB_INDEX, VOCAB_SIZE, ACTION_VOCAB

# ---- Regexes ----
BET_PCT_RE   = re.compile(r"\bbet\b\s*([0-9]+(?:\.[0-9]+)?)\s*%+", re.IGNORECASE)
BET_NUM_RE   = re.compile(r"\bbet\b\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)  # absolute bb (no %)
RAISE_NUM_RE = re.compile(r"\braise(?:\s*to)?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

FOLD_RE  = re.compile(r"\bfold\b", re.IGNORECASE)
CALL_RE  = re.compile(r"\bcall\b", re.IGNORECASE)
CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)
BET_RE   = re.compile(r"\bbet\b", re.IGNORECASE)
RAISE_RE = re.compile(r"\braise\b", re.IGNORECASE)
ALLIN_RE = re.compile(r"\ball[-\s]*in\b|\bjam\b", re.IGNORECASE)

Json = Dict[str, Any]

# ---- IO ----
def load_solver(path: str) -> Json:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def root_node(payload: Json) -> Json:
    for key in ("root","node","game_tree","tree","payload"):
        v = payload.get(key)
        if isinstance(v, dict): return v
    return payload

# ---- Tree utils ----
def _child_container(node: Json):
    for key in ("children","childrens","nodes"):
        v = node.get(key)
        if isinstance(v, list): return "list", v
        if isinstance(v, dict): return "dict", v
    return None, None

def get_children(node: Json) -> Dict[str, Json]:
    kind, cont = _child_container(node)
    if kind == "list":
        out: Dict[str, Json] = {}
        for ch in cont:  # type: ignore
            if isinstance(ch, dict):
                lab = ch.get("action") or ch.get("label") or ch.get("name")
                if isinstance(lab, str): out[lab] = ch
        return out
    if kind == "dict":
        return {str(k):v for k,v in cont.items() if isinstance(v, dict)}  # type: ignore
    return {}

def action_list(node: Json) -> List[str]:
    for key in ("actions","available_actions","menu"):
        v = node.get(key)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v  # type: ignore
    return list(get_children(node).keys())

# ---- Optional: extract pot/facing (not present in your dumps; harmless) ----
_NUMERIC_KEYS_POT  = ["pot_bb","total_pot_bb","pot","total_pot","round_pot","potSize","pot_size","current_pot"]
_NUMERIC_KEYS_FACE = ["to_call","call_amount","facing_bet_bb","facing_bet","amount_to_call","last_bet","bet_to_call"]

def _pick_number(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            try:
                v = float(d[k])
                if math.isfinite(v): return v
            except Exception:
                pass
    return None

def _extract_state_from_node(node: Json) -> Tuple[Optional[float], Optional[float]]:
    objs = [node]
    for k in ("state","info","stats","meta","ctx","context"):
        v = node.get(k)
        if isinstance(v, dict): objs.append(v)
    pot = face = None
    for obj in objs:
        pot  = pot  or _pick_number(obj, _NUMERIC_KEYS_POT)
        face = face or _pick_number(obj, _NUMERIC_KEYS_FACE)
    return pot, face

# ---- Label parsing ----
def parse_bet_pct(label: str) -> Optional[float]:
    m = BET_PCT_RE.search(label);
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

def parse_bet_amount_bb(label: str) -> Optional[float]:
    if "%" in label: return None
    m = BET_NUM_RE.search(label)
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

def parse_raise_to_bb(label: str) -> Optional[float]:
    m = RAISE_NUM_RE.search(label)
    if not m: return None
    try: return float(m.group(1))
    except ValueError: return None

# ---- Bucketing ----
def _bucket_bet_pct(pct: Optional[float]) -> Optional[str]:
    if pct is None: return None
    p = round(pct)
    if 22 <= p <= 28: return "BET_25"
    if 29 <= p <= 37: return "BET_33"
    if 45 <= p <= 55: return "BET_50"
    if 60 <= p <= 70: return "BET_66"
    if 71 <= p <= 80: return "BET_75"
    if 90 <= p <= 110: return "BET_100"
    return None

_TARGETS = [1.5, 2.0, 3.0, 4.0, 5.0]

def _nearest_raise_label(rb: float) -> str:
    t = min(_TARGETS, key=lambda x: abs(rb - x))
    if t == 1.5: return "RAISE_150"
    if t == 2.0: return "RAISE_200"
    if t == 3.0: return "RAISE_300"
    if t == 4.0: return "RAISE_400"
    return "RAISE_500"

def _infer_facing(menu_pcts: Sequence[float], pot_before: float, raise_to_bb: Optional[float]) -> float:
    if not menu_pcts or raise_to_bb is None or not math.isfinite(raise_to_bb):
        return 0.33 * pot_before
    best = None
    for p in menu_pcts:
        face = p * pot_before
        rb = raise_to_bb / max(face, 1e-6)
        score = min(abs(rb - t) for t in _TARGETS)
        if best is None or score < best[0]:
            best = (score, face)
    return best[1] if best else 0.33 * pot_before

def _bucket_raise(raise_to_bb: Optional[float], facing_bet_bb: float, pot_before: float, stack_bb: float) -> str:
    if raise_to_bb is None or not math.isfinite(raise_to_bb): return "RAISE_200"
    if raise_to_bb >= (stack_bb - 0.5): return "ALLIN"
    rb = raise_to_bb / max(facing_bet_bb, 1e-6)
    return _nearest_raise_label(rb)

# ---- DFS ----
@dataclass
class State:
    pot_bb: float
    facing_bet_bb: float
    stack_bb: float
    menu_pcts: Sequence[float]

def _bump(counts: List[float], tok: str, inc: float = 1.0) -> None:
    i = VOCAB_INDEX.get(tok)
    if i is not None: counts[i] += inc

def _dfs(node: Json, counts: List[float], st: State, depth: int, max_depth: int,
         raises_log: List[Tuple[str, float, float, float, float]]) -> None:
    if depth > max_depth: return

    kids = get_children(node)
    pot_ov, face_ov = _extract_state_from_node(node)
    base = State(
        pot_bb = pot_ov  if pot_ov  is not None else st.pot_bb,
        facing_bet_bb = face_ov if face_ov is not None else st.facing_bet_bb,
        stack_bb = st.stack_bb,
        menu_pcts = st.menu_pcts,
    )

    def handle(label: str, nxt: State, ch: Optional[Json]):
        al = label.lower()
        pot_before = nxt.pot_bb
        facing_before = nxt.facing_bet_bb

        if FOLD_RE.search(al):
            _bump(counts, "FOLD")

        elif CHECK_RE.search(al):
            _bump(counts, "CHECK")

        elif CALL_RE.search(al):
            _bump(counts, "CALL")
            if facing_before > 0:
                nxt.pot_bb = pot_before + facing_before
                nxt.facing_bet_bb = 0.0

        elif BET_RE.search(al):
            pct = parse_bet_pct(label)
            amt = parse_bet_amount_bb(label)
            if pct is not None:
                bet_amt = (pct / 100.0) * pot_before
                bet_pct = pct
            elif amt is not None:
                bet_amt = amt
                bet_pct = (amt / max(pot_before, 1e-6)) * 100.0
            else:
                bet_amt = None
                bet_pct = None

            tok = _bucket_bet_pct(bet_pct)
            if tok: _bump(counts, tok)
            if depth == 0 and tok == "BET_33":
                _bump(counts, "DONK_33")
            if bet_amt is not None and bet_amt > 0:
                nxt.pot_bb = pot_before + bet_amt
                nxt.facing_bet_bb = bet_amt

        elif RAISE_RE.search(al) or ALLIN_RE.search(al):
            to_bb = parse_raise_to_bb(label)
            pot_child_ov, face_child_ov = _extract_state_from_node(ch) if ch else (None, None)
            if face_child_ov is not None:
                facing = face_child_ov
            elif facing_before > 0:
                facing = facing_before
            else:
                facing = _infer_facing(nxt.menu_pcts, pot_before, to_bb)

            tok = _bucket_raise(to_bb, facing, pot_before, nxt.stack_bb)
            _bump(counts, tok)

            rb = (to_bb / max(facing, 1e-6)) if to_bb else float("nan")
            rp = (to_bb / max(pot_before, 1e-6)) if to_bb else float("nan")
            tag = label if (face_child_ov is not None or facing_before > 0) else f"{label} [menu-inferred]"
            raises_log.append((tag, to_bb or float("nan"), facing, rb, rp))

            if to_bb is not None:
                inc = max(to_bb - facing, 0.0)
                nxt.pot_bb = pot_before + inc
                nxt.facing_bet_bb = to_bb

    if not kids:
        for lab in action_list(node):
            nxt = State(base.pot_bb, base.facing_bet_bb, base.stack_bb, base.menu_pcts)
            handle(lab, nxt, None)
        return

    for lab, ch in kids.items():
        nxt = State(base.pot_bb, base.facing_bet_bb, base.stack_bb, base.menu_pcts)
        handle(lab, nxt, ch)
        _dfs(ch, counts, nxt, depth + 1, max_depth, raises_log)

# ---- Public API ----
def parse_solver_payload(payload: Json, *, pot_bb: float, stack_bb: float, menu_pcts: Sequence[float], max_depth: int = 18) -> Tuple[Dict[str,float], Dict[str,Any]]:
    root = root_node(payload)
    counts = [0.0] * VOCAB_SIZE
    raises_log: List[Tuple[str,float,float,float,float]] = []
    _dfs(root, counts, State(pot_bb=pot_bb, facing_bet_bb=0.0, stack_bb=stack_bb, menu_pcts=tuple(menu_pcts)), 0, max_depth, raises_log)
    total = sum(counts) or 1.0
    probs = {ACTION_VOCAB[i]: counts[i]/total for i in range(VOCAB_SIZE) if counts[i] > 0}
    meta = {
        "total_actions": int(total),
        "raises_seen": [
            {"label": lab, "raise_to_bb": rt, "facing_bet_bb": fb,
             "ratio_rb": round(r_rb,3) if math.isfinite(r_rb) else None,
             "ratio_pot": round(r_p,3) if math.isfinite(r_p) else None}
            for (lab, rt, fb, r_rb, r_p) in raises_log
        ],
    }
    return probs, meta

def parse_solver_file(path: str, *, pot_bb: float, stack_bb: float = 100.0, menu_pcts: Sequence[float] = (0.33,), max_depth: int = 18) -> Tuple[Dict[str,float], Dict[str,Any]]:
    payload = load_solver(path)
    return parse_solver_payload(payload, pot_bb=pot_bb, stack_bb=stack_bb, menu_pcts=menu_pcts, max_depth=max_depth)

def default_pot_for_filename(name: str) -> float:
    n = name.lower()
    if "3bet_hu" in n: return 24.0
    if "4bet_hu" in n: return 48.0
    if "srp_hu"  in n: return 28.0
    if "limped"  in n: return 3.5
    return 20.0