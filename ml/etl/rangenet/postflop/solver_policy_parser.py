import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ml.etl.rangenet.postflop.utils import get_children, _extract_state_from_node, _bump, parse_bet_pct, \
    parse_bet_amount_bb, _bucket_bet_pct, parse_raise_to_bb, _infer_facing, _bucket_raise, action_list, root_node, \
    _first_label_containing, _compute_facing_from_label, _bump_vec, load_solver
from ml.models.policy_consts import VOCAB_SIZE, ACTION_VOCAB

FOLD_RE  = re.compile(r"\bfold\b", re.IGNORECASE)
CALL_RE  = re.compile(r"\bcall\b", re.IGNORECASE)
CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)
BET_RE   = re.compile(r"\bbet\b", re.IGNORECASE)
RAISE_RE = re.compile(r"\braise\b", re.IGNORECASE)
ALLIN_RE = re.compile(r"\ball[-\s]*in\b|\bjam\b", re.IGNORECASE)

Json = Dict[str, Any]

@dataclass
class State:
    pot_bb: float
    facing_bet_bb: float
    stack_bb: float
    menu_pcts: Sequence[float]

class FocusMode:
    ROOT = "ROOT"
    OOP_VS_IP_ROOT_BET = "OOP_VS_IP_ROOT_BET"
    IP_VS_OOP_DONK     = "IP_VS_OOP_DONK"

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

    # --- NEW: credit FOLD from node.actions/strategy when node HAS children ---
    # (Many solver trees encode FOLD only in node.strategy, without a FOLD child.)
    if kids:
        try:
            any_fold_child = any(FOLD_RE.search(str(l).lower()) for l in kids.keys())
            if not any_fold_child:
                acts, mix = actions_and_mix(node)  # robust helper you already have
                for lab, p in zip(acts, mix):
                    if p > 0 and FOLD_RE.search(str(lab).lower()):
                        _bump(counts, "FOLD", p)  # bump only FOLD to avoid double counting
        except Exception:
            pass
    # --- END NEW ---

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

            tok = _bucket_raise(to_bb, facing, pot_before, nxt.stack_bb)  # keep your current signature
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

def default_pot_for_filename(name: str) -> float:
    n = name.lower()
    if "3bet_hu" in n: return 24.0
    if "4bet_hu" in n: return 48.0
    if "srp_hu"  in n: return 28.0
    if "limped"  in n: return 3.5
    return 20.0

def actions_and_mix(node: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    acts: List[str] = []
    mix: List[float] = []
    for k in ("actions", "action", "labels"):
        v = node.get(k)
        if isinstance(v, list) and v:
            acts = [str(a) for a in v]
            break
    if not acts:
        kids = get_children(node)
        acts = list(kids.keys())
    for k in ("strategy", "mix", "frequencies", "probs", "probabilities"):
        v = node.get(k)
        if isinstance(v, list) and len(v) == len(acts):
            mix = [float(x) for x in v]
            break
    if not mix or len(mix) != len(acts):
        p = 1.0 / max(1, len(acts))
        mix = [p] * len(acts)
    s = sum(mix) or 1.0
    mix = [x / s for x in mix]
    return acts, mix


def parse_facing_distribution(
    payload: Dict[str, Any],
    *,
    pot_bb: float,
    stack_bb: float,
    menu_pcts: Sequence[float],
    focus: str,
) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
    root = root_node(payload)
    kids = get_children(root)
    bet_node = None
    facing_bb = 0.0
    path = "root"

    if focus == FocusMode.OOP_VS_IP_ROOT_BET:
        lab = _first_label_containing(kids, "bet")  # IP bets at root
        if lab is None:
            # OOP check -> IP bet
            lab_chk = _first_label_containing(kids, "check")
            if lab_chk:
                kids2 = get_children(kids[lab_chk])
                lab = _first_label_containing(kids2, "bet")
                if lab:
                    bet_node = kids2[lab]; facing_bb = _compute_facing_from_label(lab, pot_bb)
                    path = f"root->CHECK->{lab}"
        else:
            bet_node = kids[lab]; facing_bb = _compute_facing_from_label(lab, pot_bb)
            path = f"root->{lab}"

    elif focus == FocusMode.IP_VS_OOP_DONK:
        lab = _first_label_containing(kids, "bet")  # OOP donk at root
        if lab:
            bet_node = kids[lab]; facing_bb = _compute_facing_from_label(lab, pot_bb)
            path = f"root->{lab}"

    if bet_node is None:
        return None

    resp_children = get_children(bet_node)
    if not resp_children:
        return None

    acts, mix = actions_and_mix(bet_node)
    if not acts or len(acts) != len(resp_children):
        acts = list(resp_children.keys())
        mix = [1.0 / len(acts)] * len(acts)

    probs = {a: 0.0 for a in ACTION_VOCAB}
    st = State(pot_bb=pot_bb, facing_bet_bb=facing_bb, stack_bb=stack_bb, menu_pcts=tuple(menu_pcts) if menu_pcts else None)

    for lab, p in zip(acts, mix):
        if p <= 0: continue
        al = str(lab).lower()
        if FOLD_RE.search(al):
            _bump_vec(probs, "FOLD", p)
        elif CHECK_RE.search(al):
            _bump_vec(probs, "CHECK", p)
        elif CALL_RE.search(al):
            _bump_vec(probs, "CALL", p)
        elif BET_RE.search(al):
            pct = parse_bet_pct(lab)
            tok = _bucket_bet_pct(pct)
            if tok: _bump_vec(probs, tok, p)
        elif RAISE_RE.search(al) or ALLIN_RE.search(al):
            to_bb = parse_raise_to_bb(lab)  # label-only parser in your codebase
            facing = st.facing_bet_bb if st.facing_bet_bb > 0 else _infer_facing(st.menu_pcts, st.pot_bb, to_bb)
            tok = _bucket_raise(to_bb, facing, st.stack_bb)  # your simplified version (no pot arg)
            _bump_vec(probs, tok, p)
        # else ignore

    s = sum(probs.values())
    if s <= 0:
        return None
    for k in probs: probs[k] = probs[k] / s
    meta = {"focus": focus, "path": path, "facing_bet_bb": facing_bb}
    return probs, meta


def parse_solver_file(
    path: str,
    *,
    pot_bb: float,
    stack_bb: float = 100.0,
    menu_pcts: Sequence[float] = (0.33,),
    max_depth: int = 18,
    focus: Optional[str] = None,  # NEW, default None = keep baseline
) -> Tuple[Dict[str,float], Dict[str,Any]]:
    payload = load_solver(path)

    # SAFE optional path: try facing extraction, fallback to baseline
    if focus and focus != FocusMode.ROOT:
        try:
            out = parse_facing_distribution(
                payload, pot_bb=pot_bb, stack_bb=stack_bb, menu_pcts=menu_pcts, focus=focus
            )
            if out is not None:
                return out
        except Exception:
            # hard fallback to baseline
            pass

    # Baseline untouched
    return parse_solver_payload(
        payload, pot_bb=pot_bb, stack_bb=stack_bb, menu_pcts=menu_pcts, max_depth=max_depth
    )