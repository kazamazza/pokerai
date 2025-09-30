from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple, Optional

from ml.etl.rangenet.postflop.solver_policy_kinds import ScenarioParseKind
from ml.etl.rangenet.postflop.solver_scan import presence_scan
from ml.models.policy_consts import VOCAB_INDEX, ACTION_VOCAB


def _last_number(s: str) -> Optional[float]:
    import re
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", str(s))
    if not m: return None
    try: return float(m[-1])
    except Exception: return None

def _has_any(s: str, *needles: str) -> bool:
    u = str(s).upper()
    return any(n in u for n in needles)

# --- TREE ADAPTERS (payload → nodes) ---
def root_node(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    # Many dumps place root at top-level; adjust if your schema differs.
    return payload

def get_children(node: Mapping[str, Any]) -> Dict[str, Any]:
    ch = node.get("childrens") or node.get("children") or {}
    return ch if isinstance(ch, dict) else {}

def actions_and_mix(node: Mapping[str, Any]) -> Tuple[List[str], List[float]]:
    """
    Return (actions, mix). Mix is normalized per-node if available.
    Supports both node['actions'] and node['strategy']['strategy'] style dumps.
    """
    acts = list(node.get("actions") or [])
    strat = node.get("strategy") or {}
    if not acts and isinstance(strat, dict):
        acts = list((strat.get("actions") or []))

    strat_map = strat.get("strategy") if isinstance(strat, dict) else None
    if not acts or not isinstance(strat_map, dict) or not strat_map:
        return acts, []

    k = len(acts)
    mass = [0.0] * k; nrows = 0
    for probs in strat_map.values():
        if not isinstance(probs, list): continue
        L = min(len(probs), k)
        if L <= 0: continue
        for i in range(L):
            v = probs[i]
            if v is not None and v >= 0:
                mass[i] += float(v)
        nrows += 1
    if nrows == 0: return acts, []

    s = sum(mass)
    if s > 0: mass = [m / s for m in mass]
    return acts, mass

def _norm_key(s: str) -> Tuple[str, Optional[float]]:
    u = str(s).strip().upper()
    verb = u.split()[0] if u else u
    val = _last_number(u)
    return verb, val

def resolve_child(children: Mapping[str, Any], action_label: str, eps: float = 1e-3) -> Mapping[str, Any]:
    if not children: return {}
    if action_label in children and isinstance(children[action_label], dict):
        return children[action_label]
    def _norm(s: str):
        u = str(s).strip().upper()
        verb = u.split()[0] if u else u
        # last numeric
        import re
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", u)
        val = float(nums[-1]) if nums else None
        return verb, val
    vt, nt = _norm(action_label)
    candidates = []
    for k, v in children.items():
        if not isinstance(v, dict): continue
        vk, nk = _norm(k)
        if vk != vt: continue
        if nt is None and nk is None: return v
        if (nt is not None) and (nk is not None) and abs(nt - nk) <= eps: return v
        candidates.append((k, v))
    # fallback: single same-verb child
    if len(candidates) == 1:
        return candidates[0][1]
    return {}

# --- BUCKETING ---
def bucket_bet_label(up: str, *, pot_bb: float) -> str:
    v = _last_number(up)
    if v is None or pot_bb <= 0: return "BET_100"
    pct = v if "%" in up.upper() else (v / pot_bb) * 100.0
    if pct <= 27: return "BET_25"
    if pct <= 40: return "BET_33"
    if pct <= 58: return "BET_50"
    if pct <= 70: return "BET_66"
    if pct <= 85: return "BET_75"
    return "BET_100"

def parse_raise_to_bb(label: str, *, pot_bb: float, bet_size_bb: float) -> Optional[float]:
    """
    Interpret 'RAISE X' as raise-to in **bb**. If label looks like a %,
    translate via pot_bb; else assume BB directly.
    """
    up = str(label).upper()
    v = _last_number(up)
    if v is None: return None
    if "%" in up: return (v/100.0) * pot_bb
    return v  # already BB

def bucket_raise_label(raise_label: str, *, pot_bb: float, facing_bet_bb: float | None, stack_bb: float | None = None) -> str:
    up = str(raise_label).upper()
    if _has_any(up, "ALLIN", "ALL-IN", "JAM"): return "ALLIN"
    if not facing_bet_bb or facing_bet_bb <= 0: return "RAISE_300"
    raise_to_bb = parse_raise_to_bb(up, pot_bb=pot_bb, bet_size_bb=facing_bet_bb)
    if raise_to_bb is None: return "RAISE_300"
    if stack_bb and stack_bb > 0 and raise_to_bb >= 0.95 * stack_bb: return "ALLIN"
    m = raise_to_bb / facing_bet_bb
    if m <= 1.75: return "RAISE_150"
    if m <= 2.5:  return "RAISE_200"
    return "RAISE_300"

def parse_root_bet_size_bb(root_actions: List[str], pot_bb: float) -> Optional[float]:
    best = 0.0
    for a in root_actions:
        u = str(a).upper()
        if not u.startswith("BET"): continue
        v = _last_number(u)
        if v is not None: best = max(best, float(v))
    return best or None

# --- DFS collector for OOP after IP bet ---
def collect_oop_actions_recursive(
    node: Mapping[str, Any],
    weight: float,
    *,
    pot_bb: float,
    facing_bet_bb: float,
    stack_bb: Optional[float],
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    acts, mix = actions_and_mix(node)
    if not acts or not mix: return out

    children = get_children(node)

    for a, p in zip(acts, mix):
        w = float(weight) * float(p)
        if w <= 0: continue
        up = str(a).strip().upper()

        if up.startswith("CALL"):   out.append(("CALL", w))
        elif up.startswith("FOLD"): out.append(("FOLD", w))
        elif _has_any(up, "ALLIN", "ALL-IN", "JAM"):
            if "ALLIN" in VOCAB_INDEX: out.append(("ALLIN", w))
        elif up.startswith("RAISE") or _has_any(up, "RE-RAISE", "RERAISE", "MIN-RAISE", "MINRAISE"):
            bucket = bucket_raise_label(up, pot_bb=pot_bb, facing_bet_bb=facing_bet_bb, stack_bb=stack_bb)
            if bucket in VOCAB_INDEX: out.append((bucket, w))
            # Recurse with updated facing-bet = raise-to
            ch = resolve_child(children, a)
            if isinstance(ch, Mapping) and ch:
                raise_to_bb = parse_raise_to_bb(up, pot_bb=pot_bb, bet_size_bb=facing_bet_bb) or facing_bet_bb
                out.extend(
                    collect_oop_actions_recursive(
                        ch, w, pot_bb=pot_bb, facing_bet_bb=raise_to_bb, stack_bb=stack_bb
                    )
                )
        # else: ignore unrelated labels

    return out

def infer_parse_kind_from_role(role: str) -> ScenarioParseKind:
    r = (role or "").upper()
    if r.endswith("_IP"):  # PFR_IP, AGGRESSOR_IP, SB_IP (limped)
        if r == "SB_IP":   # limped single special
            return ScenarioParseKind.LIMPed_SB_IP
        return ScenarioParseKind.IP_ROOT
    if r.endswith("_OOP"):
        # For some SRP Caller_OOP/3bet Caller_OOP solves the root actor is IP;
        # prefer OOP_VS_IP_ROOT_BET to pick OOP responses vs root bet.
        if "CALLER_OOP" in r:
            return ScenarioParseKind.OOP_VS_IP_ROOT_BET
        return ScenarioParseKind.OOP_ROOT
    # sane default
    return ScenarioParseKind.IP_ROOT

@dataclass
class PolicyParseConfig:
    pot_bb: float
    stack_bb: float
    role: str
    parse_kind: Optional[ScenarioParseKind] = None  # NEW

@dataclass
class ParseResult:
    vec: List[float]
    debug: Dict[str, Any]
    ok: bool

class SolverPolicyParser:
    def __init__(self, action_vocab: List[str] | None = None):
        self.vocab = action_vocab or ACTION_VOCAB
        self.index = {a: i for i, a in enumerate(self.vocab)}

    # --- helpers to add mass ---
    def _add(self, vec, key, w): vec[self.index[key]] += float(w)

    # --- strategies ---
    def _ip_root(self, root, acts, mix, cfg) -> Tuple[List[float], Dict[str, Any], bool]:
        vec = [0.0] * len(self.vocab)
        for a, p in zip(acts, mix):
            if p <= 0: continue
            u = str(a).upper()
            if u.startswith("CHECK"):
                self._add(vec, "CHECK", p)
            elif u.startswith("BET"):
                b = bucket_bet_label(u, pot_bb=cfg.pot_bb)
                self._add(vec, b, p)
        s = sum(vec)
        ok = s > 0
        if ok: vec = [x/s for x in vec]
        return vec, {"mode": "ip_root"}, ok

    def _oop_root(self, root, acts, mix, cfg) -> Tuple[List[float], Dict[str, Any], bool]:
        vec = [0.0] * len(self.vocab)
        any_mass = any_raise = False
        children = get_children(root)

        # OOP donks at root
        for a, p in zip(acts, mix):
            if p <= 0: continue
            if str(a).upper().startswith("BET"):
                b = bucket_bet_label(str(a).upper(), pot_bb=cfg.pot_bb)
                self._add(vec, b, p); any_mass = True

        # OOP CHECK → go to IP node, then for each IP BET collect OOP responses
        for a, p in zip(acts, mix):
            if p <= 0: continue
            if not str(a).upper().startswith("CHECK"): continue
            node_ip = resolve_child(children, a)
            if not node_ip: continue
            ip_acts, ip_mix = actions_and_mix(node_ip)
            ip_children = get_children(node_ip)
            for lab, q in zip(ip_acts, ip_mix or []):
                if q <= 0: continue
                if not str(lab).upper().startswith("BET"): continue
                node_ip_bet = resolve_child(ip_children, lab)
                if not node_ip_bet: continue
                facing_bet_bb = parse_raise_to_bb(str(lab).upper(), pot_bb=cfg.pot_bb, bet_size_bb=1.0) or 1.0
                for bucket, mass in collect_oop_actions_recursive(
                    node=node_ip_bet, weight=float(p*q),
                    pot_bb=cfg.pot_bb, facing_bet_bb=facing_bet_bb, stack_bb=cfg.stack_bb):
                    self._add(vec, bucket, mass); any_mass = True
                    if bucket.startswith("RAISE") or bucket == "ALLIN": any_raise = True

        if not any_mass:
            return vec, {"mode": "oop_root", "any_raise": False}, False

        s = sum(vec)
        if s > 0: vec = [x/s for x in vec]
        return vec, {"mode": "oop_root", "any_raise": any_raise}, True

    def _oop_vs_ip_root_bet(self, root, acts, mix, cfg) -> Tuple[List[float], Dict[str, Any], bool]:
        # Root actor is IP; we want OOP responses vs each root BET
        vec = [0.0] * len(self.vocab)
        any_mass = any_raise = False
        children = get_children(root)
        facing_bet_bb = parse_root_bet_size_bb(acts, cfg.pot_bb) or 1.0
        for a, p in zip(acts, mix):
            if p <= 0: continue
            if not str(a).upper().startswith("BET"): continue
            node_bet = resolve_child(children, a)
            if not node_bet: continue
            for bucket, mass in collect_oop_actions_recursive(
                    node=node_bet, weight=float(p),
                    pot_bb=cfg.pot_bb, facing_bet_bb=facing_bet_bb, stack_bb=cfg.stack_bb):
                self._add(vec, bucket, mass); any_mass = True
                if bucket.startswith("RAISE") or bucket == "ALLIN": any_raise = True

        if not any_mass:
            return vec, {"mode": "oop_vs_ip_root_bet", "any_raise": False}, False
        s = sum(vec)
        if s > 0: vec = [x/s for x in vec]
        return vec, {"mode": "oop_vs_ip_root_bet", "any_raise": any_raise}, True

    # --- main parse ---
    def parse(self, payload: Mapping[str, Any], cfg: PolicyParseConfig) -> ParseResult:
        vec = [0.0] * len(self.vocab)

        root = root_node(payload)
        acts, mix = actions_and_mix(root)
        if acts and (not mix or sum(mix) <= 0):
            mix = [1.0 / len(acts)] * len(acts)
        if not acts or not mix or sum(mix) <= 0:
            return ParseResult(vec, {"reason": "zero_mass_root"}, ok=False)

        kind = cfg.parse_kind or infer_parse_kind_from_role(cfg.role)

        if kind == ScenarioParseKind.IP_ROOT or kind == ScenarioParseKind.LIMPed_SB_IP:
            v, dbg, ok = self._ip_root(root, acts, mix, cfg)
        elif kind == ScenarioParseKind.OOP_ROOT:
            v, dbg, ok = self._oop_root(root, acts, mix, cfg)
        elif kind == ScenarioParseKind.OOP_VS_IP_ROOT_BET:
            v, dbg, ok = self._oop_vs_ip_root_bet(root, acts, mix, cfg)
        else:
            v, dbg, ok = self._ip_root(root, acts, mix, cfg)

        # Guided fallback: if file *contains* raises/calls/folds but computed mass=0 raise/call, dump + retry once
        pres = presence_scan(payload)
        raise_mass = sum(v[self.index[a]] for a in ("RAISE_150","RAISE_200","RAISE_300","ALLIN") if a in self.index)
        call_fold = sum(v[self.index[a]] for a in ("CALL","FOLD") if a in self.index)

        if (pres["has_raise"] and raise_mass <= 1e-12 and kind in (ScenarioParseKind.OOP_ROOT, ScenarioParseKind.OOP_VS_IP_ROOT_BET)):
            dbg["warn"] = "raises_present_but_zero_mass"
        if (pres["has_call"] and pres["has_fold"]) and (call_fold <= 1e-12):
            dbg["warn_call_fold"] = "call_fold_present_but_zero_mass"

        return ParseResult(v, dbg | {"presence": pres, "kind": kind.name}, ok=ok)