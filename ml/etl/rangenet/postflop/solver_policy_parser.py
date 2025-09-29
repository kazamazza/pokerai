from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple, Optional

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
    """
    Match by verb + last-number. Works across minor formatting variants.
    """
    if not children: return {}
    if action_label in children and isinstance(children[action_label], dict):
        return children[action_label]
    v_t, n_t = _norm_key(action_label)
    for k, v in children.items():
        if not isinstance(v, dict): continue
        v_k, n_k = _norm_key(k)
        if v_k != v_t: continue
        if n_t is None and n_k is None: return v
        if (n_t is not None) and (n_k is not None) and abs(n_t - n_k) <= eps: return v
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

# --- Config & Result types ---
@dataclass
class PolicyParseConfig:
    pot_bb: float
    stack_bb: float
    role: str              # e.g., "PFR_IP", "CALLER_OOP", "AGGRESSOR_IP" etc.

@dataclass
class ParseResult:
    vec: List[float]                   # length = len(ACTION_VOCAB)
    debug: Dict[str, Any]
    ok: bool

# --- The parser class ---
class SolverPolicyParser:
    def __init__(self, action_vocab: List[str] | None = None):
        self.vocab = action_vocab or ACTION_VOCAB
        self.index = {a: i for i, a in enumerate(self.vocab)}

    def parse(self, payload: Mapping[str, Any], cfg: PolicyParseConfig) -> ParseResult:
        """
        Convert a single solver payload to a soft policy vector over ACTION_VOCAB.
        Supports:
          - IP at root: CHECK / BET_xx buckets
          - OOP vs root BET: CALL/FOLD/RAISE_xxx/ALLIN (recursively)
          - OOP after IP CHECK: CHECK / DONK (bucketed as BET_xx)
        """
        vec = [0.0] * len(self.vocab)
        root = root_node(payload)
        acts, mix = actions_and_mix(root)

        if not acts or not mix or sum(mix) <= 0:
            return ParseResult(vec, {"reason": "zero_mass_root"}, ok=False)

        # IP at root (aggressor roles)
        is_aggressor = cfg.role.startswith("PFR") or ("AGGRESSOR" in cfg.role)
        if is_aggressor:
            for a, p in zip(acts, mix):
                if p <= 0: continue
                u = str(a).upper()
                if u.startswith("CHECK"):
                    vec[self.index["CHECK"]] += float(p)
                elif u.startswith("BET"):
                    b = bucket_bet_label(u, pot_bb=cfg.pot_bb)
                    vec[self.index[b]] += float(p)
            s = sum(vec)
            if s <= 0:
                return ParseResult(vec, {"reason": "zero_mass_child"}, ok=False)
            return ParseResult([x/s for x in vec], {"mode": "ip_root"}, ok=True)

        # OOP branch
        # (A) vs BET at root: walk each BET child and aggregate OOP responses recursively
        facing_bet_bb = parse_root_bet_size_bb(acts, cfg.pot_bb) or 1.0
        children = get_children(root)
        any_mass = False
        any_raise = False

        for a, p in zip(acts, mix):
            if p <= 0: continue
            u = str(a).upper()
            if not u.startswith("BET"): continue
            node_bet = resolve_child(children, a)
            if not isinstance(node_bet, Mapping) or not node_bet: continue

            pairs = collect_oop_actions_recursive(
                node=node_bet, weight=float(p),
                pot_bb=cfg.pot_bb, facing_bet_bb=facing_bet_bb, stack_bb=cfg.stack_bb
            )
            for bucket, mass in pairs:
                vec[self.index[bucket]] += mass
                any_mass = True
                if bucket.startswith("RAISE") or bucket == "ALLIN":
                    any_raise = True

        # (B) OOP after IP CHECK: find CHECK child and collect CHECK/DONK
        for a, p in zip(acts, mix):
            if p <= 0: continue
            u = str(a).upper()
            if not u.startswith("CHECK"): continue
            node_check = resolve_child(children, a)
            if not isinstance(node_check, Mapping) or not node_check: continue
            a2, m2 = actions_and_mix(node_check)
            if not a2 or not m2: continue
            for aa, mm in zip(a2, m2):
                mass = float(p * mm)
                if mass <= 0: continue
                uu = str(aa).upper()
                if uu.startswith("CHECK"):
                    vec[self.index["CHECK"]] += mass
                    any_mass = True
                elif uu.startswith("BET"):
                    b = bucket_bet_label(uu, pot_bb=cfg.pot_bb)
                    vec[self.index[b]] += mass
                    any_mass = True

        if not any_mass:
            return ParseResult(vec, {"reason": "zero_mass_child"}, ok=False)

        # normalize
        s = sum(vec)
        if s > 0:
            vec = [x/s for x in vec]

        dbg = {"mode": "oop_root", "any_raise": any_raise}
        return ParseResult(vec, dbg, ok=True)