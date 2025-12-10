# ml/inference/policy/helpers.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------
# Basic tensor utilities
# ---------------------------

def as_batch(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.view(1, -1)

def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    T: float,
    eps: float,
) -> torch.Tensor:
    """Softmax over legal actions only; adds epsilon-smoothing."""
    logits = as_batch(logits)
    mask = as_batch(mask).to(logits.dtype)
    big_neg = torch.finfo(logits.dtype).min / 4
    masked = torch.where(mask > 0.5, logits / max(T, 1e-6), big_neg)
    p = F.softmax(masked, dim=-1)
    # epsilon redistribute over legal mass
    legal = (mask > 0.5).to(p.dtype)
    denom = legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
    p = p * (1 - eps) + (eps / denom) * legal
    # renormalize (just in case)
    p = p * legal
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return p

# ---------------------------
# Menu normalization
# ---------------------------

def normalize_raise_buckets(raise_buckets: Optional[Iterable[Any]]) -> set[int]:
    """
    Accept either multipliers (1.5, 2.0, 3.0) or centi-bb (150,200,300).
    Strings like "1.5" or "150" are also accepted.
    """
    out: set[int] = set()
    for x in (raise_buckets or []):
        try:
            if isinstance(x, str):
                x = x.strip().replace("%", "")
            v = float(x)
            suf = int(round(v * 100)) if v < 10 else int(round(v))  # 1.5 → 150, 150 → 150
            out.add(suf)
        except Exception:
            pass
    return out

def normalize_bet_sizes(bet_sizes: Optional[Iterable[Any]]) -> set[int]:
    """
    Accept fractional [0.33, 0.66] or percent [33, 66] or strings.
    """
    out: set[int] = set()
    for x in (bet_sizes or []):
        try:
            if isinstance(x, str):
                x = x.strip().replace("%", "")
            v = float(x)
            pct = int(round(v * 100)) if v <= 1.0 else int(round(v))
            out.add(pct)
        except Exception:
            pass
    return out

def menu_mask(
    actions: List[str],
    *,
    actor: str,
    facing_bet: bool,
    bet_sizes: Optional[List[float]],
    raise_buckets: Optional[List[int]],
    allow_allin: Optional[bool],
) -> torch.Tensor:
    """Return [V] mask aligned to actions."""
    allow_ai = True if allow_allin is None else bool(allow_allin)
    m = torch.zeros(len(actions), dtype=torch.float32)

    bet_percents = normalize_bet_sizes(bet_sizes)
    raise_suffixes = normalize_raise_buckets(raise_buckets)

    act = actor.lower()

    for i, tok in enumerate(actions):
        T = tok.upper()
        legal = False

        if not facing_bet:
            # ROOT
            if T == "CHECK":
                legal = True
            elif T.startswith("BET_"):
                if not bet_percents:
                    legal = True
                else:
                    try:
                        pct = int(T.split("_", 1)[1])   # e.g. BET_33
                        legal = (pct in bet_percents)
                    except Exception:
                        legal = False
            elif T.startswith("DONK_"):
                legal = (act == "oop")
            elif T == "ALLIN":
                legal = allow_ai

        else:
            # FACING
            if T in ("FOLD", "CALL"):
                legal = True
            elif T.startswith("RAISE_"):
                if not raise_suffixes:
                    legal = True
                else:
                    try:
                        suf = int(T.split("_", 1)[1])  # e.g. RAISE_150
                        legal = (suf in raise_suffixes)
                    except Exception:
                        legal = False
            elif T == "ALLIN":
                legal = allow_ai

        if legal:
            m[i] = 1.0

    if m.sum().item() == 0:
        m.fill_(1.0)  # fail-safe
    return m

# ---------------------------
# Vocab utilities
# ---------------------------

def update_vocab_cache(
    current_actions: List[str],
    new_actions: List[str],
    cached_proj: Optional[torch.Tensor] = None,
) -> Tuple[List[str], Dict[str, int], Optional[torch.Tensor]]:
    """Returns (action_vocab, vocab_index, maybe_reset_cached_proj)."""
    if new_actions == current_actions:
        return current_actions, {a: i for i, a in enumerate(current_actions)}, cached_proj
    vix = {a: i for i, a in enumerate(new_actions)}
    return list(new_actions), vix, None  # invalidate cached_proj

# ---------------------------
# Equity / EV helpers
# ---------------------------

def equity_delta_vector(
    *,
    eq_margin: float,
    hero_is_ip: bool,
    facing_bet: bool,
    action_vocab: Sequence[str],
    vocab_index: Mapping[str, int],
) -> torch.Tensor:
    """Map equity margin into gentle [1,V] logit deltas."""
    V = len(action_vocab)
    delta = torch.zeros(V, dtype=torch.float32)
    m = max(-0.25, min(0.25, float(eq_margin)))
    base = m

    def bump(tok: str, amt: float):
        j = vocab_index.get(tok)
        if j is not None:
            delta[j] += amt

    if facing_bet:
        bump("CALL", +1.00 * base)
        for tok in ("RAISE_150", "RAISE_200", "RAISE_300"):
            bump(tok, +0.50 * base)
        bump("FOLD", -1.25 * base)
    else:
        for tok in ("BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100"):
            bump(tok, +0.80 * base)
        if not hero_is_ip:
            bump("DONK_33", +0.80 * base)
        bump("CHECK", -1.00 * base)

    return delta.view(1, -1)

def cap_allins(
    p: torch.Tensor,
    *,
    eff_stack_bb: float,
    action_vocab: Sequence[str],
    max_allin_freq: float,
    risk_floor_stack_bb: float,
) -> torch.Tensor:
    """Cap ALLIN probability mass when deep; preserve distribution over others."""
    if eff_stack_bb <= float(risk_floor_stack_bb):
        return p
    try:
        idx = action_vocab.index("ALLIN")
    except ValueError:
        return p

    p = p if p.dim() == 2 else p.view(1, -1)
    if p[0, idx] <= max_allin_freq:
        return p

    over = p[0, idx] - max_allin_freq
    p = p.clone()
    p[0, idx] = max_allin_freq
    rest = [i for i in range(len(action_vocab)) if i != idx]
    denom = float(p[0, rest].sum().item()) or 1e-12
    scale = 1.0 + (over / denom)
    p[0, rest] *= scale
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return p

# ---------------------------
# Preflop helpers
# ---------------------------

def compute_temperature(evs: np.ndarray) -> float:
    """
    Temperature from EV spread:
      - High spread → lower T (confident)
      - Low spread  → higher T (uncertain)
    """
    if len(evs) == 0 or np.allclose(evs, evs[0]):
        return 2.5
    spread = float(np.max(evs) - np.min(evs))
    norm = np.clip(spread / 0.5, 0.0, 1.0)  # tune 0.5 to your scale
    return 2.5 - 2.0 * norm  # T in [0.5, 2.5]

def soft_prior_blend(
    tokens: Sequence[str],
    req: Any,
    evs: Optional[Mapping[str, float]] = None,
    eq_sig: Optional[Any] = None,
) -> List[float]:
    """
    Lightweight prior based on equity, EV, and rough size desirability.
    Returns unnormalized logits (one per token).
    """
    stack = float(getattr(req, "eff_stack_bb", None) or getattr(req, "pot_bb", None) or 100.0)
    facing = bool(getattr(req, "facing_bet", False))

    eq_boost = 1.0
    if eq_sig and getattr(eq_sig, "available", False) and getattr(eq_sig, "p_win", None) is not None:
        # p_win in [0,1] → scale to [0.8, 1.3]
        eq_boost = 0.8 + 0.5 * float(eq_sig.p_win)

    priors: List[float] = []
    for a in tokens:
        if a == "FOLD":
            base = 0.2 if facing else 0.01
            if eq_sig and getattr(eq_sig, "available", False):
                base *= (1.2 - eq_boost)  # invert equity
            priors.append(base)

        elif a == "CALL":
            base = 0.35 if facing else 0.0
            base *= eq_boost
            priors.append(base)

        elif a == "CHECK":
            base = 0.35 if not facing else 0.0
            priors.append(base)

        elif a.startswith("RAISE_") or a.startswith("OPEN_") or a.startswith("BET_") or a.startswith("DONK_"):
            try:
                suf = a.split("_", 1)[1]
                # interpret suffix as % or centi-bb or multiple; keep heuristic simple
                if a.startswith("RAISE_"):   # total multiple in centi-bb
                    mult = float(int(suf)) / 100.0
                    frac = min(mult / 3.0, 1.0)  # gently prefer mid multiples
                else:
                    pct = float(int(suf))        # e.g. 33
                    frac = min(abs(pct) / 100.0, 1.0)

                base = 1.0 - abs(frac - 0.5)  # prefer ~50%
                if evs and a in evs:
                    base *= max(float(evs[a]), 0.0) + 1.0
                base *= eq_boost
                priors.append(base)
            except Exception:
                priors.append(0.01)
        else:
            priors.append(0.01)

    return priors

# ---------------------------
# Side derivation
# ---------------------------

def derive_side(req: Any, *, hero_is_ip: bool, pol_post: Any) -> str:
    """
    Determine 'root' or 'facing' from request/routers deterministically.
    """
    fb = getattr(req, "facing_bet", None)
    if isinstance(fb, bool):
        return "facing" if fb else "root"

    # explicit faced size implies facing
    if getattr(req, "faced_size_frac", None) is not None or getattr(req, "faced_size_pct", None) is not None:
        return "facing"

    # reuse facing parser from router if available
    try:
        facing_flag, _ = pol_post.facing.infer_facing_and_size(req, hero_is_ip=hero_is_ip)
        return "facing" if facing_flag else "root"
    except Exception:
        return "root"