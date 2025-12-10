import math
from typing import Optional, Sequence, List, Dict
import torch.nn.functional as F
import torch


def _masked_softmax_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.view(1, -1)
    mask = mask.view(1, -1).to(logits.dtype)
    big_neg = torch.finfo(logits.dtype).min / 4
    masked = torch.where(mask > 0.5, logits, big_neg)
    return F.softmax(masked, dim=-1) * mask / (mask.sum(dim=-1, keepdim=True).clamp_min(1e-12))

def _single_delta_for_target_share(z: torch.Tensor,
                                  legal_idx: List[int],
                                  target_idx: int,
                                  tau_target: float,
                                  max_boost: float) -> float:
    """
    Approximate delta to give target_idx probability ~ tau_target among legal_idx
    by solving exp(T+δ)/Σ = τ. Using base Σ from current logits.
    """
    if tau_target <= 0.0:
        return 0.0
    z = z.view(1, -1)
    l = z[0]
    L = torch.logsumexp(l[legal_idx], dim=0)
    T = l[target_idx]
    try:
        raw = math.log(max(min(tau_target, 0.99), 1e-6)) + float(L - T)
    except (ValueError, OverflowError):
        raw = 0.0
    return float(max(0.0, min(raw, max_boost)))

def _best_idx_by_evs(actions: Sequence[str],
                     mask: torch.Tensor,
                     evs: Dict[str, float],
                     allow_tokens: Optional[Sequence[str]] = None) -> Optional[int]:
    allow = set(t.upper() for t in (allow_tokens or []))
    best_v = None
    best_i = None
    for i, a in enumerate(actions):
        if mask[i] <= 0.5:
            continue
        A = a.upper()
        if allow and (A not in allow):
            continue
        v = float(evs.get(a, evs.get(A, float("-inf"))))
        if best_v is None or v > best_v:
            best_v, best_i = v, i
    return best_i

def _parse_pct_from_token(tok: str) -> Optional[int]:
    try:
        return int(tok.split("_", 1)[1])
    except Exception:
        return None

def _aggressive_indices(actions: Sequence[str], *, side: str) -> List[int]:
    up = [a.upper() for a in actions]
    if side == "facing":
        return [i for i, a in enumerate(up) if a.startswith("RAISE_")]
    if side == "root":
        return [i for i, a in enumerate(up) if a.startswith("BET_") or a.startswith("DONK_")]
    # preflop unopened convenience
    return [i for i, a in enumerate(up) if a.startswith("OPEN_") or a.startswith("RAISE_")]

def _passive_token(side: str, *, preflop_unopened: bool, free_check: bool) -> str:
    if side == "facing":
        return "CALL"
    if side == "root":
        return "CHECK"
    # preflop special
    if preflop_unopened:
        return "CHECK" if free_check else "FOLD"
    return "CALL"