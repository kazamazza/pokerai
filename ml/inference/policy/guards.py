# ml/inference/policy/helpers/guards.py

from typing import Dict, Optional, Sequence, Tuple
import torch

def clamp_fold_probabilities(
    p: torch.Tensor,                       # [1,V]
    actions: Sequence[str],
    *,
    equity: Optional[float] = None,        # eq_sig.p_win in [0,1]
    ev_map: Optional[Dict[str, float]] = None,
    hero_mask: Optional[torch.Tensor] = None,  # [V] 0/1
    spr: Optional[float] = None,
    cap_hi: float = 0.05,                  # equity >= 0.65
    cap_mid: float = 0.15,                 # 0.55 <= equity < 0.65
    cap_low: float = 0.35,                 # low-SR safety net
) -> Tuple[torch.Tensor, Dict[str, float]]:
    p = p.clone()
    try:
        f_idx = actions.index("FOLD")
    except ValueError:
        return p, {"fold_cap_applied": 0.0}

    # Decide a cap
    cap: Optional[float] = None
    if equity is not None:
        if equity >= 0.65:
            cap = cap_hi
        elif equity >= 0.55:
            cap = cap_mid

    if cap is None and ev_map:
        call_ev = ev_map.get("CALL", None)
        best_nonfold = max([v for a, v in ev_map.items() if a != "FOLD"], default=None)
        if best_nonfold is not None and (call_ev is None or best_nonfold >= call_ev):
            cap = cap_mid

    if cap is None and spr is not None and spr <= 4.0:
        cap = cap_low

    if cap is None:
        return p, {"fold_cap_applied": 0.0}

    cap = float(max(1e-4, min(0.9, cap)))
    if p[0, f_idx] <= cap:
        return p, {"fold_cap_applied": 0.0, "cap": cap}

    excess = float(p[0, f_idx] - cap)
    p[0, f_idx] = cap

    idxs = [i for i, a in enumerate(actions) if a != "FOLD" and (hero_mask is None or hero_mask[i] > 0.5)]
    denom = float(p[0, idxs].sum().item()) or 1e-12
    p[0, idxs] += (p[0, idxs] / denom) * excess

    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return p, {"fold_cap_applied": excess, "cap": cap}