import math
from typing import Optional, Sequence, List, Dict
import torch.nn.functional as F
import torch


def _masked_softmax_from_logits(z: torch.Tensor, mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Softmax over masked indices only."""
    z = z.clone()
    neg_inf = torch.finfo(z.dtype).min / 4
    z_masked = z.clone()
    off = (mask <= 0.5).nonzero(as_tuple=False).view(-1)
    if off.numel():
        z_masked[0, off] = neg_inf
    # stabilize
    z_masked[0] -= torch.max(z_masked[0][mask > 0.5])
    e = torch.exp(z_masked)
    e[0, off] = 0.0
    s = e.sum(dim=1, keepdim=True).clamp_min(eps)
    return e / s


def _aggressive_indices(tokens: Sequence[str], *, side: str, allow_allin: bool = True) -> List[int]:
    up = [t.upper() for t in tokens]
    idx: List[int] = []
    if side == "facing":
        idx = [i for i, t in enumerate(up) if t.startswith("RAISE_")]
        if allow_allin:
            idx += [i for i, t in enumerate(up) if t == "ALLIN"]
    else:  # root
        idx = [i for i, t in enumerate(up) if t.startswith("BET_")]
    return idx


def _single_delta_for_target_share(
    z: torch.Tensor,
    legal_idx: Sequence[int],
    chosen_idx: int,
    tau: float,
    max_logit_boost: float,
) -> float:
    """
    Closed-form delta so that softmax over legal places ~tau on chosen_idx.
    """
    if chosen_idx not in legal_idx:
        return 0.0
    # Z = sum_{k in legal} exp(z_k); a = exp(z_chosen)
    z_leg = z[0, legal_idx]
    m = torch.max(z_leg).item()
    exp_leg = torch.exp(z_leg - m)  # stabilize
    Z = float(exp_leg.sum().item())
    a = float(math.exp(float(z[0, chosen_idx].item()) - m))
    # p' = (a r) / ((Z - a) + a r), r = e^{delta} => solve for r
    tau = float(max(1e-6, min(1.0 - 1e-6, tau)))
    numerator = tau * (Z - a)
    denom = a * (1.0 - tau)
    if denom <= 0:
        return 0.0
    r = numerator / denom
    if r <= 1e-9:
        return 0.0
    delta = math.log(r)
    return float(max(0.0, min(max_logit_boost, delta)))


def _cap_token_prob(
    z: torch.Tensor,
    legal_idx: Sequence[int],
    cap_idx: int,
    p_cap: float,
) -> None:
    """
    Reduce probability of index 'cap_idx' to <= p_cap by subtracting gamma from its logit.
    """
    if cap_idx not in legal_idx:
        return
    p = _masked_softmax_from_logits(z, _mask_from_indices(z, legal_idx))[0]
    cur = float(p[cap_idx].item())
    if cur <= p_cap:
        return

    # Let X = sum_{k!=cap} exp(z_k), a = exp(z_cap)
    z_leg = z[0, legal_idx]
    m = torch.max(z_leg).item()
    exp_leg = torch.exp(z_leg - m)
    # map back to indices
    cap_local = legal_idx.index(cap_idx)
    a = float(exp_leg[cap_local].item())
    X = float(exp_leg.sum().item() - a)
    p_cap = float(max(1e-6, min(1.0 - 1e-6, p_cap)))
    # p' = a * e^{-g} / (X + a * e^{-g}) => r = e^{-g} = p' X / (a (1 - p'))
    r = (p_cap * X) / (a * (1.0 - p_cap))
    if r <= 0:
        return
    gamma = -math.log(max(r, 1e-12))
    if gamma <= 1e-9:
        return
    z[0, cap_idx] -= float(gamma)


def _mask_from_indices(z: torch.Tensor, legal_idx: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros_like(z[0])
    mask[legal_idx] = 1.0
    return mask