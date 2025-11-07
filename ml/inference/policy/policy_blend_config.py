from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from ml.inference.policy.action_vocab import ActionVocab

@dataclass
class PolicyBlendConfig:
    """Blend config applied once at the orchestrator."""
    temperature: float = 1.0
    min_legal_prob: float = 1e-6
    tie_mix_threshold: float = 0.02
    epsilon_explore: float = 0.00
    lambda_eq: float = 0.0
    eq_min_abs_margin: float = 0.01
    eq_max_logit_delta: float = 2.0
    lambda_expl: float = 0.0
    risk_floor_stack_bb: float = 0.0
    max_allin_freq: float = 0.20
    equity_nudge_pre: float = 0.02  # preflop only

    @classmethod
    def default(cls) -> "PolicyBlendConfig":
        return cls()

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, T: float, eps: float) -> torch.Tensor:
    logits = logits if logits.dim() == 2 else logits.unsqueeze(0)
    mask = mask if mask.dim() == 2 else mask.unsqueeze(0)
    big_neg = torch.finfo(logits.dtype).min / 4
    masked = torch.where(mask > 0.5, logits / max(T, 1e-6), big_neg)
    p = F.softmax(masked, dim=-1)
    p = p * (1 - eps) + (eps / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)) * (mask > 0.5).to(p.dtype)
    p = p * mask
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return p

def mix_ties_if_close(p: torch.Tensor, thresh: float) -> torch.Tensor:
    p = p if p.dim() == 2 else p.unsqueeze(0)
    k = min(2, p.size(-1))
    if k < 2: return p
    top2 = torch.topk(p, k=k, dim=-1)
    close = (top2.values[:, 0] - top2.values[:, 1]).abs() <= thresh
    if not close.any(): return p
    for b in torch.nonzero(close).view(-1):
        i1, i2 = int(top2.indices[b, 0]), int(top2.indices[b, 1])
        mass = p[b, i1] + p[b, i2]
        p[b, :] *= 0.0
        p[b, i1] = 0.6 * mass
        p[b, i2] = 0.4 * mass
    return p

def epsilon_explore(p: torch.Tensor, eps: float, mask: torch.Tensor) -> torch.Tensor:
    if eps <= 0: return p
    p = p if p.dim() == 2 else p.unsqueeze(0)
    mask = mask if mask.dim() == 2 else mask.unsqueeze(0)
    uni = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    p = (1 - eps) * p + eps * uni
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return p

def cap_allins(p: torch.Tensor, actions: List[str], eff_stack_bb: float, *, risk_floor_stack_bb: float, max_allin_freq: float) -> torch.Tensor:

    p = p if p.dim() == 2 else p.unsqueeze(0)
    if eff_stack_bb <= risk_floor_stack_bb:
        return p
    av = ActionVocab.from_actions(actions)
    idx = av.allin_idx()
    if idx is None: return p
    if p[0, idx] <= max_allin_freq: return p
    over = p[0, idx] - max_allin_freq
    p = p.clone()
    p[0, idx] = max_allin_freq
    rest = [i for i in range(len(actions)) if i != idx]
    denom = float(p[0, rest].sum().item()) or 1e-12
    scale = 1.0 + (over / denom)
    p[0, rest] *= scale
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return p
