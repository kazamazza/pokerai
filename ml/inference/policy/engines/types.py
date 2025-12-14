from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

@dataclass
class Baseline:
    actions: List[str]
    logits: torch.Tensor      # [1, V], router logits or log-probs
    mask_router: torch.Tensor # [V] in {0,1} float

@dataclass
class Masks:
    router: torch.Tensor      # [V]
    role: torch.Tensor        # [V]
    size: torch.Tensor        # [V]
    hero: torch.Tensor        # [V]

@dataclass
class SignalsPack:
    p_win: Optional[float]
    ex_probs: Optional[Tuple[float, float, float]]
    evs: Dict[str, float]
    size_frac: Optional[float]
    bet_menu_pcts: Optional[List[int]]
    spr: Optional[float]
    ctx: str
    side: str
    actor: str
    hero_is_ip: bool