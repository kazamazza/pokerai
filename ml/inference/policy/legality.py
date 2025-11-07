from __future__ import annotations
from typing import List, Optional
import torch

class Menus:
    def __init__(self, bet_sizes: Optional[List[float]], raise_buckets: Optional[List[int]], allow_allin: Optional[bool]):
        self.bet_sizes = bet_sizes
        self.raise_buckets = raise_buckets
        self.allow_allin = True if allow_allin is None else bool(allow_allin)

def build_mask(actions: List[str], *, actor: str, facing_bet: bool, menus: Menus) -> torch.Tensor:
    """Actor-aware, menu-aware mask aligned with actions."""
    m = torch.zeros(len(actions), dtype=torch.float32)
    for i, tok in enumerate(actions):
        T = tok.upper()
        legal = False
        if not facing_bet:
            if T == "CHECK":
                legal = True
            elif T.startswith("BET_"):
                if menus.bet_sizes is None:
                    legal = True
                else:
                    try:
                        pct = int(T.split("_", 1)[1])
                        legal = any(int(round(s * 100)) == pct for s in menus.bet_sizes)
                    except Exception:
                        legal = False
            elif T.startswith("DONK_"):
                legal = (actor == "oop")  # why: donk by OOP only
            elif T == "ALLIN":
                legal = menus.allow_allin
        else:
            if T in ("FOLD", "CALL"):
                legal = True
            elif T.startswith("RAISE_"):
                if menus.raise_buckets is None:
                    legal = True
                else:
                    try:
                        mult = int(T.split("_", 1)[1])
                        legal = mult in set(menus.raise_buckets)
                    except Exception:
                        legal = False
            elif T == "ALLIN":
                legal = menus.allow_allin
        if legal:
            m[i] = 1.0
    if m.sum().item() == 0:
        m.fill_(1.0)  # safety
    return m