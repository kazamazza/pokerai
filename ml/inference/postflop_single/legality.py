from __future__ import annotations
from typing import Iterable, Optional, Sequence, Set
import torch

ROOT_TOKENS: Set[str] = {"CHECK", "BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100", "DONK_33", "ALLIN"}
FACING_TOKENS: Set[str] = {"FOLD", "CALL", "RAISE_150", "RAISE_200", "RAISE_300", "RAISE_400", "RAISE_500", "ALLIN"}


def mask_root(actions: Sequence[str], *, actor: str, bet_menu: Optional[Sequence[float]]) -> torch.Tensor:
    legal = set(ROOT_TOKENS)
    if actor.lower() != "oop":
        legal.discard("DONK_33")

    if bet_menu:
        want = set()

        def has(x: float) -> bool:
            return any(abs(float(s) - x) < 1e-3 for s in bet_menu)

        if has(0.25): want.add("BET_25")
        if has(0.33): want.update({"BET_33", "DONK_33"})
        if has(0.50): want.add("BET_50")
        if has(0.66): want.add("BET_66")
        if has(0.75): want.add("BET_75")
        if has(1.00): want.add("BET_100")

        for b in {"BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100"}:
            if b not in want:
                legal.discard(b)
        if "DONK_33" not in want:
            legal.discard("DONK_33")

    m = torch.zeros(len(actions), dtype=torch.float32)
    for i, a in enumerate(actions):
        if a in legal:
            m[i] = 1.0
    if m.sum().item() == 0:
        m.fill_(1.0)
    return m


def mask_facing(actions: Sequence[str]) -> torch.Tensor:
    legal = set(FACING_TOKENS)
    m = torch.zeros(len(actions), dtype=torch.float32)
    for i, a in enumerate(actions):
        if a in legal:
            m[i] = 1.0
    if m.sum().item() == 0:
        m.fill_(1.0)
    return m