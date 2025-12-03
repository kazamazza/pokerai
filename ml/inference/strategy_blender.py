from typing import List, Optional

import torch


class StrategyBlender:
    """
    Applies temperature blending logic based on game dynamics.
    Returns adjusted temperature to be used when sampling from logits.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        actions: List[str],
        hero_mask: torch.Tensor,
        temperature: float = 1.0,
        min_temp: float = 0.2,
        max_temp: float = 1.5,
    ):
        self.logits = logits
        self.actions = actions
        self.hero_mask = hero_mask
        self.initial_temperature = temperature
        self.min_temp = min_temp
        self.max_temp = max_temp

    def adjust_temperature(
        self,
        eff_stack_bb: Optional[float] = None,
        pot_bb: Optional[float] = None,
        delta: Optional[float] = None,
        dbg_out: Optional[dict] = None,
    ) -> float:
        """
        Returns adjusted temperature based on EV spread (delta) and SPR.
        Designed to soften or sharpen action probabilities before sampling.
        """
        temp = self.initial_temperature

        # Adjust based on EV spread
        if delta is not None:
            if delta > 0.25:
                temp *= 0.85  # Sharper distribution
            elif delta < 0.05:
                temp *= 1.15  # More exploratory

        # Optional: adjust based on SPR (stack-to-pot ratio)
        if eff_stack_bb is not None and pot_bb is not None:
            spr = eff_stack_bb / max(pot_bb, 1e-4)
            if spr > 6:
                temp *= 1.1
            elif spr < 2:
                temp *= 0.9
            if dbg_out is not None:
                dbg_out["spr"] = round(spr, 2)

        # Clamp temperature
        temp = max(self.min_temp, min(self.max_temp, temp))

        if dbg_out is not None:
            dbg_out["adjusted_temperature"] = round(temp, 3)

        return temp