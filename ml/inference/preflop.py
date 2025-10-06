from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ml.features.hands import hand_to_169_label
from ml.inference.policy.types import PolicyRequest, PolicyResponse, Action


@dataclass
class PreflopDeps:
    range_pre: Any         # must expose .predict_proba([row]) -> Tensor/ndarray [1,169]
    equity: Optional[Any] = None  # optional EquityNet, must expose .predict([{...}]) -> [(pwin,ptie,plose)]

class PreflopPolicy:
    """
    Stateless facade around your preflop submodels.
    Keeps the orchestration simple and returns normalized PolicyResponse.
    """
    def __init__(self, deps: PreflopDeps):
        if deps.range_pre is None:
            raise ValueError("PreflopPolicy requires range_pre")
        self.rng = deps.range_pre
        self.eq  = deps.equity

    def _encode_action(self, a: Action) -> str:
        """
        Convert an Action object into a string token matching ACTION_VOCAB format.
        This keeps output consistent across preflop/postflop models.
        """
        k = str(a.kind).upper()

        # 1️⃣ simple atomic actions
        if k in ("FOLD", "CHECK", "CALL", "ALLIN"):
            return k

        # 2️⃣ bets (by % or BB)
        if k == "BET":
            if a.size_pct is not None:
                pct = int(round(a.size_pct))
                if pct in (25, 33, 50, 66, 75, 100):
                    return f"BET_{pct}"
                # normalize arbitrary to closest common
                nearest = min([25, 33, 50, 66, 75, 100], key=lambda x: abs(x - pct))
                return f"BET_{nearest}"
            elif a.size_bb is not None:
                return "BET_50"  # generic fallback if absolute only
            return "BET_33"

        # 3️⃣ raises (by multiple or %)
        if k == "RAISE":
            if a.size_mult is not None:
                mult = round(float(a.size_mult), 2)
                # map common preflop multipliers to legacy bucket tokens
                if abs(mult - 1.5) < 0.1: return "RAISE_150"
                if abs(mult - 2.0) < 0.1: return "RAISE_200"
                if abs(mult - 3.0) < 0.1: return "RAISE_300"
                if abs(mult - 4.0) < 0.1: return "RAISE_400"
                if abs(mult - 5.0) < 0.1: return "RAISE_500"
                return "RAISE_200"
            if a.size_pct is not None:
                pct = int(round(a.size_pct))
                return f"RAISE_{pct}"
            return "RAISE_200"

        # 4️⃣ fallback (unknown)
        return k

    def predict(self, req: PolicyRequest) -> PolicyResponse:
        # ---- Build range-net row (tokens are fine; the infer handles id mapping) ----
        row = {
            "stack_bb": float(req.stack_bb or req.eff_stack_bb or 100.0),
            "hero_pos": (req.hero_pos or "").upper(),
            "opener_pos": (req.opener_pos or "").upper(),
            "opener_action": (req.opener_action or "RAISE").upper(),
            "ctx": (req.ctx or "SRP").upper(),
        }

        rng = self.rng.predict_proba([row])  # [1,169] tensor/ndarray
        if hasattr(rng, "detach"):
            rng = rng.detach().cpu().numpy()
        rng_169: List[float] = (rng[0].tolist() if len(rng) else [1.0 / 169.0] * 169)

        # Optional: how much mass the hero’s exact hand has in villain range
        hero_mass = None
        if req.hero_hand:
            label = hand_to_169_label(req.hero_hand)
            idx = getattr(self.rng, "hand_to_id", {}).get(label) if hasattr(self.rng, "hand_to_id") else None
            if idx is not None and 0 <= idx < 169:
                hero_mass = float(rng_169[idx])

        # ---- Simple action prior (stub) ----
        facing_open = bool(req.facing_open) or (
            row["opener_action"] == "RAISE" and row["hero_pos"] in ("BB", "SB")
        )

        if facing_open:
            # FOLD / CALL / 3-bet (as 3x last raise amount—use size_mult)
            actions = [
                Action("FOLD"),
                Action("CALL"),
                Action("RAISE", size_mult=3.0),
            ]
            probs = [0.35, 0.40, 0.25]
        else:
            # Open or fold (2.5x open size as absolute raise-to in bb)
            actions = [
                Action("FOLD"),
                Action("RAISE", size_bb=2.5),
            ]
            probs = [0.30, 0.70]

        # ---- Optional equity call (neutral if not wired) ----
        equity = {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}
        if self.eq and req.hero_hand:
            try:
                out = self.eq.predict([{"street": 0, "hand_id": hand_to_169_label(req.hero_hand)}])
                if out:
                    p_win, p_tie, p_lose = out[0]
                    equity = {"p_win": float(p_win), "p_tie": float(p_tie), "p_lose": float(p_lose)}
            except Exception:
                pass

        # actions may be List[Action]; convert to wire tokens (strings)
        tokens = [
            self._encode_action(a) if isinstance(a, Action) else str(a)
            for a in actions
        ]

        resp = PolicyResponse(
            actions=tokens,  # List[str]
            probs=[float(x) for x in probs],  # ensure plain floats
            evs=[0.0] * len(tokens),  # keep stub EVs aligned with actions
            notes=["preflop stub; villain 169-range included for downstream"],
            debug={
                "street": 0,
                "villain_range_169": rng_169,
                "hero_prior_mass_in_villain_range": hero_mass,
                "equity": equity,
            },
        )

        return resp