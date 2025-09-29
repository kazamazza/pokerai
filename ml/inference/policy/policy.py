from __future__ import annotations
from typing import Any, Dict
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.preflop import PreflopPolicy, PreflopDeps
from ml.inference.policy.types import Action, PolicyRequest, PolicyResponse

ACTION_VOCAB = []


class PolicyInfer:
    def __init__(self, deps: PolicyInferDeps):
        if deps.exploit is None:
            raise ValueError("exploit infer is required")
        if deps.equity is None:
            raise ValueError("equity infer is required")
        if deps.range_pre is None:
            raise ValueError("range_pre (RangeNetPreflopInfer) is required")

        # deps
        self.pol_post   = deps.policy_post
        self.pop        = deps.pop
        self.expl       = deps.exploit
        self.eq         = deps.equity
        self.rng_pre    = deps.range_pre
        self.clusterer  = deps.clusterer
        self.p          = deps.params or {}

        # preflop module (stateless facade)
        self._preflop = PreflopPolicy(PreflopDeps(range_pre=self.rng_pre, equity=self.eq))

        # action vocab (used only by stubbed postflop for now)
        try:
            from ml.models.postflop_policy_net import ACTION_VOCAB as _VOC
            self.action_vocab = list(_VOC)
        except Exception:
            self.action_vocab = ["CHECK", "BET_33", "BET_66", "FOLD", "CALL", "RAISE_200", "ALLIN"]

    # --- tiny encoder to keep your existing string action wire-format ---
    def _encode_action(self, a: Action) -> str:
        k = a.kind.upper()
        if k in ("FOLD", "CHECK", "CALL", "ALLIN"):
            return k
        if k == "BET":
            if a.size_pct is not None:
                return f"BET_{int(round(a.size_pct))}"
            # fallback: try bb → rough pct placeholder (unsafe but keeps shape)
            if a.size_bb is not None:
                return "BET_50"
            return "BET_33"
        if k == "RAISE":
            if a.size_mult is not None:
                # map common preflop sizes to your legacy tokens
                m = round(float(a.size_mult), 2)
                if abs(m - 2.0) < 1e-3: return "RAISE_200"
                if abs(m - 2.5) < 1e-3: return "RAISE_250"
                if abs(m - 3.0) < 1e-3: return "RAISE_300"
                return "RAISE_300"
            if a.size_pct is not None:
                return f"RAISE_{int(round(a.size_pct))}"
            return "RAISE_200"
        return k

    def predict(self, req_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize inbound to a PolicyRequest once
        req = PolicyRequest(**req_dict)

        # Best-effort street inference if missing / malformed
        try:
            street = int(req.street or 0)
        except Exception:
            street = 0

        if street not in (0, 1, 2, 3):
            board = (req.board or "").strip()
            n = len(board)
            street = 3 if n >= 10 else 2 if n >= 8 else 1 if n >= 6 else 0

        if street == 0:
            # ---- Preflop path ----
            out: PolicyResponse = self._preflop.predict(req).normalized()
            return {
                "actions": [self._encode_action(a) for a in out.actions],
                "probs": out.probs,
                "evs": out.evs,
                "debug": out.debug,
                "notes": out.notes,
            }
        return self._predict_postflop(dict(req_dict, street=street))