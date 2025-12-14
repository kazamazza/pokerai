from typing import Any, Dict, List, Optional, Tuple

class ContextInferer:
    VALID = {"VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"}

    @staticmethod
    def _norm_ctx(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        s = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
        # tolerant aliases you mentioned earlier
        if s in {"BLIND_VS_STEAL", "STEAL", "SRP", "SINGLE_RAISED"}:
            return "VS_OPEN"
        if s in {"LIMPED", "LIMP"}:
            return "LIMPED_SINGLE"
        return s if s in ContextInferer.VALID else None

    @staticmethod
    def _infer_from_actions_hist(req: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Strictly use structured history to classify the *preflop* situation:
          - count preflop RAISE/BET events
          - detect limped pots (no raises, at least one CALL/CHECK)
          - map counts to VS_OPEN / VS_3BET / VS_4BET
        """
        hist = getattr(req, "actions_hist", None) or []
        if not hist:
            # no history → safest default that matches most training: single-raised
            return "VS_OPEN", {"source": "default_no_history"}

        n_raises = 0
        saw_preflop_call_or_check = False

        for e in hist:
            try:
                street = int(getattr(e, "street", 0) or 0)
            except Exception:
                street = 0
            if street != 0:
                continue  # preflop only

            a = str(getattr(e, "action", "")).upper()
            if a in {"RAISE", "BET"}:
                n_raises += 1
            elif a in {"CALL", "CHECK"}:
                saw_preflop_call_or_check = True

        if n_raises == 0:
            if saw_preflop_call_or_check:
                return "LIMPED_SINGLE", {"source": "hist", "n_raises": n_raises, "limp": True}
            # No preflop activity recorded at all → default to VS_OPEN
            return "VS_OPEN", {"source": "hist_empty_preflop", "n_raises": 0}

        if n_raises == 1:
            return "VS_OPEN", {"source": "hist", "n_raises": 1}
        if n_raises == 2:
            return "VS_3BET", {"source": "hist", "n_raises": 2}
        # 3 or more
        return "VS_4BET", {"source": "hist", "n_raises": n_raises}

    @classmethod
    def infer_with_reason(cls, req: Any) -> Tuple[str, Dict[str, Any]]:
        # 1) If the request explicitly sets ctx, honor it (normalized)
        ctx_norm = cls._norm_ctx(getattr(req, "ctx", None))
        if ctx_norm:
            return ctx_norm, {"source": "explicit", "matched": ctx_norm}

        # 2) Else infer from structured preflop history
        return cls._infer_from_actions_hist(req)