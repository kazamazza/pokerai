from __future__ import annotations

import numpy as np
import torch

from ml.features.hands import hand169_id_from_hand_code
from ml.inference.policy.engines.postflop import PromotionApplier
from ml.inference.policy.helpers import compute_temperature
from ml.inference.policy.types import PolicyResponse
from ml.inference.preflop_legal_action_generator import PreflopLegalActionGenerator


class PreflopEngine:
    def __init__(self, ev_router, range_pre, promo: PromotionApplier):
        self.ev_router = ev_router
        self.range_pre = range_pre
        self.promo = promo

    @torch.no_grad()
    def run(self, req, eq_sig) -> "PolicyResponse":
        import numpy as np
        import torch

        # ------------------ 1) Normalize facing size (preflop) ------------------
        facing_bet = bool(getattr(req, "facing_bet", False))
        if facing_bet:
            if getattr(req, "faced_size_frac", None) is None and getattr(req, "faced_size_pct", None) is not None:
                try:
                    req.faced_size_frac = float(req.faced_size_pct) / 100.0
                except Exception:
                    req.faced_size_frac = None
            if getattr(req, "faced_size_frac", None) is None:
                req.faced_size_frac = 0.33
        faced_frac = float(getattr(req, "faced_size_frac", 0.0) or 0.0)

        # ------------------ 2) Menu generation ------------------
        stack_bb = float(getattr(req, "eff_stack_bb", None) or getattr(req, "pot_bb", None) or 100.0)
        hero_pos = (getattr(req, "hero_pos", "") or "").upper()
        free_check = (not facing_bet) and (hero_pos == "BB")
        allow_allin = bool(getattr(req, "allow_allin", False))

        gen = PreflopLegalActionGenerator(
            open_sizes_cbb=(200, 250, 300),
            raise_totals_cbb=(600, 750, 900, 1200),
            allow_allin=allow_allin,
            max_open_cbb=None,
        )
        tokens = gen.generate(
            stack_bb=stack_bb,
            facing_bet=facing_bet,
            faced_frac=faced_frac,
            free_check=free_check,
        )
        if not tokens:
            tokens = ["FOLD"] + (["CHECK"] if free_check else [])

        # Clamp to EV-router vocab (if present)
        try:
            allowed = set(self.ev_router.vocab("preflop"))
            filtered = [t for t in tokens if t in allowed]
            tokens = filtered or tokens
        except Exception:
            pass

        # Facing fallback: ensure some RAISE_* exist if stack allows
        if facing_bet and not any(t.startswith("RAISE_") or t == "ALLIN" for t in tokens):
            fallback = [f"RAISE_{r}" for r in (600, 750, 900, 1200) if (r / 100.0) < stack_bb]
            try:
                allowed = set(self.ev_router.vocab("preflop"))
                fallback = [t for t in fallback if t in allowed] or fallback
            except Exception:
                pass
            if fallback:
                tokens = list(dict.fromkeys(tokens + fallback))

        # ------------------ 3) EVs from router ------------------
        if not getattr(self, "ev_router", None):
            raise RuntimeError("EV router not attached.")
        ev_out = self.ev_router.predict(req, side="preflop", tokens=tokens)
        ev_values = np.asarray(ev_out.evs, dtype=np.float32) if (ev_out and ev_out.available) \
            else np.zeros(len(tokens), np.float32)

        ev_values_raw = ev_values.copy()

        # ---- Cost-correction shim: convert state-values -> net EV (price to continue) ----
        posted = 1.0 if hero_pos == "BB" else (0.5 if hero_pos == "SB" else 0.0)

        def _total_from_token(tok: str):
            if tok.startswith(("OPEN_", "RAISE_")):
                try:
                    return int(tok.split("_", 1)[1]) / 100.0  # e.g., RAISE_750 → 7.5bb total
                except Exception:
                    return None
            return None

        S = float(faced_frac) if facing_bet else 0.0  # size we face (bb), e.g. 2.5

        for i, tok in enumerate(tokens):
            if tok == "CALL":
                ev_values[i] -= max(S - posted, 0.0)
            elif tok.startswith("OPEN_") or tok.startswith("RAISE_"):
                tot = _total_from_token(tok)
                if tot is not None:
                    ev_values[i] -= max(tot - posted, 0.0)
            elif tok == "ALLIN":
                ev_values[i] -= max(stack_bb - posted, 0.0)

        # anchors
        for i, tok in enumerate(tokens):
            if tok == "FOLD" or (tok == "CHECK" and free_check):
                ev_values[i] = 0.0

        # ------------------ 4) Baseline logits from EV ------------------
        T_base = float(compute_temperature(ev_values))
        base_logits = ev_values.astype(np.float32) / max(T_base, 1e-6)
        if base_logits.size:
            base_logits -= float(base_logits.max())

        # Deep-stack ALLIN guard
        if stack_bb > 20.0 and "ALLIN" in tokens:
            base_logits[tokens.index("ALLIN")] += -8.0

        # ------------------ 5) Optional range prior nudge ------------------
        range_dbg = None
        if getattr(self, "range_pre", None) is not None and getattr(req, "hero_hand", None):
            try:
                rvec = self.range_pre.predict(req, quiet=True)  # 169-d
                hid = hand169_id_from_hand_code(req.hero_hand)
                if hid is not None and 0 <= hid < rvec.size:
                    order = np.argsort(rvec)
                    pos = np.where(order == hid)[0]
                    if pos.size:
                        rank = float(pos[0]) / max(rvec.size - 1, 1)  # 0..1
                        for i, t in enumerate(tokens):
                            if t.startswith(("OPEN_", "RAISE_")):
                                base_logits[i] += 0.4 * rank
                            elif t == "CALL":
                                base_logits[i] += 0.2 * (1.0 - abs(rank - 0.5) * 2.0)
                            elif t == "FOLD":
                                base_logits[i] -= 0.3 * rank
            except Exception as e:
                range_dbg = {"error": str(e)}

        # ------------------ 6) Promotion (override mode) ------------------
        promo_dbg = None
        try:
            if not hasattr(self, "promo"):
                promoter = getattr(self, "promoter", None)
                self.promo = promoter if isinstance(promoter, PromotionApplier) else PromotionApplier(promoter)
            ev_map = {t: float(ev_values[i]) for i, t in enumerate(tokens)}
            new_logits_np, promo_dbg = self.promo.preflop(
                base_logits=base_logits,
                tokens=tokens,
                evs=ev_map,
                p_win=(eq_sig.p_win if (eq_sig and eq_sig.available) else None),
                facing_bet=facing_bet,
                free_check=free_check,
                allow_allin=allow_allin,
            )
            base_logits = new_logits_np.astype(np.float32)
        except Exception as e:
            promo_dbg = {"error": str(e)}

        # ------------------ 7) Probabilities ------------------
        promo_applied = bool(promo_dbg and promo_dbg.get("applied"))
        if promo_applied:
            # In override mode, base_logits are LOG-PROBS for legal actions.
            probs_np = np.exp(base_logits)
            s = float(probs_np.sum())
            if not np.isfinite(s) or s <= 0.0:
                probs_np = np.ones(len(tokens), np.float32) / float(len(tokens))
        else:
            # Baseline path: standard softmax
            if base_logits.size:
                base_logits -= float(base_logits.max())
            probs_np = np.exp(base_logits)
            Z = float(probs_np.sum())
            probs_np = (probs_np / Z) if (Z > 0 and np.isfinite(Z)) else np.ones(len(tokens), np.float32) / float(
                len(tokens))

        probs = probs_np.astype(float).tolist()

        # ------------------ 8) Selection: promotion-first ------------------
        best_idx_dist = int(np.argmax(probs_np))
        best_action_dist = tokens[best_idx_dist]
        best_action = best_action_dist
        best_action_source = "distribution"
        promo_best = promo_dbg.get("best_bucket") if promo_applied else None
        if promo_applied and isinstance(promo_best, str) and promo_best in tokens:
            best_action = promo_best
            best_action_source = "promotion"

        # ------------------ 9) Debug ------------------
        debug = None
        if getattr(req, "debug", False):
            debug = {
                "tokens": list(tokens),
                "ev_values": [float(x) for x in ev_values],  # cost-corrected
                "ev_values_raw": [float(x) for x in ev_values_raw],  # from EV router
                "temp": T_base,
                "facing_bet": facing_bet,
                "faced_frac": faced_frac,
                "free_check": free_check,
                "allow_allin": allow_allin,
                "stack_bb": stack_bb,
                "equity": (eq_sig.p_win if (eq_sig and eq_sig.available) else None),
                "promotion": promo_dbg,
                "range_dbg": range_dbg,
                "ctx": getattr(req, "ctx", None),
                "final_T": 1.0,
                "final_eps": 0.0,
                "selection": {
                    "best_action": best_action,
                    "best_action_source": best_action_source,
                    "best_action_distribution": best_action_dist,
                    "promotion_best_bucket": promo_best,
                },
            }

        return PolicyResponse(
            actions=list(tokens),
            probs=probs,
            evs=[float(x) for x in ev_values],  # cost-corrected EVs
            best_action=best_action,
            notes=[
                f"preflop policy (EV→logits → PromotionGateway[override]); promoter={'on' if hasattr(self, 'promo') and self.promo else 'off'}"],
            debug=debug,
        )