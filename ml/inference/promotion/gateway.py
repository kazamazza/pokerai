# ml/inference/promotion/gateway.py
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any
import math
import numpy as np
import torch


from ml.inference.promotion.config import PromotionConfig
from ml.inference.promotion.helpers import _masked_softmax_from_logits, _aggressive_indices, \
    _single_delta_for_target_share, _cap_token_prob


class PromotionGateway:
    def __init__(self, cfg: PromotionConfig | None = None):
        self.cfg = cfg or PromotionConfig.default()

    # ---- score / tau ------------------------------------------------------
    def _exploit_bias(self, ex_probs: Optional[Sequence[float]], *, want_aggro: bool) -> float:
        if not ex_probs or len(ex_probs) < 3:
            return 0.0
        pf, pc, pr = [float(x) for x in ex_probs[:3]]  # [F,C/R]
        return max(0.0, (pr if want_aggro else pf) - max(pc, (pf if want_aggro else pr)))

    def _score(
        self,
        ev_gap_bb: float,
        p_win: Optional[float],
        ex_probs: Optional[Sequence[float]],
        *,
        want_aggro: bool,
    ) -> float:
        # EV (scaled & capped to [0,1])
        if self.cfg.ev_is_centi_bb:
            ev_gap_bb = ev_gap_bb / 100.0
        ev_term = min(max(ev_gap_bb, 0.0), self.cfg.ev_cap_bb) / max(self.cfg.ev_cap_bb, 1e-6)

        # Equity
        eq_term = 0.0
        if p_win is not None:
            eq_term = max(0.0, float(p_win) - self.cfg.eq_gate) / max(1.0 - self.cfg.eq_gate, 1e-6)

        # Exploit
        expl_term = 0.0
        if ex_probs is not None:
            bias = self._exploit_bias(ex_probs, want_aggro=want_aggro) - self.cfg.expl_gate
            expl_term = max(0.0, bias) / max(1.0 - self.cfg.expl_gate, 1e-6)

        s = (self.cfg.w_ev * ev_term) + (self.cfg.w_eq * eq_term) + (self.cfg.w_expl * expl_term)
        return float(max(0.0, min(1.0, s)))

    def _tau_from_score(self, score: float) -> float:
        t = self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) * float(score)
        return float(max(self.cfg.tau_min, min(self.cfg.tau_max, t)))

    # ---- postflop ---------------------------------------------------------
    def promote_postflop(
        self,
        *,
        tokens: Sequence[str],
        base_logits: torch.Tensor,
        hero_mask: torch.Tensor,
        side: str,
        actor: str,
        p_win: Optional[float],
        ex_probs: Optional[Sequence[float]],
        evs: Dict[str, float],
        size_frac: Optional[float] = None,
        bet_menu_pcts: Optional[Sequence[int]] = None,
        ctx: Optional[str] = None,
        spr: Optional[float] = None,
        allow_allin: bool = False,
        action_ctx: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Adjust postflop logits (torch [1,V]) with a single-bucket promotion when justified.
        """
        z = base_logits
        V = len(tokens)
        mask = (hero_mask > 0.5).float()
        legal_idx = [i for i in range(V) if mask[i] > 0.5]
        dbg: Dict[str, Any] = {
            "applied": False,
            "why": None,
            "ctx": ctx,
            "actor": actor,
            "spr": spr,
            "size_frac": size_frac,
        }

        if not legal_idx:
            dbg["why"] = "no_legal"
            return {"logits": z, "debug": dbg}

        up = [a.upper() for a in tokens]
        p_base = _masked_softmax_from_logits(z, mask)
        base_idx = int(torch.argmax(p_base[0]))
        base_tok = up[base_idx]

        # Derive passive/aggressive sets
        passive_tok = "CALL" if side == "facing" else "CHECK"
        aggr_idx = _aggressive_indices(tokens, side=side, allow_allin=allow_allin)

        # Root bet menu filtering
        if side == "root" and bet_menu_pcts:
            bet_allow = set(f"BET_{int(x)}" for x in bet_menu_pcts)
            aggr_idx = [i for i in aggr_idx if tokens[i].upper() in bet_allow]

        if not aggr_idx:
            dbg["why"] = "no_aggressive_in_menu"
            return {"logits": z, "debug": dbg}

        # EV gap
        passive_ev = float(evs.get(passive_tok, evs.get(passive_tok.capitalize(), 0.0)))
        best_i, best_ev = None, None
        for i in aggr_idx:
            v = float(evs.get(tokens[i], evs.get(tokens[i].upper(), float("-inf"))))
            if (best_ev is None) or (v > best_ev):
                best_ev, best_i = v, i
        if best_i is None or best_ev is None or not math.isfinite(best_ev):
            dbg["why"] = "no_aggressive_ev"
            return {"logits": z, "debug": dbg}

        ev_gap = max(0.0, best_ev - passive_ev)

        # Respect baseline fold only if NOT strong
        strong_by_eq = (p_win is not None) and (float(p_win) >= self.cfg.strong_eq_floor)
        strong_by_ev = (ev_gap >= self.cfg.strong_ev_margin_bb)
        strong = bool(strong_by_eq or strong_by_ev)

        if side == "facing" and self.cfg.respect_fold_when_facing and base_tok == "FOLD" and not strong:
            dbg.update({"why": "respect_fold_baseline", "baseline": base_tok})
            return {"logits": z, "debug": dbg}

        # Score → tau
        score = self._score(ev_gap, p_win, ex_probs, want_aggro=True)
        tau = self._tau_from_score(score)
        if tau <= 0.0:
            dbg.update({"why": "score_zero", "score": score})
            return {"logits": z, "debug": dbg}

        # Promotion
        chosen_idx = best_i
        delta = _single_delta_for_target_share(z, legal_idx, chosen_idx, tau, self.cfg.max_logit_boost)
        if delta <= 0.0:
            dbg.update({"why": "delta_zero", "score": score, "tau": tau})
            return {"logits": z, "debug": dbg}

        z = z.clone()
        z[0, chosen_idx] += float(delta)

        # Gentle temperature nudging on legal set
        p_after = _masked_softmax_from_logits(z, mask)
        spread = float(p_after[0, legal_idx].max() - p_after[0, legal_idx].min())
        T = max(self.cfg.min_temp, min(self.cfg.max_temp, 1.2 - 0.8 * spread))
        if abs(T - 1.0) > 1e-3:
            z[0, legal_idx] /= T

        # Strong-hand no-fold guard
        if side == "facing" and "FOLD" in up and strong:
            fold_idx = up.index("FOLD")
            _cap_token_prob(z, legal_idx, fold_idx, self.cfg.fold_cap_when_strong)
            p_after = _masked_softmax_from_logits(z, mask)

        dbg.update({
            "applied": True,
            "score": round(score, 3),
            "tau": round(tau, 3),
            "delta": round(float(delta), 3),
            "best_bucket": tokens[chosen_idx],
            "ev_gap_bb": round(ev_gap, 3),
            "baseline": base_tok,
            "T": round(T, 3),
            "strong_guard": strong,
        })
        return {"logits": z, "debug": dbg}

    # ---- preflop ----------------------------------------------------------
    def promote_preflop(
        self,
        *,
        base_logits: np.ndarray,
        tokens: Sequence[str],
        evs: Dict[str, float],
        p_win: Optional[float],
        facing_bet: bool,
        free_check: bool,
        allow_allin: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adjusts preflop logits (numpy) with a single OPEN_/RAISE_ promotion.
        Returns (new_logits, debug).
        """
        dbg: Dict[str, Any] = {"applied": False, "why": None}
        if not tokens:
            dbg["why"] = "no_tokens"
            return base_logits, dbg

        up = [t.upper() for t in tokens]
        z = torch.tensor(base_logits, dtype=torch.float32).view(1, -1)
        mask = torch.ones(len(tokens), dtype=torch.float32)

        passive_tok = ("CALL" if facing_bet else ("CHECK" if free_check else "FOLD"))
        passive_ev = float(evs.get(passive_tok, evs.get(passive_tok.capitalize(), 0.0)))

        # Aggressive set
        if facing_bet:
            aggr_idx = [i for i, t in enumerate(up) if t.startswith("RAISE_") or (t == "ALLIN" and allow_allin)]
        else:
            aggr_idx = [i for i, t in enumerate(up) if t.startswith("OPEN_")]

        if not aggr_idx:
            dbg["why"] = "no_aggressive_bucket"
            return base_logits, dbg

        # Choose by EV
        best_i, best_ev = None, None
        for i in aggr_idx:
            v = float(evs.get(tokens[i], evs.get(up[i], float("-inf"))))
            if (best_ev is None) or (v > best_ev):
                best_ev, best_i = v, i
        if best_i is None or best_ev is None or not math.isfinite(best_ev):
            dbg["why"] = "no_aggressive_ev"
            return base_logits, dbg

        ev_gap = max(0.0, best_ev - passive_ev)
        score = self._score(ev_gap, p_win, None, want_aggro=True)  # exploit off preflop
        tau = self._tau_from_score(score)
        if tau <= 0.0:
            dbg.update({"why": "score_zero", "score": score})
            return base_logits, dbg

        legal_idx = [i for i in range(len(tokens)) if mask[i] > 0.5]
        delta = _single_delta_for_target_share(z, legal_idx, best_i, tau, self.cfg.max_logit_boost)
        if delta <= 0.0:
            dbg.update({"why": "delta_zero", "score": score, "tau": tau})
            return base_logits, dbg

        z = z.clone()
        z[0, best_i] += float(delta)

        # Gentle temp
        p_after = _masked_softmax_from_logits(z, mask)
        spread = float(p_after.max() - p_after.min())
        T = max(self.cfg.min_temp, min(self.cfg.max_temp, 1.2 - 0.8 * spread))
        if abs(T - 1.0) > 1e-3:
            z[0, legal_idx] /= T

        dbg.update({
            "applied": True,
            "score": round(score, 3),
            "tau": round(tau, 3),
            "delta": round(float(delta), 3),
            "best_bucket": tokens[best_i],
            "ev_gap_bb": round(ev_gap, 3),
            "T": round(T, 3),
        })
        return z[0].detach().cpu().numpy(), dbg