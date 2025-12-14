# ml/inference/promotion/gateway.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from ml.inference.promotion.config import PromotionConfig
from ml.inference.promotion.helpers import _aggressive_indices, \
    _cap_token_prob


class PromotionGateway:
    def __init__(self, cfg: PromotionConfig | None = None):
        self.cfg = cfg or PromotionConfig.default()

    # ------------------------ shared helpers ------------------------

    @staticmethod
    def _softmax_masked(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Softmax over masked entries only. z: [1,V], mask: [V] in {0,1}.
        Returns probs [1,V] (zeros for masked-out positions).
        """
        z = z.clone()
        mask = (mask > 0.5).float()
        if mask.sum() <= 0:
            # avoid nan: uniform over full vocab (shouldn’t happen if caller guards)
            p = torch.full_like(z, 1.0 / max(z.numel(), 1.0))
            return p
        # set masked logits to -inf
        z_masked = z.clone()
        z_masked[0, mask < 0.5] = float("-inf")
        p = torch.softmax(z_masked, dim=-1)
        p = torch.where(mask.unsqueeze(0) > 0.5, p, torch.zeros_like(p))
        return p

    @staticmethod
    def _relative_target(base_p: float, tau: float) -> float:
        """
        Relative target share: push a fraction tau of remaining headroom (1-base_p).
        Ensures small, controlled nudges even when base_p is already large.
        """
        base_p = float(max(0.0, min(1.0, base_p)))
        tau = float(max(0.0, min(1.0, tau)))
        return float(max(0.0, min(1.0, base_p + tau * (1.0 - base_p))))

    @staticmethod
    def _delta_for_target_prob(
        z: torch.Tensor,
        legal_idx: Sequence[int],
        chosen_idx: int,
        target: float,
        max_boost: float,
    ) -> float:
        """
        Closed-form delta so that softmax(prob(chosen_idx | legal)) = target.
        Softmax over legal set only (illegal are excluded from Z).
        Let a = exp(z_i), S = sum_{j!=i} exp(z_j), then:
            target = a e^d / (a e^d + S)  =>  d = logit(target) + log(S/a)
        Use max-trick (subtract m) for stability, clamp to [-max_boost, +max_boost].
        """
        if not legal_idx or chosen_idx not in legal_idx:
            return 0.0

        # stability: subtract max over LEGAL logits
        with torch.no_grad():
            z_leg = z[0, legal_idx].detach()
            m = torch.max(z_leg)
            e = torch.exp(z[0] - m)  # [V], unmasked exponentials

            a = float(e[chosen_idx])
            S = float(e[legal_idx].sum().item() - a)

            # guard: if target <= base or numerically degenerate -> tiny/zero delta
            target = float(max(1e-6, min(1.0 - 1e-6, target)))
            if a <= 0.0:
                return 0.0

            # δ = logit(t) + log(S/a)
            logit_t = math.log(target / (1.0 - target))
            if S <= 0.0:
                # everyone else ~0 => already ~1.0 prob
                delta = 0.0
            else:
                delta = logit_t + math.log(S / a)

            if not math.isfinite(delta):
                return 0.0

            delta = max(-float(max_boost), min(float(max_boost), float(delta)))
            return float(delta)

    @staticmethod
    def _out(z: torch.Tensor, dbg: Dict[str, Any]) -> Dict[str, Any]:
        """Standardized postflop output shape."""
        return {"logits": z, "debug": dbg}

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
        # EV (scaled & capped)
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

    def promote_postflop(
            self, *, tokens, base_logits, hero_mask, side, actor,
            p_win, ex_probs, evs, size_frac=None, bet_menu_pcts=None, ctx=None, spr=None, allow_allin=False
    ) -> Dict[str, Any]:
        z = base_logits
        V = len(tokens)
        mask = (hero_mask > 0.5).float()
        legal_idx = [i for i in range(V) if mask[i] > 0.5]
        dbg = {"applied": False, "why": None, "ctx": ctx, "actor": actor, "spr": spr, "size_frac": size_frac,
               "mode": "override"}

        if not legal_idx:
            dbg["why"] = "no_legal";
            return self._out(z, dbg)

        up = [a.upper() for a in tokens]
        p_base = self._softmax_masked(z, mask)
        base_idx = int(torch.argmax(p_base[0]));
        base_tok = up[base_idx]

        passive_tok = "CALL" if side == "facing" else "CHECK"
        aggr_idx = _aggressive_indices(tokens, side=side, allow_allin=allow_allin)
        if side == "root" and bet_menu_pcts:
            bet_allow = set(f"BET_{int(x)}" for x in bet_menu_pcts)
            aggr_idx = [i for i in aggr_idx if tokens[i].upper() in bet_allow]
        aggr_idx = [i for i in aggr_idx if i in legal_idx]
        if not aggr_idx:
            dbg["why"] = "no_aggressive_in_menu";
            return self._out(z, dbg)

        # EV gap & best aggressive by EV
        if "FOLD" in evs:
            evs["FOLD"] = 0.0
        passive_ev = float(evs.get(passive_tok, evs.get(passive_tok.capitalize(), 0.0)))
        best_i, best_ev = None, None
        for i in aggr_idx:
            v = float(evs.get(tokens[i], evs.get(tokens[i].upper(), float("-inf"))))
            if (best_ev is None) or (v > best_ev):
                best_ev, best_i = v, i
        if best_i is None or best_ev is None or not math.isfinite(best_ev):
            dbg["why"] = "no_aggressive_ev";
            return self._out(z, dbg)

        ev_gap = max(0.0, best_ev - passive_ev)
        strong_by_eq = (p_win is not None) and (float(p_win) >= self.cfg.strong_eq_floor)
        strong_by_ev = (ev_gap >= self.cfg.strong_ev_margin_bb)
        ex_bias = self._exploit_bias(ex_probs, want_aggro=True)
        expl_ok = (ex_probs is not None) and (ex_bias > self.cfg.expl_gate)

        if side == "facing" and self.cfg.respect_fold_when_facing and base_tok == "FOLD" and not (
                strong_by_eq or strong_by_ev):
            dbg.update({"why": "respect_fold_baseline", "baseline": base_tok});
            return self._out(z, dbg)

        # ---------- OVERRIDE GATE ----------
        # Simple, deterministic ladder — no amplification, just set a fixed share when strong.
        share = 0.0
        if strong_by_ev or strong_by_eq or expl_ok:
            # EV ladder first
            if ev_gap >= 4.0:
                share = 0.40
            elif ev_gap >= 3.0:
                share = 0.33
            elif ev_gap >= 2.0:
                share = 0.25
            elif ev_gap >= 1.0:
                share = 0.18

            # Equity floor can lift small EV gaps
            if p_win is not None:
                if p_win >= 0.85:
                    share = max(share, 0.40)
                elif p_win >= 0.75:
                    share = max(share, 0.30)
                elif p_win >= 0.65:
                    share = max(share, 0.22)

            # Exploit bias adds a gentle bump (capped)
            if expl_ok: share = min(max(share, 0.22) + 0.05, 0.50)

        if share <= 0.0:
            dbg["why"] = "gate_not_met"
            return self._out(z, dbg)

        # ---------- ABSOLUTE OVERRIDE ----------
        base_p = float(p_base[0, best_i])
        target_abs = float(min(max(share, base_p), 1.0 - 1e-6))  # never reduce
        p_new = torch.zeros_like(p_base[0])
        legal_rest = [j for j in legal_idx if j != best_i]

        if not legal_rest:
            p_new[best_i] = 1.0
        else:
            rest_mass = 1.0 - target_abs
            w = float(p_base[0, legal_rest].sum().item())
            if w <= 0.0:
                share_rest = rest_mass / float(len(legal_rest))
                for j in legal_rest: p_new[j] = share_rest
            else:
                for j in legal_rest: p_new[j] = rest_mass * float(p_base[0, j]) / w
            p_new[best_i] = target_abs

        # Strong-hand no-fold cap when facing
        if side == "facing" and "FOLD" in up and (strong_by_eq or strong_by_ev):
            fold_idx = up.index("FOLD")
            if fold_idx in legal_idx and float(p_new[fold_idx]) > float(self.cfg.fold_cap_when_strong):
                excess = float(p_new[fold_idx]) - float(self.cfg.fold_cap_when_strong)
                p_new[fold_idx] = float(self.cfg.fold_cap_when_strong)
                others = [j for j in legal_idx if j != fold_idx]
                S = float(p_new[others].sum().item())
                if S > 0.0:
                    scale = (S + excess) / S
                    for j in others: p_new[j] = float(p_new[j]) * scale
                elif others:
                    add = excess / float(len(others))
                    for j in others: p_new[j] = float(p_new[j]) + add

        # Normalize over legal and convert to logits
        total_legal = float(p_new[legal_idx].sum().item())
        if total_legal > 0:
            for j in legal_idx: p_new[j] = float(p_new[j]) / total_legal

        z_out = z.clone()
        eps = 1e-9
        for j in legal_idx: z_out[0, j] = math.log(max(float(p_new[j]), eps))

        dbg.update({
            "applied": True,
            "override_share": round(share, 3),
            "target_abs": round(target_abs, 6),
            "best_bucket": tokens[best_i],
            "ev_gap_bb": round(ev_gap, 3),
            "baseline": base_tok,
            "base_p": round(base_p, 6),
            "p_after_best": round(float(p_new[best_i]), 6),
            "strong_by_ev": strong_by_ev, "strong_by_eq": strong_by_eq, "expl_ok": expl_ok,
        })
        return self._out(z_out, dbg)

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
        OVERRIDE (absolute share). No temps, no amplification.
        """
        import math
        import numpy as np
        import torch

        dbg: Dict[str, Any] = {"applied": False, "why": None, "mode": "override_preflop"}
        if not tokens:
            dbg["why"] = "no_tokens"
            return base_logits, dbg

        up = [t.upper() for t in tokens]
        z = torch.tensor(base_logits, dtype=torch.float32).view(1, -1)
        mask = torch.ones(len(tokens), dtype=torch.float32)
        legal_idx = [i for i in range(len(tokens)) if mask[i] > 0.5]

        passive_tok = "CALL" if facing_bet else ("CHECK" if free_check else "FOLD")
        if facing_bet:
            aggr_idx = [i for i, t in enumerate(up) if t.startswith("RAISE_") or (t == "ALLIN" and allow_allin)]
        else:
            aggr_idx = [i for i, t in enumerate(up) if t.startswith("OPEN_")]

        if not aggr_idx:
            dbg["why"] = "no_aggressive_in_menu"
            return base_logits, dbg

        passive_ev = float(evs.get(passive_tok, evs.get(passive_tok.capitalize(), 0.0)))
        best_i, best_ev = None, None
        for i in aggr_idx:
            v = float(evs.get(tokens[i], evs.get(up[i], float("-inf"))))
            if (best_ev is None) or (v > best_ev):
                best_ev, best_i = v, i
        if best_i is None or best_ev is None or not math.isfinite(best_ev):
            dbg["why"] = "no_aggressive_ev"
            return base_logits, dbg

        ev_gap = max(0.0, best_ev - passive_ev)

        # EV ladder + equity lift -> absolute target share
        share = 0.0
        if ev_gap >= 4.0:
            share = 0.40
        elif ev_gap >= 3.0:
            share = 0.33
        elif ev_gap >= 2.0:
            share = 0.25
        elif ev_gap >= 1.0:
            share = 0.18
        if p_win is not None:
            if p_win >= 0.85:
                share = max(share, 0.40)
            elif p_win >= 0.75:
                share = max(share, 0.30)
            elif p_win >= 0.65:
                share = max(share, 0.22)

        if share <= 0.0:
            dbg["why"] = "gate_not_met"
            dbg.update({"ev_gap_bb": round(ev_gap, 3), "p_win": float(p_win) if p_win is not None else None})
            return base_logits, dbg

        p_base = self._softmax_masked(z, mask)  # [1,V]
        base_p = float(p_base[0, best_i])

        # Never reduce existing share; if no increase, it's a no-op.
        target_abs = float(min(max(share, base_p), 1.0 - 1e-6))
        if target_abs <= base_p + 1e-6:
            dbg.update({
                "applied": False,
                "why": "already_at_or_above_target",
                "base_p": round(base_p, 6),
                "target_abs": round(target_abs, 6),
                "best_bucket": tokens[best_i],
                "ev_gap_bb": round(ev_gap, 3),
                "passive": passive_tok,
            })
            return base_logits, dbg

        # Build new prob vector over legal set, with absolute mass on best_i.
        p_new = torch.zeros_like(p_base[0])
        legal_rest = [j for j in legal_idx if j != best_i]
        if not legal_rest:
            p_new[best_i] = 1.0
        else:
            rest_mass = 1.0 - target_abs
            w = float(p_base[0, legal_rest].sum().item())
            if w <= 0.0:
                share_rest = rest_mass / float(len(legal_rest))
                for j in legal_rest: p_new[j] = share_rest
            else:
                for j in legal_rest: p_new[j] = rest_mass * float(p_base[0, j]) / w
            p_new[best_i] = target_abs

        # Normalize (guard)
        s = float(p_new[legal_idx].sum().item())
        if s > 0.0:
            for j in legal_idx: p_new[j] = float(p_new[j]) / s

        # Return LOG-PROBS to avoid double-softmax downstream.
        z_out = z.clone()
        eps = 1e-9
        for j in legal_idx:
            z_out[0, j] = math.log(max(float(p_new[j]), eps))

        dbg.update({
            "applied": True,
            "override_share": round(share, 3),
            "target_abs": round(target_abs, 6),
            "best_bucket": tokens[best_i],
            "ev_gap_bb": round(ev_gap, 3),
            "passive": passive_tok,
            "base_p": round(base_p, 6),
            "p_after_best": round(float(p_new[best_i]), 6),
        })
        return z_out[0].detach().cpu().numpy(), dbg