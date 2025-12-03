from dataclasses import dataclass
import torch, math

from ml.inference.aggro_bucket_chooser import AggroBucketChooser
from ml.inference.strategy_blender import StrategyBlender


@dataclass
class TunerKnobs:
    enable: bool = True
    debug: bool = True
    step: float = 0.6
    tau_floor: float = 0.0
    tau_ceil: float = 0.6

    # equity → tau
    eq_tau_gate: float = 0.56
    eq_tau_scale: float = 0.8
    eq_tau_max: float = 0.18

    # exploit → tau
    expl_fold_gate: float = 0.10
    expl_fold_scale: float = 0.5
    expl_fold_max: float = 0.20

    expl_aggr_gate: float = 0.10
    expl_aggr_scale: float = 0.25
    expl_aggr_max: float = 0.08

    # EV → tau (NEW)
    ev_tau_gate: float = 0.0
    ev_tau_scale: float = 0.5
    ev_tau_max: float = 0.15

    # guards
    raise_block_if_allin_legal: bool = True
    raise_when_faced_min_size: float = 0.30
    raise_when_faced_max_size: float = 1.00
    raise_max_logit_boost: float = 8.0

    min_gto_engage_prob: float = 0.01
    raise_bias_eq_boost_gate: float = 0.60
    bet_tau_equity_gate: float = 0.60
    ctx_exploit_bonus: float = 0.2
    min_temp: float = 0.2
    max_temp: float = 1.5


class PostflopTuner:
    def __init__(self, knobs: TunerKnobs):
        self.k = knobs

    def _ruler_group_delta(self, z, legal_idx, group_idx, tau, max_boost):
        if not legal_idx or not group_idx or tau <= 0.0:
            return 0.0
        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        G = torch.logsumexp(l[group_idx], dim=0)
        g_share = math.exp(float(G - L))
        if tau <= g_share + 1e-9:
            return 0.0
        Sg = math.exp(float(G))
        So = max(math.exp(float(L)) - Sg, 1e-12)
        ratio = max(tau / max(1.0 - tau, 1e-6), 1e-6)
        d = math.log(ratio * (So / max(Sg, 1e-12)))
        return float(max(0.0, min(d, max_boost)))

    def _tau_from_signals(self, base, p_win, ex_probs, ev=None, action_ctx=None):
        tau = base

        # 1. Equity + fold equity: effective equity
        if p_win is not None and action_ctx is not None:
            fold_score = action_ctx.fold_bias_score()
            effective_equity = p_win + fold_score * (1 - p_win)
            gap = max(0.0, effective_equity - self.k.eq_tau_gate)
            tau += min(self.k.eq_tau_scale * gap, self.k.eq_tau_max)
        elif p_win is not None:
            gap = max(0.0, p_win - self.k.eq_tau_gate)
            tau += min(self.k.eq_tau_scale * gap, self.k.eq_tau_max)

        # 2. Exploit signal boost (fold or raise tendencies)
        if ex_probs is not None:
            pf, pc, pr = ex_probs

            # Fold bias boost
            fold_adv = max(0.0, pf - max(pc, pr) - self.k.expl_fold_gate)
            tau += min(self.k.expl_fold_scale * fold_adv, self.k.expl_fold_max)

            # Aggro bias boost
            aggr_gap = pr - max(pc, pf)
            if aggr_gap > self.k.expl_aggr_gate:
                tau += min(self.k.expl_aggr_scale * aggr_gap, self.k.expl_aggr_max)

        # 3. EV contribution
        if ev is not None:
            tau += min(self.k.ev_tau_scale * ev, self.k.ev_tau_max)

        # 4. SPR-aware adjustment (optional soft scaling)
        if hasattr(self, "pot_bb") and hasattr(self, "eff_stack_bb") and self.pot_bb and self.eff_stack_bb:
            spr = self.eff_stack_bb / self.pot_bb
            if spr > 6:
                tau *= 0.7
            elif spr < 2:
                tau *= 1.1

        # 5. Contextual scaling
        if action_ctx is not None:
            tau += self.k.ctx_exploit_bonus * action_ctx.fold_bias_score()

        return float(min(max(tau, self.k.tau_floor), self.k.tau_ceil))

    def _ruler_single_delta(self, z, legal_idx, target_idx, tau_target, max_boost):
        """
        Computes how much to boost a single logit to reach the desired tau_target share.
        """
        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        T = l[target_idx]

        # New logit after delta: T + delta
        # New probability: exp(T + delta - logsumexp)
        # Solve: exp(T + delta - L_new) = tau_target
        # But since L_new changes with delta, we approximate:
        #   exp(T + delta - L) ≈ tau_target
        # → delta ≈ log(tau_target) + L - T

        try:
            raw_delta = math.log(tau_target) + float(L - T)
        except (ValueError, OverflowError):
            raw_delta = 0.0

        delta = max(0.0, min(raw_delta, max_boost))
        return float(delta)

    def apply_facing_raise(self, z, actions, hero_mask, p_win, ex_probs, size_frac, evs=None, action_ctx=None):
        dbg = {}
        if not self.k.enable:
            return {"applied": False, "reason": "tuner_disabled"}

        V = len(actions)
        legal_idx = [i for i in range(V) if hero_mask[i] > 0.5]
        group_idx = [i for i in legal_idx if actions[i].startswith("RAISE_")]
        passive_idx = [i for i in legal_idx if actions[i] == "CALL"]
        allin_legal = any(actions[i] == "ALLIN" for i in legal_idx)

        if not group_idx:
            return {"applied": False, "reason": "no_raise_in_menu_mask"}
        if self.k.raise_block_if_allin_legal and allin_legal:
            return {"applied": False, "reason": "allin_block"}
        if size_frac is not None and not (
                self.k.raise_when_faced_min_size <= size_frac <= self.k.raise_when_faced_max_size
        ):
            return {"applied": False, "reason": "size_gate_fail", "size_frac": float(size_frac)}

        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        if passive_idx:
            C = torch.logsumexp(l[passive_idx], dim=0)
            p_share = math.exp(float(C - L))
        else:
            p_share = 0.0

        # ----- EV gating -----
        if evs:
            call_ev = evs.get("CALL", None)
            raise_evs = {a: ev for a, ev in evs.items() if a.startswith("RAISE_")}
            if raise_evs and call_ev is not None:
                best_raise_ev = max(raise_evs.values())
                if best_raise_ev < call_ev:
                    dbg.update({
                        "promotion_mode": "ev_block",
                        "reason": "all_raises_worse_than_call",
                        "call_ev": round(call_ev, 3),
                        "best_raise_ev": round(best_raise_ev, 3),
                        "evs": raise_evs,
                    })
                    return dbg

        tau_target = self._tau_from_signals(
            base=p_share,
            p_win=p_win,
            ex_probs=ex_probs,
            ev=max(evs.values()) if evs else None,
            action_ctx=action_ctx,
        )

        chosen_idx = None
        if evs:
            chooser = AggroBucketChooser(actions, z, hero_mask, evs)
            chosen_idx = chooser.choose_best()
            dbg["chooser_debug"] = chooser.debug_info()

        applied = False
        if chosen_idx is not None:
            delta = self._ruler_single_delta(z, legal_idx, chosen_idx, tau_target, self.k.raise_max_logit_boost)
            if delta > 0.0:
                z[0, chosen_idx] += delta
                applied = True
            dbg.update({
                "promotion_mode": "bucket_target",
                "chosen_action": actions[chosen_idx],
                "delta": delta,
            })
        else:
            G = torch.logsumexp(l[group_idx], dim=0)
            g_share = math.exp(float(G - L))
            tau_move = g_share + self.k.step * (tau_target - g_share)
            delta = self._ruler_group_delta(z, legal_idx, group_idx, tau_move, self.k.raise_max_logit_boost)
            if delta > 0.0:
                z[0, group_idx] += delta
                applied = True
            dbg.update({
                "promotion_mode": "group_soft",
                "delta": delta,
            })

        # ⚡ Inline StrategyBlender usage
        try:
            blender = StrategyBlender(
                logits=z[0],
                actions=actions,
                hero_mask=hero_mask,
                temperature=getattr(self, "temperature", 1.0),
                min_temp=self.k.min_temp,
                max_temp=self.k.max_temp,
            )
            self.temperature = blender.adjust_temperature(
                eff_stack_bb=getattr(self, "eff_stack_bb", None),
                pot_bb=getattr(self, "pot_bb", None),
                delta=max(evs.values()) - min(evs.values()) if evs else None,
                dbg_out=dbg,
            )
            if self.temperature != 1.0:
                z[0, legal_idx] /= self.temperature
                dbg["logit_temperature_scaled"] = round(float(self.temperature), 3)
        except Exception as e:
            dbg["strategy_blender_error"] = str(e)

        g_after = math.exp(float(torch.logsumexp(z[0][group_idx], dim=0) - torch.logsumexp(z[0][legal_idx], dim=0)))
        dbg.update({
            "applied": applied,
            "promoted_from": "CALL" if passive_idx else None,
            "tau_target": tau_target,
            "g_share_after": g_after,
            "p_win": p_win,
            "ex_probs": ex_probs,
            "size_frac": size_frac,
            "ev": evs,
            "action_context": action_ctx.summary() if action_ctx else None,
            "debug_name": "apply_facing_raise",
        })

        return dbg

    def apply_root_bet(self, z, actions, hero_mask, p_win, ex_probs, evs=None, action_ctx=None):
        dbg = {}
        if not self.k.enable:
            return {"applied": False, "reason": "tuner_disabled"}

        V = len(actions)
        legal_idx = [i for i in range(V) if hero_mask[i] > 0.5]
        group_idx = [i for i in legal_idx if actions[i].startswith("BET_")]
        passive_idx = [i for i in legal_idx if actions[i] == "CHECK"]

        if not group_idx:
            return {"applied": False, "reason": "no_bet_in_menu_mask"}

        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        if passive_idx:
            C = torch.logsumexp(l[passive_idx], dim=0)
            p_share = math.exp(float(C - L))
        else:
            p_share = 0.0

        # ----- EV gating -----
        if evs:
            check_ev = evs.get("CHECK", None)
            bet_evs = {a: ev for a, ev in evs.items() if a.startswith("BET_")}
            if bet_evs and check_ev is not None:
                best_bet_ev = max(bet_evs.values())
                if best_bet_ev < check_ev:
                    dbg.update({
                        "promotion_mode": "ev_block",
                        "reason": "all_bets_worse_than_check",
                        "check_ev": round(check_ev, 3),
                        "best_bet_ev": round(best_bet_ev, 3),
                        "evs": bet_evs,
                    })
                    return dbg

        tau_target = self._tau_from_signals(
            base=p_share,
            p_win=p_win,
            ex_probs=ex_probs,
            ev=max(evs.values()) if evs else None,
            action_ctx=action_ctx,
        )

        chosen_idx = None
        if evs:
            chooser = AggroBucketChooser(actions, z, hero_mask, evs)
            chosen_idx = chooser.choose_best()
            dbg["chooser_debug"] = chooser.debug_info()

        applied = False
        if chosen_idx is not None:
            delta = self._ruler_single_delta(z, legal_idx, chosen_idx, tau_target, self.k.raise_max_logit_boost)
            if delta > 0.0:
                z[0, chosen_idx] += delta
                applied = True
            dbg.update({
                "promotion_mode": "bucket_target",
                "chosen_action": actions[chosen_idx],
                "delta": delta,
            })
        else:
            G = torch.logsumexp(l[group_idx], dim=0)
            g_share = math.exp(float(G - L))
            tau_move = g_share + self.k.step * (tau_target - g_share)
            delta = self._ruler_group_delta(z, legal_idx, group_idx, tau_move, self.k.raise_max_logit_boost)
            if delta > 0.0:
                z[0, group_idx] += delta
                applied = True
            dbg.update({
                "promotion_mode": "group_soft",
                "delta": delta,
            })

        # ⚡ Inline StrategyBlender usage
        try:
            blender = StrategyBlender(
                logits=z[0],
                actions=actions,
                hero_mask=hero_mask,
                temperature=getattr(self, "temperature", 1.0),
                min_temp=self.k.min_temp,
                max_temp=self.k.max_temp,
            )
            self.temperature = blender.adjust_temperature(
                eff_stack_bb=getattr(self, "eff_stack_bb", None),
                pot_bb=getattr(self, "pot_bb", None),
                delta=max(evs.values()) - min(evs.values()) if evs else None,
                dbg_out=dbg,
            )
            if self.temperature != 1.0:
                z[0, legal_idx] /= self.temperature
                dbg["logit_temperature_scaled"] = round(float(self.temperature), 3)
        except Exception as e:
            dbg["strategy_blender_error"] = str(e)

        g_after = math.exp(float(torch.logsumexp(z[0][group_idx], dim=0) - torch.logsumexp(z[0][legal_idx], dim=0)))
        dbg.update({
            "applied": applied,
            "promoted_from": "CHECK" if passive_idx else None,
            "tau_target": tau_target,
            "g_share_after": g_after,
            "p_win": p_win,
            "ex_probs": ex_probs,
            "ev": evs,
            "action_context": action_ctx.summary() if action_ctx else None,
            "debug_name": "apply_root_bet",
        })

        return dbg