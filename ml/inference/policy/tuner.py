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
    tau_ceil: float = 0.60

    # Weights for how much each signal steers tau (must sum <= 1.0; rest is baseline)
    eq_weight: float = 0.35
    expl_weight: float = 0.45
    ev_weight: float = 0.20

    # equity → component
    eq_tau_gate: float = 0.56
    eq_tau_scale: float = 0.8
    eq_tau_max: float = 0.18

    # exploit → component (foldy villain or aggro villain)
    expl_fold_gate: float = 0.10
    expl_fold_scale: float = 0.50
    expl_fold_max: float = 0.20

    expl_aggr_gate: float = 0.10
    expl_aggr_scale: float = 0.25
    expl_aggr_max: float = 0.08

    # EV → component
    ev_tau_gate: float = 0.00
    ev_tau_scale: float = 0.50
    ev_tau_max: float = 0.15

    # guards and limits
    raise_block_if_allin_legal: bool = True
    raise_when_faced_min_size: float = 0.30
    raise_when_faced_max_size: float = 1.00
    raise_max_logit_boost: float = 8.0

    min_gto_engage_prob: float = 0.01
    raise_bias_eq_boost_gate: float = 0.60
    bet_tau_equity_gate: float = 0.60

    # strategy blending temp limits
    min_temp: float = 0.20
    max_temp: float = 1.50


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

    def _ruler_single_delta(self, z, legal_idx, target_idx, tau_target, max_boost):
        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        T = l[target_idx]
        try:
            raw_delta = math.log(max(tau_target, 1e-6)) + float(L - T)
        except (ValueError, OverflowError):
            raw_delta = 0.0
        return float(max(0.0, min(raw_delta, max_boost)))

    # ---------- new: weighted tau composition ----------
    def _tau_from_signals(self, base_share, p_win, ex_probs, ev=None, action_ctx=None):
        """
        base_share: current passive share (CALL at facing, CHECK at root)
        Produces tau_target for aggressive group as:
          tau = clamp( base*(1-w) + eq_w*eq_part + expl_w*expl_part + ev_w*ev_part )
        """
        # Components live in [0, tau_ceil]; we sum them with weights.
        w_eq = float(self.k.eq_weight)
        w_ex = float(self.k.expl_weight)
        w_ev = float(self.k.ev_weight)
        w_sum = max(0.0, min(w_eq + w_ex + w_ev, 1.0))

        # --- equity part ---
        eq_part = 0.0
        if p_win is not None:
            # fold-bias context can inflate effective equity
            if action_ctx is not None and hasattr(action_ctx, "fold_bias_score"):
                eff_equity = float(p_win) + float(action_ctx.fold_bias_score()) * (1.0 - float(p_win))
            else:
                eff_equity = float(p_win)
            eq_gap = max(0.0, eff_equity - self.k.eq_tau_gate)
            eq_part = min(self.k.eq_tau_scale * eq_gap, self.k.eq_tau_max)

        # --- exploit part ---
        ex_part = 0.0
        if ex_probs is not None and len(ex_probs) == 3:
            pf, pc, pr = [float(x) for x in ex_probs]
            fold_adv = max(0.0, pf - max(pc, pr) - self.k.expl_fold_gate)
            ex_part += min(self.k.expl_fold_scale * fold_adv, self.k.expl_fold_max)
            aggr_gap = pr - max(pc, pf)
            if aggr_gap > self.k.expl_aggr_gate:
                ex_part += min(self.k.expl_aggr_scale * aggr_gap, self.k.expl_aggr_max)

        # --- EV part ---
        ev_part = 0.0
        if ev is not None:
            # Treat ev as a non-negative advantage scalar (you pass best raise EV or spread)
            ev_pos = max(0.0, float(ev) - self.k.ev_tau_gate)
            ev_part = min(self.k.ev_tau_scale * ev_pos, self.k.ev_tau_max)

        # Compose
        tau = (1.0 - w_sum) * float(base_share) + w_eq * eq_part + w_ex * ex_part + w_ev * ev_part

        # Optional SPR adjustment (gentle)
        try:
            spr = float(self.eff_stack_bb) / max(float(self.pot_bb), 1e-6)  # set by caller if you want
            if spr > 6:
                tau *= 0.85
            elif spr < 2.0:
                tau *= 1.10
        except Exception:
            pass

        return float(min(self.k.tau_ceil, max(self.k.tau_floor, tau)))

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
            base_share=p_share,
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

    # ml/inference/tuning/tuner.py

    def apply_root_bet(
            self,
            z,
            actions,
            hero_mask,
            p_win,
            ex_probs,
            evs=None,
            action_ctx=None,
            *,
            bet_menu_pcts: list[int] | None = None,  # <-- NEW
    ):
        dbg = {}
        if not self.k.enable:
            return {"applied": False, "reason": "tuner_disabled"}

        V = len(actions)
        legal_idx = [i for i in range(V) if hero_mask[i] > 0.5]

        # --- define the aggressive group strictly from router’s bet menu if provided ---
        bet_set = set(int(x) for x in bet_menu_pcts) if bet_menu_pcts else None
        group_idx = []
        for i in legal_idx:
            tok = actions[i]
            if not tok.startswith("BET_"):
                continue
            if bet_set is None:
                group_idx.append(i)
            else:
                try:
                    pct = int(tok.split("_", 1)[1])
                    if pct in bet_set:
                        group_idx.append(i)
                except Exception:
                    pass

        passive_idx = [i for i in legal_idx if actions[i] == "CHECK"]

        if not group_idx:
            return {
                "applied": False,
                "reason": "no_bet_in_menu_mask",
                "bet_menu_pcts": (list(bet_set) if bet_set else None),
            }

        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        if passive_idx:
            C = torch.logsumexp(l[passive_idx], dim=0)
            p_share = math.exp(float(C - L))  # current CHECK share
        else:
            p_share = 0.0

        # ----- EV gating (unchanged) -----
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
                        "bet_menu_pcts": (list(bet_set) if bet_set else None),
                    })
                    return dbg

        # ----- new weighted tau from signals (your updated implementation) -----
        tau_target = self._tau_from_signals(
            base_share=p_share,
            p_win=p_win,
            ex_probs=ex_probs,
            ev=(max(evs.values()) if evs else None),
            action_ctx=action_ctx,
        )

        # ----- chooser path if EVs present -----
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
            # group soft move
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

        # optional StrategyBlender (unchanged)
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
                delta=(max(evs.values()) - min(evs.values())) if evs else None,
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
            "bet_menu_pcts": (list(bet_set) if bet_set else None),
            "debug_name": "apply_root_bet",
        })
        return dbg