# policy/tuner.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch, math

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

    # guards
    raise_block_if_allin_legal: bool = True
    raise_when_faced_min_size: float = 0.30
    raise_when_faced_max_size: float = 1.00
    raise_max_logit_boost: float = 8.0

    # safeguard: avoid tuning dead actions
    min_gto_engage_prob: float = 0.01


class PostflopTuner:
    def __init__(self, knobs: TunerKnobs):
        self.k = knobs

    @staticmethod
    def _ruler_group_delta(z, legal_idx, group_idx, tau, max_boost):
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

    def _tau_from_signals(self, base, p_win, ex_probs):
        tau = base
        if p_win is not None:
            gap = max(0.0, p_win - self.k.eq_tau_gate)
            tau += min(self.k.eq_tau_scale * gap, self.k.eq_tau_max)
        if ex_probs is not None:
            pf, pc, pr = ex_probs
            fold_adv = max(0.0, pf - max(pc, pr) - self.k.expl_fold_gate)
            aggr_adv = max(0.0, pr - max(pc, pf) - self.k.expl_aggr_gate)
            tau += min(self.k.expl_fold_scale * fold_adv, self.k.expl_fold_max)
            tau += min(self.k.expl_aggr_scale * aggr_adv, self.k.expl_aggr_max)
        return float(min(max(tau, self.k.tau_floor), self.k.tau_ceil))

    def _engage_threshold_met(self, z, legal_idx, group_idx):
        if not group_idx:
            return False
        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        G = torch.logsumexp(l[group_idx], dim=0)
        g_share = math.exp(float(G - L))
        return g_share > self.k.min_gto_engage_prob

    def apply_facing_raise(self, z, actions, hero_mask, p_win, ex_probs, size_frac):
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
                self.k.raise_when_faced_min_size <= size_frac <= self.k.raise_when_faced_max_size):
            return {"applied": False, "reason": "size_gate_fail", "size_frac": float(size_frac)}

        # Compute logit share
        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        G = torch.logsumexp(l[group_idx], dim=0)
        P = torch.logsumexp(l[passive_idx], dim=0) if passive_idx else G
        g_share = math.exp(float(G - L))
        p_share = math.exp(float(P - L))

        tau_target = self._tau_from_signals(p_share, p_win, ex_probs)
        tau_move = g_share + self.k.step * (tau_target - g_share)

        delta = self._ruler_group_delta(z, legal_idx, group_idx, tau_move, self.k.raise_max_logit_boost)
        if delta > 0.0:
            z[0, group_idx] += delta

        # Optional hard promotion override
        gto_idx = int(torch.argmax(z[0][legal_idx]))
        gto_action = actions[legal_idx[gto_idx]]
        top_raise_idx = max(group_idx, key=lambda i: z[0, i].item())
        promo_trigger = False

        if gto_action == "CALL" and p_win is not None and ex_probs is not None:
            # Heuristic promotion condition
            if p_win > self.k.raise_bias_eq_boost_gate and ex_probs[2] > 0.25:  # strong win chance + high raise freq
                promo_trigger = True
                z[0] = z[0] - 100.0
                z[0, top_raise_idx] = 10.0
                dbg["forced_promotion"] = {
                    "from": "CALL",
                    "to": actions[top_raise_idx],
                    "reason": "high_equity_and_exploit_raise_prob",
                    "p_win": p_win,
                    "expl_raise": ex_probs[2]
                }

        g_after = math.exp(float(torch.logsumexp(z[0][group_idx], dim=0) - torch.logsumexp(z[0][legal_idx], dim=0)))
        dbg.update({
            "applied": delta > 0.0 or promo_trigger,
            "promoted_from": "CALL" if passive_idx else None,
            "g_share_before": g_share,
            "tau_target": tau_target,
            "tau_move": tau_move,
            "delta": delta,
            "g_share_after": g_after,
            "p_win": p_win,
            "ex_probs": ex_probs,
            "size_frac": size_frac,
            "n_group": len(group_idx),
            "allin_legal": allin_legal,
        })
        return dbg

    def apply_root_bet(self, z, actions, hero_mask, p_win, ex_probs):
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
        G = torch.logsumexp(l[group_idx], dim=0)
        P = torch.logsumexp(l[passive_idx], dim=0) if passive_idx else G
        g_share = math.exp(float(G - L))
        p_share = math.exp(float(P - L))

        tau_target = self._tau_from_signals(p_share, p_win, ex_probs)
        tau_move = g_share + self.k.step * (tau_target - g_share)

        delta = self._ruler_group_delta(z, legal_idx, group_idx, tau_move, self.k.raise_max_logit_boost)
        if delta > 0.0:
            z[0, group_idx] += delta

        # Optional hard promotion override
        gto_idx = int(torch.argmax(z[0][legal_idx]))
        gto_action = actions[legal_idx[gto_idx]]
        top_bet_idx = max(group_idx, key=lambda i: z[0, i].item())
        promo_trigger = False

        if gto_action == "CHECK" and p_win is not None and ex_probs is not None:
            if p_win > self.k.bet_tau_equity_gate and ex_probs[2] > 0.25:
                promo_trigger = True
                z[0] = z[0] - 100.0
                z[0, top_bet_idx] = 10.0
                dbg["forced_promotion"] = {
                    "from": "CHECK",
                    "to": actions[top_bet_idx],
                    "reason": "high_equity_and_exploit_raise_prob",
                    "p_win": p_win,
                    "expl_raise": ex_probs[2]
                }

        g_after = math.exp(float(torch.logsumexp(z[0][group_idx], dim=0) - torch.logsumexp(z[0][legal_idx], dim=0)))
        dbg.update({
            "applied": delta > 0.0 or promo_trigger,
            "promoted_from": "CHECK" if passive_idx else None,
            "g_share_before": g_share,
            "tau_target": tau_target,
            "tau_move": tau_move,
            "delta": delta,
            "g_share_after": g_after,
            "p_win": p_win,
            "ex_probs": ex_probs,
            "n_group": len(group_idx),
        })
        return dbg