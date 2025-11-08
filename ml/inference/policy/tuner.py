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
    # (optional) bet knobs could live here too

class PostflopTuner:
    def __init__(self, knobs: TunerKnobs):
        self.k = knobs

    @staticmethod
    def _ruler_group_delta(z: torch.Tensor, legal_idx: List[int], group_idx: List[int],
                           tau: float, max_boost: float) -> float:
        if not legal_idx or not group_idx or tau <= 0.0:
            return 0.0
        l = z[0]
        L = torch.logsumexp(l[legal_idx], dim=0)
        G = torch.logsumexp(l[group_idx], dim=0)
        g_share = math.exp(float(G - L))
        if tau <= g_share + 1e-9:
            return 0.0
        Sg = math.exp(float(G)); So = max(math.exp(float(L)) - Sg, 1e-12)
        ratio = max(tau / max(1.0 - tau, 1e-6), 1e-6)
        d = math.log(ratio * (So / max(Sg,1e-12)))
        return float(max(0.0, min(d, max_boost)))

    def apply_facing_raise(self,
                           z: torch.Tensor, actions: List[str], hero_mask: torch.Tensor,
                           p_win: Optional[float], ex_probs: Optional[List[float]],
                           size_frac: Optional[float]) -> Dict[str, Any]:
        dbg: Dict[str, Any] = {}
        if not self.k.enable: return {"applied": False, "reason": "tuner_disabled"}

        V = len(actions)
        legal_idx = [i for i in range(V) if hero_mask[i] > 0.5]
        group_idx = [i for i in legal_idx if actions[i].startswith("RAISE_")]
        allin_legal = any(actions[i] == "ALLIN" for i in legal_idx)
        if not group_idx:
            return {"applied": False, "reason": "no_raise_in_menu_mask"}
        if self.k.raise_block_if_allin_legal and allin_legal:
            return {"applied": False, "reason": "allin_block"}

        # facing-size gate
        if size_frac is not None:
            if not (self.k.raise_when_faced_min_size <= size_frac <= self.k.raise_when_faced_max_size):
                return {"applied": False, "reason": "size_gate_fail", "size_frac": float(size_frac)}

        # current group share
        l = z[0]; L = torch.logsumexp(l[legal_idx], dim=0); G = torch.logsumexp(l[group_idx], dim=0)
        g_share = math.exp(float(G - L))

        tau_target = g_share
        # equity nudge
        if p_win is not None:
            gap = max(0.0, p_win - self.k.eq_tau_gate)
            tau_target += min(self.k.eq_tau_scale * gap, self.k.eq_tau_max)
        # exploit nudges
        if ex_probs is not None:
            pf, pc, pr = ex_probs
            fold_adv = max(0.0, pf - max(pc, pr) - self.k.expl_fold_gate)
            aggr_adv = max(0.0, pr - max(pc, pf) - self.k.expl_aggr_gate)
            tau_target += min(self.k.expl_fold_scale * fold_adv, self.k.expl_fold_max)
            tau_target += min(self.k.expl_aggr_scale * aggr_adv, self.k.expl_aggr_max)

        tau_target = float(min(max(tau_target, self.k.tau_floor), self.k.tau_ceil))
        tau_move = g_share + self.k.step * (tau_target - g_share)

        delta = self._ruler_group_delta(z, legal_idx, group_idx, tau_move, self.k.raise_max_logit_boost)
        if delta > 0.0:
            z[0, group_idx] += delta

        # share after
        l2 = z[0]; L2 = torch.logsumexp(l2[legal_idx], dim=0); G2 = torch.logsumexp(l2[group_idx], dim=0)
        g_after = math.exp(float(G2 - L2))
        dbg.update({
            "applied": bool(delta > 0.0),
            "g_share_before": g_share, "tau_target": tau_target, "tau_move": tau_move,
            "delta": float(delta), "g_share_after": g_after,
            "p_win": p_win, "ex_probs": ex_probs, "size_frac": size_frac,
            "n_group": int(len(group_idx)), "allin_legal": bool(allin_legal),
        })
        return dbg