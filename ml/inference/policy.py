# policy_infer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math

from ml.features.boards import BoardClusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.exploit import ExploitNetInfer
from ml.inference.population import PopulationNetInfer
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import RangeNetPreflopInfer
from ml.policy.utils import summarize_169, hand_to_169, hand_to_id

ACTION_VOCAB = []  # your real vocab will be imported in runtime path


@dataclass
class PolicyInferDeps:
    # optional
    pop: PopulationNetInfer | None = None
    # required
    exploit: ExploitNetInfer | None = None
    equity: EquityNetInfer | None = None
    # rangenets
    range_pre: RangeNetPreflopInfer | None = None
    # postflop policy
    policy_post: PostflopPolicyInfer | None = None
    # utils
    clusterer: BoardClusterer | None = None
    params: Dict[str, Any] | None = None


class PolicyInfer:
    """
    Routes:
      - Preflop:  RangeNetPreflop + Equity + Exploit/Pop blend -> coarse {FOLD,CALL,RAISE} + EV
      - Postflop: PostflopPolicyInfer -> ACTION_VOCAB + EV
    Output: dict with actions, probs, evs, debug
    """

    # ---------- construction ----------
    def __init__(self, deps: PolicyInferDeps):
        if deps.exploit is None:
            raise ValueError("exploit infer is required")
        if deps.equity is None:
            raise ValueError("equity infer is required")
        if deps.range_pre is None:
            raise ValueError("range_pre (RangeNetPreflopInfer) is required")
        if deps.policy_post is None:
            raise ValueError("policy_post (PostflopPolicyInfer) is required")

        self.pop   = deps.pop
        self.expl  = deps.exploit
        self.eq    = deps.equity
        self.rng_pre  = deps.range_pre
        self.pol_post = deps.policy_post
        self.clusterer = deps.clusterer
        self.p = deps.params or {}

    # ---------- public ----------
    def predict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        street = int(req["street"])
        return self._predict_preflop(req) if street == 0 else self._predict_postflop(req)

    # ---------- preflop ----------
    def _predict_preflop(self, req: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Range (169)
        feats = {
            "stack_bb": req["stack_bb"],
            "hero_pos": str(req["hero_pos"]),
            "opener_pos": str(req.get("opener_pos", "")),
            "ctx": str(req.get("ctx", "")),
            "opener_action": str(req.get("opener_action", "")),
        }
        y169 = self.rng_pre.predict([feats])[0]  # List[float] length 169
        summaries = summarize_169(y169)

        # 2) Equity
        eq = self._equity(req)

        # 3) Exploit/Population response mix
        ex = self._exploit(req)  # {p_fold, p_call, p_raise, weight}
        pop = self._population(req)  # optional
        opp_mix = self._blend_response(ex, pop)

        # 4) Coarse logits → probs
        logits = self._blend_to_logits(eq, {"y169": y169, "summaries": summaries}, ex, pop, req)
        actions, probs = self._softmax_logits(logits, keys=["FOLD", "CALL", "RAISE"])

        # 5) EV per action (simple one-step template)
        pot_bb   = float(req.get("pot_bb", 0.0))
        stack_bb = float(req.get("stack_bb", 0.0))
        evs = [self._ev_one(a, pot_bb, stack_bb, eq, opp_mix) for a in actions]

        # 6) Guardrails (optional tweaks)
        actions, probs, notes = self._guardrails(actions, probs, req, eq, {"y169": y169, "summaries": summaries})

        return {
            "actions": actions,
            "probs": probs,
            "evs": evs,
            "debug": {
                "street": 0,
                "equity": eq,
                "range": summaries,
                "exploit": ex,
                "population": pop,
                "response_mix": opp_mix,
            },
            "notes": notes,
        }

    # ---------- postflop ----------
    def _predict_postflop(self, req: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Policy probs directly from the policy infer
        row = self._postflop_policy_row(req)
        p_vec = self.pol_post.predict_proba([row])[0]  # np.ndarray [V]
        actions, probs = self._policy_vec_to_actions(p_vec)

        # 2) Equity + response mix
        eq = self._equity(req)
        ex = self._exploit(req)
        pop = self._population(req)
        opp_mix = self._blend_response(ex, pop)

        # 3) EV per action
        pot_bb   = float(req.get("pot_bb", 0.0))
        stack_bb = float(req.get("eff_stack_bb", req.get("stack_bb", 0.0)))
        evs = [self._ev_one(a, pot_bb, stack_bb, eq, opp_mix) for a in actions]

        # 4) Guardrails
        actions, probs, notes = self._guardrails(actions, probs, req, eq, {})

        return {
            "actions": actions,
            "probs": probs,
            "evs": evs,
            "debug": {
                "street": int(req["street"]),
                "equity": eq,
                "exploit": ex,
                "population": pop,
                "response_mix": opp_mix,
            },
            "notes": notes,
        }

    # ---------- helpers: inputs for postflop policy ----------
    def _postflop_policy_row(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        row = {
            "hero_pos": str(req["hero_pos"]),
            "ip_pos":   str(req.get("ip_pos", req.get("hero_pos"))),
            "oop_pos":  str(req.get("oop_pos", req.get("villain_pos", "BB"))),
            "ctx":      str(req.get("ctx", "")),
            "street":   int(req["street"]),
            "pot_bb":   float(req.get("pot_bb", 0.0)),
            "eff_stack_bb": float(req.get("eff_stack_bb", req.get("stack_bb", 0.0))),
            "actor":    str(req.get("actor", "ip")).lower(),
        }
        if "board_mask_52" in req:
            row["board_mask_52"] = req["board_mask_52"]
        return row

    def _policy_vec_to_actions(self, p_vec: Sequence[float]) -> Tuple[List[str], List[float]]:
        from ml.models.postflop_policy_net import ACTION_VOCAB as _VOC
        return list(_VOC), list(map(float, p_vec))

    # ---------- equity / exploit / population ----------
    def _equity(self, req: Mapping[str, Any]) -> Dict[str, float]:
        street = int(req["street"])
        hero_hand = req.get("hero_hand", "")
        if not hero_hand:
            raise ValueError("`hero_hand` is required for equity.")
        board_cluster_id = req.get("board_cluster_id", None)
        if street > 0 and board_cluster_id is None:
            board = req.get("board", "")
            if board and self.clusterer is not None:
                board_cluster_id = int(self.clusterer.predict([board])[0])
        hand_code = hand_to_169(hero_hand) if street == 0 else None
        hand_id   = None if street == 0 else hand_to_id(hero_hand)
        p_win, p_tie, p_lose = self.eq.predict(
            street=street,
            stack_bb=req["stack_bb"],
            hero_pos=req["hero_pos"],
            opener_action=req.get("opener_action", ""),
            board_cluster_id=board_cluster_id,
            hand_code=hand_code,
            hand_id=hand_id,
        )
        return {"p_win": float(p_win), "p_tie": float(p_tie), "p_lose": float(p_lose)}

    def _exploit(self, req: Mapping[str, Any]) -> Dict[str, float]:
        ef = req.get("exploit_features")
        p_fold, p_call, p_raise, weight = self.expl.predict(ef, context=req)
        return {"p_fold": float(p_fold), "p_call": float(p_call), "p_raise": float(p_raise), "weight": float(weight)}

    def _population(self, req: Mapping[str, Any]) -> Optional[Dict[str, float]]:
        if not self.pop:
            return None
        y = self.pop.predict(
            stakes_id=req["stakes_id"],
            street=req["street"],
            ctx_id=req.get("ctx_id", 10),
            hero_pos=req["hero_pos"],
            villain_pos=req["villain_pos"],
        )
        return {"p_fold": float(y[0]), "p_call": float(y[1]), "p_raise": float(y[2])}

    # ---------- response blend + logits ----------
    def _blend_response(self, ex: Dict[str, float], pop: Optional[Dict[str, float]]) -> Dict[str, float]:
        base = pop or {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3}
        w = min(1.0, (ex.get("weight", 0.0)) / self.p.get("exploit_full_weight", 200))
        return {
            "p_fold": (1 - w) * base["p_fold"] + w * ex["p_fold"],
            "p_call": (1 - w) * base["p_call"] + w * ex["p_call"],
            "p_raise": (1 - w) * base["p_raise"] + w * ex["p_raise"],
        }

    def _blend_to_logits(self, eq, rng, ex, pop, req) -> Dict[str, float]:
        alpha = self.p.get("alpha", 0.35)
        beta  = self.p.get("beta",  0.35)
        gamma = self.p.get("gamma", 0.30)

        base = pop or {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3}
        w = min(1.0, (ex["weight"] or 0) / self.p.get("exploit_full_weight", 200))
        ex_mix = {
            "p_fold": (1 - w) * base["p_fold"] + w * ex["p_fold"],
            "p_call": (1 - w) * base["p_call"] + w * ex["p_call"],
            "p_raise": (1 - w) * base["p_raise"] + w * ex["p_raise"],
        }

        e = float(eq["p_win"]) + 0.5 * float(eq["p_tie"])
        bias_agg = max(0.0, e - 0.5)

        def slog(p): return math.log(max(p, 1e-9))
        return {
            "FOLD":  alpha * slog(ex_mix["p_fold"]) + beta * slog(base["p_fold"]) - gamma * bias_agg,
            "CALL":  alpha * slog(ex_mix["p_call"]) + beta * slog(base["p_call"]),
            "RAISE": alpha * slog(ex_mix["p_raise"]) + beta * slog(base["p_raise"]) + gamma * bias_agg,
        }

    def _softmax_logits(self, logits: Mapping[str, float], keys: Sequence[str]) -> Tuple[List[str], List[float]]:
        xs = [float(logits[k]) for k in keys]
        m = max(xs)
        exps = [math.exp(x - m) for x in xs]
        s = sum(exps)
        probs = [x / s for x in exps]
        return list(keys), probs

    # ---------- EV helpers ----------
    def _ev_one(
        self,
        action: str,
        pot_bb: float,
        stack_bb: float,
        eq: Mapping[str, float],
        opp: Mapping[str, float],
    ) -> float:
        """One-step EV (incremental vs current pot). Replace later with a richer tree if you like."""
        eq_win = float(eq["p_win"]); eq_tie = float(eq["p_tie"])
        p_fold = self._safe(opp["p_fold"]); p_call = self._safe(opp["p_call"]); p_raise = self._safe(opp["p_raise"])
        invest = self._size_map(action, pot_bb, stack_bb)
        invest = min(invest, stack_bb)
        e = self._safe(eq_win + 0.5 * eq_tie)

        up = action.upper()
        if up == "FOLD":
            return 0.0
        if up == "CHECK":
            return 0.0
        if up == "CALL":
            call_amt = min(stack_bb, 0.5 * pot_bb)  # proxy if unknown facing bet
            final_pot = pot_bb + call_amt + call_amt
            return e * final_pot - (1 - e) * call_amt

        # Bets/Raises
        if up.startswith("BET_") or up.startswith("DONK_") or up.startswith("RAISE_") or up == "ALLIN":
            win_if_fold = pot_bb
            call_amt = invest
            final_pot = pot_bb + call_amt + call_amt
            ev_call  = e * final_pot - (1 - e) * call_amt
            ev_raise = ev_call  # conservative fallback
            return p_fold * win_if_fold + p_call * ev_call + p_raise * ev_raise

        return 0.0

    def _size_map(self, action: str, pot_bb: float, stack_bb: float) -> float:
        up = action.upper()
        if up.startswith("BET_"):
            return float(up.split("_")[1]) / 100.0 * pot_bb
        if up.startswith("DONK_"):
            return float(up.split("_")[1]) / 100.0 * pot_bb
        if up.startswith("RAISE_"):
            # Treat suffix as % multiplier of pot if facing bet unknown (e.g., RAISE_200 -> 2.0 * pot)
            mult = float(up.split("_")[1]) / 100.0
            return mult * pot_bb
        if up == "ALLIN":
            return max(0.0, float(stack_bb))
        return 0.0

    @staticmethod
    def _safe(x: float, lo: float = 1e-9, hi: float = 1 - 1e-9) -> float:
        return max(lo, min(hi, float(x)))

    # ---------- guardrails ----------
    def _guardrails(
        self,
        actions: List[str],
        probs: List[float],
        req: Mapping[str, Any],
        eq: Mapping[str, float],
        rng_feats: Mapping[str, Any],
    ) -> Tuple[List[str], List[float], List[str]]:
        notes: List[str] = []
        # example: low SPR + decent equity -> push aggression a bit
        spr = float(req.get("stack_bb", 0.0)) / max(1e-6, float(req.get("pot_bb", 1.0)))
        if spr < self.p.get("spr_shove_threshold", 1.5) and float(eq.get("p_win", 0.0)) > 0.55:
            actions, probs = self._tilt_toward_raise(actions, probs, 0.15)
            notes.append("spr_low_push_aggression")
        # TODO: mask illegal actions if you have a legality mask
        return actions, probs, notes

    @staticmethod
    def _tilt_toward_raise(actions: List[str], probs: List[float], amount: float) -> Tuple[List[str], List[float]]:
        if "RAISE" not in actions:
            return actions, probs
        i = actions.index("RAISE")
        add = min(amount, 1.0 - probs[i])
        # take equally from others
        rest = [j for j,_ in enumerate(actions) if j != i]
        take_each = add / max(len(rest), 1)
        new_probs = probs[:]
        for j in rest:
            new_probs[j] = max(0.0, new_probs[j] - take_each)
        s = sum(new_probs)
        if s > 0:
            new_probs = [p / s for p in new_probs]
        new_probs[i] = min(1.0, new_probs[i] + add)  # renorm safety
        s = sum(new_probs)
        return actions, [p / s for p in new_probs]