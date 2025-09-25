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


# --- add near imports ---
import math

class PolicyInfer:
    def __init__(self, deps: PolicyInferDeps):
        if deps.exploit is None:
            raise ValueError("exploit infer is required")
        if deps.equity is None:
            raise ValueError("equity infer is required")
        if deps.range_pre is None:
            raise ValueError("range_pre (RangeNetPreflopInfer) is required")

        # postflop policy can be stubbed (see below)
        self.pol_post = deps.policy_post

        self.pop   = deps.pop
        self.expl  = deps.exploit
        self.eq    = deps.equity
        self.rng_pre  = deps.range_pre
        self.clusterer = deps.clusterer
        self.p = deps.params or {}

        # Cache vocab once (and allow fallback for stub)
        try:
            from ml.models.postflop_policy_net import ACTION_VOCAB as _VOC
            self.action_vocab = list(_VOC)
        except Exception:
            # Safe default for stub/testing; adjust to your real vocab as needed
            self.action_vocab = ["CHECK", "BET_33", "BET_66", "FOLD", "CALL", "RAISE_200", "ALLIN"]

    # ---------- public ----------
    def predict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        street = int(req["street"])
        return self._predict_preflop(req) if street == 0 else self._predict_postflop(req)

    # ---------- postflop ----------
    def _predict_postflop(self, req: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Policy probs (stub or real)
        if self.pol_post is not None:
            row = self._postflop_policy_row(req)
            p_vec = self.pol_post.predict_proba([row])[0]  # np.ndarray [V]
            actions, probs = self._policy_vec_to_actions(p_vec)
        else:
            # ---- STUB for smoke testing: simple legal distribution
            actions = self.action_vocab
            probs = self._stub_probs(req, actions)

        # 1b) Legality mask (always)
        actions, probs = self._apply_legality_mask(actions, probs, req)

        # 2) Equity + response mix
        eq = self._equity(req)
        ex = self._exploit(req)
        pop = self._population(req)
        opp_mix = self._blend_response(ex, pop)

        # 3) EV per action (conservative)
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

    # ---------- equity / exploit / population ----------
    def _equity(self, req: Mapping[str, Any]) -> Dict[str, float]:
        street = int(req["street"])
        hero_hand = req.get("hero_hand", "")

        # Neutral prior if no hand available (lets you smoke test)
        if not hero_hand:
            return {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}

        board_cluster_id = req.get("board_cluster_id", None)
        if street > 0 and board_cluster_id is None and self.clusterer is not None:
            # light fallback to clusterer if available
            try:
                board = req.get("board", "")
                if board:
                    board_cluster_id = int(self.clusterer.predict([board])[0])
            except Exception:
                pass  # stay None

        # If postflop and no cluster, still proceed—equity model may accept None or will use broad prior internally
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

    def _blend_response(self, ex: Dict[str, float], pop: Optional[Dict[str, float]]) -> Dict[str, float]:
        base = pop or {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3}
        w_raw = ex.get("weight", 0.0)
        try:
            w_raw = float(w_raw)
        except Exception:
            w_raw = 0.0
        w = max(0.0, min(1.0, w_raw / float(self.p.get("exploit_full_weight", 200))))
        return {
            "p_fold": (1 - w) * base["p_fold"] + w * ex.get("p_fold", base["p_fold"]),
            "p_call": (1 - w) * base["p_call"] + w * ex.get("p_call", base["p_call"]),
            "p_raise": (1 - w) * base["p_raise"] + w * ex.get("p_raise", base["p_raise"]),
        }

    # ---------- vocab & legality ----------
    def _policy_vec_to_actions(self, p_vec) -> tuple[list[str], list[float]]:
        if len(p_vec) != len(self.action_vocab):
            raise ValueError(f"Policy output mismatch: got {len(p_vec)} vs vocab {len(self.action_vocab)}")
        return self.action_vocab, list(map(float, p_vec))

    def _apply_legality_mask(self, actions: list[str], probs: list[float], req: Mapping[str, Any]) -> tuple[list[str], list[float]]:
        pairs = [(a, p) for a, p in zip(actions, probs) if self._is_legal(a, req)]
        if not pairs:
            # conservative fallback
            return ["FOLD"], [1.0]
        acts, ps = zip(*pairs)
        s = sum(ps)
        ps = [x / s for x in ps] if s > 0 else [1.0 / len(ps)] * len(ps)
        return list(acts), list(ps)

    def _is_legal(self, action: str, req: Mapping[str, Any]) -> bool:
        """
        Very light legality: disallow CHECK if facing bet; allow others.
        (You can enrich this later with stack/pot boundary checks.)
        """
        facing_bet = bool(req.get("facing_bet", False))
        up = str(action).upper()
        if facing_bet and up == "CHECK":
            return False
        return True

    # ---------- EV helpers (conservative) ----------
    def _ev_one(
        self,
        action: str,
        pot_bb: float,
        stack_bb: float,
        eq: Mapping[str, float],
        opp: Mapping[str, float],
    ) -> float:
        eq_win = float(eq.get("p_win", 0.5)); eq_tie = float(eq.get("p_tie", 0.0))
        e = self._safe(eq_win + 0.5 * eq_tie)
        p_fold = self._safe(opp.get("p_fold", 1/3)); p_call = self._safe(opp.get("p_call", 1/3)); p_raise = self._safe(opp.get("p_raise", 1/3))
        up = str(action).upper()

        if up in ("FOLD", "CHECK"):
            return 0.0

        if up == "CALL":
            # Without a facing-bet amount, avoid making up EV
            return 0.0

        invest = min(self._size_map(up, pot_bb, stack_bb), stack_bb)
        # Symmetric proxy: if called, villain matches invest; if fold, we win current pot
        final_pot = pot_bb + invest + invest
        ev_call  = e * final_pot - (1 - e) * invest
        return p_fold * pot_bb + (p_call + p_raise) * ev_call

    def _size_map(self, up: str, pot_bb: float, stack_bb: float) -> float:
        try:
            if up.startswith("BET_") or up.startswith("DONK_"):
                return float(up.split("_")[1]) / 100.0 * pot_bb
            if up.startswith("RAISE_"):
                mult = float(up.split("_")[1]) / 100.0
                return mult * pot_bb
            if up == "ALLIN":
                return max(0.0, float(stack_bb))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _safe(x: float, lo: float = 1e-9, hi: float = 1 - 1e-9) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(lo, min(hi, v))

    # ---------- postflop stub probs ----------
    def _stub_probs(self, req: Mapping[str, Any], actions: list[str]) -> list[float]:
        """
        Very simple prior:
          - If not facing bet: put mass on CHECK and BET_33.
          - If facing bet: put mass on FOLD/CALL/RAISE_200.
        """
        facing = bool(req.get("facing_bet", False))
        scores = []
        for a in actions:
            up = a.upper()
            if not facing:
                scores.append(0.5 if up == "CHECK" else (0.4 if up == "BET_33" else 0.1 if up == "BET_66" else 0.0))
            else:
                scores.append(0.45 if up == "CALL" else (0.4 if up in ("RAISE_200", "ALLIN") else 0.15 if up == "FOLD" else 0.0))
        s = sum(scores) or 1.0
        return [x / s for x in scores]