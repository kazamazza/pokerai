from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.datasets.postflop_rangenet import PostflopPolicyDatasetParquet, postflop_policy_collate_fn
from ml.features.boards import BoardClusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.exploit import ExploitNetInfer
from ml.inference.population import PopulationNetInfer
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import RangeNetPreflopInfer
from ml.models.postflop_policy_net import VOCAB_SIZE
from ml.policy.utils import tilt_toward_raise, renormalize_and_mask, hand_to_id, hand_to_169, summarize_169
from typing import Any, Dict, Mapping, Sequence
from dataclasses import dataclass

import math


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
    Routes to:
      - Preflop:   RangeNetPreflop + Equity + Exploit/Pop blend -> {FOLD,CALL,RAISE} (coarse)
      - Postflop:  PostflopPolicyInfer (ACTION_VOCAB) (+ guardrails via equity/exploit if desired)
    """

    def __init__(self, deps: PolicyInferDeps):
        p = deps.params or {}
        # required checks
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
        self.p = p

    # ---------- public ----------
    def predict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        street = int(req["street"])
        if street == 0:
            return self._predict_preflop(req)
        else:
            return self._predict_postflop(req)

    @torch.no_grad()
    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """
        rows: list of dicts matching dataset schema
        returns: np.ndarray [B,V] with probs over ACTION_VOCAB
        """
        ds = PostflopPolicyDatasetParquet.from_rows(
            rows,  # you'd need a small `from_rows` helper to bypass parquet file
            device=self.device
        )
        dl = DataLoader(ds, batch_size=len(rows), collate_fn=postflop_policy_collate_fn)
        for x_cat, x_cont, *_ in dl:
            li, lo = self.model(x_cat, x_cont)  # logits_ip, logits_oop
            logits = li + lo  # simple merge if you want, or pick by actor
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        return np.empty((0, VOCAB_SIZE))

    # ---------- preflop ----------
    def _predict_preflop(self, req: Dict[str, Any]) -> Dict[str, Any]:
        # Range (169)
        feats = {
            "stack_bb": req["stack_bb"],
            "hero_pos": str(req["hero_pos"]),
            "opener_pos": str(req.get("opener_pos", "")),
            "ctx": str(req.get("ctx", "")),
            "opener_action": str(req.get("opener_action", "")),
        }
        y169 = self.rng_pre.predict([feats])[0]           # List[float] length 169
        summaries = summarize_169(y169)

        # Equity
        eq = self._equity(req)

        # Exploit/Population
        ex = self._exploit(req)
        pop = self._population(req)

        # Blend to coarse logits {FOLD,CALL,RAISE}
        logits = self._blend_to_logits(eq, {"y169": y169, "summaries": summaries}, ex, pop, req)
        actions, probs = self._postprocess_logits(logits, req)

        # Guardrails
        actions, probs, notes = self._guardrails(actions, probs, req, eq, {"y169": y169, "summaries": summaries})

        return {
            "actions": actions,
            "probs": probs,
            "debug": {
                "street": 0,
                "equity": eq,
                "range": summaries,
                "exploit": ex,
                "population": pop,
                "blend": {
                    "alpha": self.p.get("alpha", 0.35),
                    "beta":  self.p.get("beta", 0.35),
                    "gamma": self.p.get("gamma", 0.30),
                },
                "guardrails": notes,
            },
        }

    def _predict_postflop(self, req: Dict[str, Any]) -> Dict[str, Any]:
        # Build row and call PostflopPolicyInfer
        row = self._postflop_policy_row(req)
        p_vec = self.pol_post.predict_proba([row])[0]  # [V] probs over ACTION_VOCAB
        actions, probs = self._postflop_policy_to_actions(p_vec)

        # Equity + exploit guardrails (optional)
        eq = self._equity(req)
        ex = self._exploit(req)
        pop = self._population(req)
        actions, probs, notes = self._guardrails(actions, probs, req, eq, {})

        return {
            "actions": actions,
            "probs": probs,
            "debug": {
                "street": req["street"],
                "equity": eq,
                "exploit": ex,
                "population": pop,
                "guardrails": notes,
            },
        }

    # ---------- helpers ----------
    def _postflop_policy_row(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Build the single-row dict expected by PostflopPolicyInfer.
        Required (categoricals must match sidecar feature_order):
          hero_pos, ip_pos, oop_pos, ctx, street
        Continuous:
          pot_bb, eff_stack_bb (or stack_bb), optional board_mask_52
        Also pass 'actor' ("ip"|"oop") to select the correct head.
        """
        row = {
            "hero_pos": str(req["hero_pos"]),
            "ip_pos":   str(req.get("ip_pos", req.get("hero_pos"))),
            "oop_pos":  str(req.get("oop_pos", req.get("villain_pos", "BB"))),
            "ctx":      str(req.get("ctx", "")),
            "street":   int(req["street"]),
            "pot_bb":   float(req.get("pot_bb", 0.0)),
            "eff_stack_bb": float(req.get("eff_stack_bb", req.get("stack_bb", 0.0))),
            "actor":    str(req.get("actor", "ip")).lower(),  # default to IP if not provided
        }
        # Optional board mask
        if "board_mask_52" in req:
            row["board_mask_52"] = req["board_mask_52"]
        return row

    def _postflop_range_feats(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        # Minimal set if you trained a postflop range model
        return {
            "stack_bb": float(req.get("stack_bb", 0.0)),
            "villain_pos": str(req.get("villain_pos", "")),
            "board_cluster_id": self._maybe_cluster(req.get("board")) if "board_cluster_id" not in req else int(req["board_cluster_id"]),
            "ctx": str(req.get("ctx", "")),
            "street": int(req["street"]),
        }

    def _maybe_cluster(self, board_str: str | None) -> int | None:
        if not board_str:
            return None
        if self.clusterer is None:
            return None
        return int(self.clusterer.predict([board_str])[0])

    # ---- equity/exploit/pop (same spirit as your previous code) ----
    def _equity(self, req: Mapping[str, Any]) -> Dict[str, float]:
        street = int(req["street"])
        hero_hand = req.get("hero_hand", "")
        if not hero_hand:
            raise ValueError("`hero_hand` is required for equity.")
        board_cluster_id = req.get("board_cluster_id", None)
        if street > 0 and board_cluster_id is None:
            board = req.get("board", "")
            if board:
                board_cluster_id = self._maybe_cluster(board)
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
        return {"p_win": p_win, "p_tie": p_tie, "p_lose": p_lose}

    def _exploit(self, req: Mapping[str, Any]) -> Dict[str, float]:
        ef = req.get("exploit_features")
        p_fold, p_call, p_raise, weight = self.expl.predict(ef, context=req)
        return {"p_fold": p_fold, "p_call": p_call, "p_raise": p_raise, "weight": weight}

    def _population(self, req: Mapping[str, Any]) -> Dict[str, float] | None:
        if not self.pop:
            return None
        y = self.pop.predict(
            stakes_id=req["stakes_id"],
            street=req["street"],
            ctx_id=req.get("ctx_id", 10),
            hero_pos=req["hero_pos"],
            villain_pos=req["villain_pos"],
        )
        return {"p_fold": y[0], "p_call": y[1], "p_raise": y[2]}

    # ---- coarse blend (preflop) ----
    def _blend_to_logits(self, eq, rng, ex, pop, req):
        alpha = self.p.get("alpha", 0.35)
        beta  = self.p.get("beta",  0.35)
        gamma = self.p.get("gamma", 0.30)

        base = pop or {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3}

        w = min(1.0, (ex["weight"] or 0) / self.p.get("exploit_full_weight", 200))
        ex_mix = {
            "p_fold": (1-w)*base["p_fold"] + w*ex["p_fold"],
            "p_call": (1-w)*base["p_call"] + w*ex["p_call"],
            "p_raise":(1-w)*base["p_raise"]+ w*ex["p_raise"],
        }

        e = eq["p_win"] + 0.5*eq["p_tie"]
        bias_agg  = max(0.0, e - 0.5)

        def safe_log(p): return math.log(max(p, 1e-9))
        log_fold = alpha*safe_log(ex_mix["p_fold"]) + beta*safe_log(base["p_fold"]) - gamma*bias_agg
        log_call = alpha*safe_log(ex_mix["p_call"]) + beta*safe_log(base["p_call"])
        log_raise= alpha*safe_log(ex_mix["p_raise"])+ beta*safe_log(base["p_raise"]) + gamma*bias_agg

        return {"FOLD": log_fold, "CALL": log_call, "RAISE": log_raise}

    def _postprocess_logits(self, logits, req):
        keys = ["FOLD","CALL","RAISE"]
        xs = [logits[k] for k in keys]
        m = max(xs)
        exps = [math.exp(x-m) for x in xs]
        s = sum(exps)
        probs = [x/s for x in exps]
        return keys, probs

    def _postflop_policy_to_actions(self, p_vec: Sequence[float]) -> tuple[list[str], list[float]]:
        # Trust the policy head ordering (ACTION_VOCAB inside the policy module)
        from ml.models.postflop_policy_net import ACTION_VOCAB
        return list(ACTION_VOCAB), list(p_vec)

    def _guardrails(self, actions, probs, req, eq, rng):
        notes = []
        # Example: equity + low SPR → nudge mass toward aggression
        spr = float(req.get("stack_bb", 0.0)) / max(1e-6, float(req.get("pot_bb", 1.0)))
        if spr < self.p.get("spr_shove_threshold", 1.5) and eq.get("p_win", 0.0) > 0.55:
            actions, probs = tilt_toward_raise(actions, probs, amount=0.15)
            notes.append("spr_low_push_aggression")

        # Mask illegal actions (if you have a legality mask—placeholder)
        actions, probs = renormalize_and_mask(actions, probs, mask=set())
        return actions, probs, notes