from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from ml.inference.policy.types import PolicyRequest, PolicyResponse
from ml.inference.policy.utils import apply_legality_mask, ev_one, guardrails


class PostflopPolicyInfer:
    """
    Postflop policy inference (Flop/Turn/River).

    Inputs:
      - pol_post: model with .predict_proba([row]) -> array-like [B,V]
      - pop     : population net infer (object with .predict_proba / .predict; returns dict p_fold,p_call,p_raise)
      - exploit : exploit infer (object with .predict(req_like) -> dict p_fold,p_call,p_raise, weight)
      - equity  : equity infer (object with .predict([row]) -> [[p_win,p_tie,p_lose]])
      - clusterer: optional board clusterer with .predict([board]) -> [cluster_id]
      - params: behavior knobs (see defaults below)
      - action_vocab: override action vocabulary for policy head

    Output:
      Dict (or PolicyResponse if you wire it) with:
        actions: [str], probs: [float], evs: [float], debug: {...}, notes: [str]
    """

    def __init__(
        self,
        *,
        pol_post: Any = None,
        pop: Any = None,
        exploit: Any = None,
        equity: Any = None,
        clusterer: Any = None,
        params: Optional[Dict[str, Any]] = None,
        action_vocab: Optional[List[str]] = None,
    ):
        self.pol_post = pol_post
        self.pop = pop
        self.exploit = exploit
        self.equity = equity
        self.clusterer = clusterer

        # Tunables (with safe defaults)
        self.p: Dict[str, Any] = {
            # how strongly exploit overrides population, based on exploit["weight"]
            # effective weight = clip(exploit_weight / exploit_full_weight, 0..1)
            "exploit_full_weight": 200.0,
            # minimum probability per action before renormalization
            "prob_floor": 1e-6,
            # optional hard caps (per street) for aggression/folds, etc. (not enforced by default)
            # "cap_raise_prob": {1: 0.85, 2: 0.85, 3: 0.90}
        }
        if params:
            self.p.update(params)

        # vocab
        if action_vocab:
            self.action_vocab = list(action_vocab)
        else:
            self.action_vocab = [
                "CHECK",
                "BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100",
                "DONK_33",
                "FOLD", "CALL",
                "RAISE_150", "RAISE_200", "RAISE_300",
                "ALLIN",
            ]

    # ------------------ public API ------------------
    def _mk_response(self, actions, probs, evs, debug, notes):
        # If PolicyResponse is available, return an instance; else return dict fallback.
        try:
            # if import succeeded and class is dataclass
            if 'PolicyResponse' in globals() and is_dataclass(PolicyResponse):
                return PolicyResponse(
                    actions=list(actions),
                    probs=list(probs),
                    evs=list(evs),
                    debug=dict(debug),
                    notes=list(notes),
                )
        except Exception:
            pass
        # fallback: plain dict (keeps runtime flexible)
        return {
            "actions": list(actions),
            "probs": list(probs),
            "evs": list(evs),
            "debug": dict(debug),
            "notes": list(notes),
        }

    def predict(self, req: PolicyRequest | Dict[str, Any]) -> PolicyResponse:
        r = self._req_to_dict(req)
        street = int(r.get("street", 1))

        # 1) Policy probs
        if self.pol_post is not None:
            row = self._postflop_policy_row(r)
            raw = self.pol_post.predict_proba([row])
            p_vec = self._coerce_to_numpy(raw)[0]
            actions, probs = self._policy_vec_to_actions(p_vec)
        else:
            actions = list(self.action_vocab)
            probs = self._stub_probs(r, actions)

        # 2) Legality
        actions, probs = apply_legality_mask(actions, probs, r)

        # 3) Signals
        eq  = self._equity(r)
        pop = self._population(r)   # may be None
        ex  = self._exploit(r)      # dict with p_* + weight
        opp = self._blend_response(ex, pop)

        # 4) EVs
        pot_bb   = float(r.get("pot_bb", 0.0))
        stack_bb = float(r.get("eff_stack_bb", r.get("stack_bb", 0.0)))
        evs = [ev_one(a, pot_bb, stack_bb, eq, opp) for a in actions]

        # 5) Guardrails
        actions, probs, notes = guardrails(actions, probs, r, params=self.p)

        debug = {
            "street": street,
            "equity": eq,
            "population": pop,
            "exploit": ex,
            "response_mix": opp,
        }

        return self._mk_response(actions, probs, evs, debug, notes)

    # ------------------ feature row for postflop policy ------------------
    def _postflop_policy_row(self, req: Dict[str, Any]) -> Dict[str, Any]:
        street = int(req.get("street", 1))
        pot_bb = float(req.get("pot_bb", 0.0))
        eff_stack_bb = float(req.get("eff_stack_bb", req.get("stack_bb", 100.0)))

        # optional villain pos (helpful if your model includes it)
        villain_pos = None
        pos_by = req.get("position_by_player") or {}
        v = req.get("villain_actor")
        if isinstance(pos_by, dict) and v in pos_by:
            villain_pos = pos_by[v].get("name")
        villain_pos = villain_pos or req.get("villain_pos") or "BTN"

        base = {
            "street": street,
            "pot_bb": pot_bb,
            "eff_stack_bb": eff_stack_bb,
            "villain_pos": villain_pos,
            # add keys your trained model expects: hero_pos/ip_pos/oop_pos/ctx/board_cluster/etc.
        }

        # Respect model-declared order if exposed
        feat_order = getattr(self.pol_post, "feature_order", None)
        if isinstance(feat_order, (list, tuple)) and feat_order:
            row = {}
            for k in feat_order:
                row[k] = base.get(k)
            return row
        return base

    # ------------------ signals ------------------
    def _equity(self, req: Dict[str, Any]) -> Dict[str, float]:
        if self.equity is None:
            return {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}
        try:
            street = int(req.get("street", 1))
            hero_hand = str(req.get("hero_hand", "")).strip()
            if not hero_hand:
                return {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}

            row = {"street": street, "hand_id": hero_hand}
            # attach board cluster if we can
            cluster_id = req.get("board_cluster_id", None)
            if cluster_id is None and self.clusterer is not None and street > 0:
                board = str(req.get("board", "")).strip()
                if board:
                    try:
                        cluster_id = int(self.clusterer.predict([board])[0])
                    except Exception:
                        cluster_id = None
            if cluster_id is not None:
                row["board_cluster_id"] = int(cluster_id)

            out = self.equity.predict([row])  # -> [[p_win, p_tie, p_lose]]
            if not out:
                return {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}
            p_win, p_tie, p_lose = out[0]
            return {"p_win": float(p_win), "p_tie": float(p_tie), "p_lose": float(p_lose)}
        except Exception:
            return {"p_win": 0.5, "p_tie": 0.0, "p_lose": 0.5}

    def _population(self, req: Dict[str, Any]) -> Optional[Dict[str, float]]:
        if self.pop is None:
            return None
        try:
            # Support both .predict_proba(dict_of_ids) and .predict(req_like)
            if hasattr(self.pop, "predict_proba"):
                # You may need an id-feats builder here; for now assume caller passes it in req["pop_feats"]
                feats = req.get("pop_feats")
                if feats is None:
                    return None
                return self.pop.predict_proba(feats)
            elif hasattr(self.pop, "predict"):
                out = self.pop.predict(req)
                # normalize keys if needed
                return {
                    "p_fold": float(out.get("FOLD", out.get("p_fold", 1/3))),
                    "p_call": float(out.get("CALL", out.get("p_call", 1/3))),
                    "p_raise": float(out.get("RAISE", out.get("p_raise", 1/3))),
                }
            else:
                return None
        except Exception:
            return None

    def _exploit(self, req: Dict[str, Any]) -> Dict[str, float]:
        """
        Expect exploit.predict(req_like) -> {"p_fold","p_call","p_raise", "weight"?}
        Weight encodes confidence / sample size; blended downstream.
        """
        if self.exploit is None:
            return {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3, "weight": 0.0}
        try:
            out = self.exploit.predict(req) if hasattr(self.exploit, "predict") else self.exploit(req)
            return {
                "p_fold": float(out.get("p_fold", 1/3)),
                "p_call": float(out.get("p_call", 1/3)),
                "p_raise": float(out.get("p_raise", 1/3)),
                "weight": float(out.get("weight", out.get("w", 0.0))),
            }
        except Exception:
            return {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3, "weight": 0.0}

    def _blend_response(self, ex: Dict[str, float], pop: Optional[Dict[str, float]]) -> Dict[str, float]:
        """
        Population prior blended toward exploit as exploit weight grows.
        """
        base = pop or {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3}
        w_raw = float(ex.get("weight", 0.0))
        full = float(self.p.get("exploit_full_weight", 200.0))
        w = max(0.0, min(1.0, w_raw / full))
        return {
            "p_fold": (1 - w) * base["p_fold"] + w * ex.get("p_fold", base["p_fold"]),
            "p_call": (1 - w) * base["p_call"] + w * ex.get("p_call", base["p_call"]),
            "p_raise": (1 - w) * base["p_raise"] + w * ex.get("p_raise", base["p_raise"]),
        }

    # ------------------ small utils ------------------
    def _policy_vec_to_actions(self, p_vec: Sequence[float]) -> Tuple[List[str], List[float]]:
        if len(p_vec) != len(self.action_vocab):
            raise ValueError(f"Policy output mismatch: got {len(p_vec)} vs vocab {len(self.action_vocab)}")
        return list(self.action_vocab), [float(x) for x in p_vec]

    def _coerce_to_numpy(self, out) -> np.ndarray:
        if isinstance(out, np.ndarray):
            return out
        if torch.is_tensor(out):
            return out.detach().cpu().numpy()
        # list/tuple -> np
        return np.asarray(out, dtype=np.float32)

    def _stub_probs(self, req: Dict[str, Any], actions: List[str]) -> List[float]:
        """
        Simple prior:
          - Not facing bet: mass on CHECK/BET_33
          - Facing bet: mass on CALL / (RAISE_200|ALLIN) / some FOLD
        """
        facing = bool(req.get("facing_bet", False))
        scores = []
        for a in actions:
            up = a.upper()
            if not facing:
                if up == "CHECK":
                    scores.append(0.55)
                elif up in ("BET_25", "BET_33"):
                    scores.append(0.35)
                else:
                    scores.append(0.10 if up in ("BET_50", "BET_66") else 0.0)
            else:
                if up == "CALL":
                    scores.append(0.45)
                elif up in ("RAISE_200", "ALLIN"):
                    scores.append(0.40)
                elif up == "FOLD":
                    scores.append(0.15)
                else:
                    scores.append(0.0)
        s = sum(scores) or 1.0
        return [x / s for x in scores]

    @staticmethod
    def _req_to_dict(req: PolicyRequest | Dict[str, Any]) -> Dict[str, Any]:
        if is_dataclass(req):
            return asdict(req)  # type: ignore[arg-type]
        return dict(req)