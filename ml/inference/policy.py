from typing import Optional, Dict, Any

from ml.features.boards import BoardClusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.exploit import ExploitNetInfer
from ml.inference.population import PopulationNetInfer
from ml.inference.rangenet import RangeNetInfer
from ml.policy.utils import tilt_toward_raise, renormalize_and_mask, hand_to_id, hand_to_169, summarize_169


class PolicyInfer:
    def __init__(
        self,
        pop_infer: PopulationNetInfer,   # optional
        exploit_infer: ExploitNetInfer,           # required
        range_infer: RangeNetInfer,             # required
        equity_infer: EquityNetInfer,             # required (unified pre+post)
        board_clusterer: BoardClusterer,          # your clusterer type
        params: Dict[str, Any],                     # knobs for blending/guardrails
    ):
        if exploit_infer is None:
            raise ValueError("exploit_infer is required")
        if range_infer is None:
            raise ValueError("range_infer is required")
        if equity_infer is None:
            raise ValueError("equity_infer is required")

        # Keep explicit attribute types so IDE/static analysis know predict() exists
        self.pop: PopulationNetInfer = pop_infer
        self.expl: ExploitNetInfer = exploit_infer
        self.rng: RangeNetInfer = range_infer
        self.eq: EquityNetInfer = equity_infer
        self.clusterer: BoardClusterer = board_clusterer
        self.p: Dict[str, Any] = params or {}

    def predict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        street = int(req["street"])

        # --- range ---
        rng_feats = self._range_features(req)

        # --- equity ---
        eq = self._equity(req, rng_feats)

        # --- exploit ---
        ex = self._exploit(req)

        # --- population (optional) ---
        pop = self._population(req)

        # --- blend ---
        logits = self._blend_to_logits(eq, rng_feats, ex, pop, req)
        actions, probs = self._postprocess_logits(logits, req)

        # --- guardrails ---
        actions, probs, notes = self._guardrails(actions, probs, req, eq, rng_feats)

        return {
            "actions": actions,
            "probs": probs,
            "debug": {
                "equity": eq,
                "range": rng_feats.get("summaries", {}),
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

    # ---- helpers ----
    def _range_features(self, req):
        """
        Build the single-row feature dict expected by RangeNetInfer and get a (169,) prob vector.
        Preflop uses opener_* fields; postflop uses villain_pos + board_cluster_id.
        """
        street = int(req["street"])
        feats = {
            "stack_bb": req["stack_bb"],
            "hero_pos": str(req["hero_pos"]),
            "street": street,
        }

        if street == 0:
            # Preflop
            feats.update({
                "opener_pos": str(req.get("opener_pos", "")),
                "opener_action": str(req.get("opener_action", "")),
            })
        else:
            # Postflop
            board = req.get("board", "")
            if board:
                cluster_id = int(self.clusterer.predict([board])[0])
            else:
                # allow caller to pass precomputed id
                cluster_id = int(req.get("board_cluster_id", -1))
            feats.update({
                "villain_pos": str(req["villain_pos"]),
                "board_cluster_id": cluster_id,
                # allow optional node_key for deeper tree extraction; default root
                "node_key": str(req.get("node_key", "root")),
            })

        # One-row inference → (169,) np.ndarray
        y169 = self.rng.predict_one(feats)

        # Optional summaries for blending/diagnostics
        summaries = summarize_169(y169.tolist())
        return {"y169": y169, "summaries": summaries}

    def _equity(self, req, _rng_feats):
        """
        Unified equity call using a single EquityNetInfer (`self.eq`) for all streets.
        Expects:
          - street: int (0=preflop, >0 postflop)
          - stack_bb, hero_pos, opener_action (always)
          - hero_hand (always; converted to 169 code preflop, to id postflop)
          - board (postflop) or board_cluster_id (optional; derived if missing and clusterer available)
        """
        if not hasattr(self, "eq") or self.eq is None:
            raise RuntimeError("Equity infer (`self.eq`) is not initialized on PolicyInfer.")

        street = int(req["street"])
        hero_hand = req.get("hero_hand", "")
        opener_action = req.get("opener_action", "")

        if not hero_hand:
            raise ValueError("`hero_hand` is required for equity computation (e.g., 'AsKd').")

        # derive board_cluster_id if needed (postflop)
        board_cluster_id = req.get("board_cluster_id", None)
        if street > 0 and board_cluster_id is None:
            board = req.get("board", "")
            if board and getattr(self, "clusterer", None) is not None:
                board_cluster_id = int(self.clusterer.predict([board])[0])
            # else: leave as None if not available; Equity wrapper should handle/validate

        # hand representation per street
        hand_code = hand_to_169(hero_hand) if street == 0 else None
        hand_id = None if street == 0 else hand_to_id(hero_hand)

        p_win, p_tie, p_lose = self.eq.predict(
            street=street,
            stack_bb=req["stack_bb"],
            hero_pos=req["hero_pos"],
            opener_action=opener_action,
            board_cluster_id=board_cluster_id,  # None for preflop
            hand_code=hand_code,  # used preflop
            hand_id=hand_id,  # used postflop
        )
        return {"p_win": p_win, "p_tie": p_tie, "p_lose": p_lose}

    def _exploit(self, req):
        # Either use provided rolling stats or have ExploitInfer compute them
        ef = req.get("exploit_features")
        p_fold, p_call, p_raise, weight = self.expl.predict(ef, context=req)
        return {"p_fold": p_fold, "p_call": p_call, "p_raise": p_raise, "weight": weight}

    def _population(self, req):
        if not self.pop:
            return None
        y = self.pop.predict(stakes_id=req["stakes_id"],
                             street=req["street"],
                             ctx_id=req.get("ctx_id", 10),
                             hero_pos=req["hero_pos"],
                             villain_pos=req["villain_pos"])
        return {"p_fold": y[0], "p_call": y[1], "p_raise": y[2]}

    def _blend_to_logits(self, eq, rng, ex, pop, req):
        # Example: combine as weighted log-space mixture + equity adjustment
        alpha = self.p.get("alpha", 0.35)  # exploit weight
        beta  = self.p.get("beta",  0.35)  # population weight
        gamma = self.p.get("gamma", 0.30)  # equity-derived delta

        # Base priors
        base = pop or {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3}

        # Exploit smoothing by sample size
        w = min(1.0, (ex["weight"] or 0) / self.p.get("exploit_full_weight", 200))
        ex_mix = {
            "p_fold": (1-w)*base["p_fold"] + w*ex["p_fold"],
            "p_call": (1-w)*base["p_call"] + w*ex["p_call"],
            "p_raise":(1-w)*base["p_raise"]+ w*ex["p_raise"],
        }

        # Equity → simple bias: higher equity pushes away from FOLD, toward CALL/RAISE
        e = eq["p_win"] + 0.5*eq["p_tie"]
        bias_fold = max(0.0, 0.5 - e)      # prefer folding with low equity
        bias_agg  = max(0.0, e - 0.5)      # prefer aggression with high equity

        # Logits
        import math
        def safe_log(p): return math.log(max(p, 1e-9))
        log_fold = alpha*safe_log(ex_mix["p_fold"]) + beta*safe_log(base["p_fold"]) - gamma*bias_agg
        log_call = alpha*safe_log(ex_mix["p_call"]) + beta*safe_log(base["p_call"])
        log_raise= alpha*safe_log(ex_mix["p_raise"])+ beta*safe_log(base["p_raise"]) + gamma*bias_agg

        return {"FOLD": log_fold, "CALL": log_call, "RAISE": log_raise}

    def _postprocess_logits(self, logits, req):
        # If you have raise size buckets, split "RAISE" into [RAISE_50, RAISE_75, ALLIN] here
        import math
        keys = ["FOLD","CALL","RAISE"]
        xs = [logits[k] for k in keys]
        m = max(xs); exps = [math.exp(x-m) for x in xs]
        s = sum(exps); probs = [x/s for x in exps]
        return keys, probs

    def _guardrails(self, actions, probs, req, eq, rng):
        notes = []
        # Example: if SPR < 1.5 and equity high → shift mass from CALL to RAISE/ALLIN
        spr = req["stack_bb"] / max(1e-6, req["pot_bb"])
        if spr < self.p.get("spr_shove_threshold", 1.5) and eq["p_win"] > 0.55:
            actions, probs = tilt_toward_raise(actions, probs, amount=0.15)
            notes.append("spr_low_push_aggression")
        # Remove illegal actions by street/game rules if needed
        actions, probs = renormalize_and_mask(actions, probs, mask=set())
        return actions, probs, notes