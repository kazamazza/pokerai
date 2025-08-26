from ml.policy.utils import tilt_toward_raise, renormalize_and_mask, hand_to_id, hand_to_169, summarize_169


class PolicyInfer:
    def __init__(self,
                 pop_infer,        # PopulationNetInfer (optional if not used)
                 exploit_infer,    # ExploitNetInfer
                 range_infer,      # RangeNetInfer (pre/post based on street)
                 equity_infer_pre, # EquityPreInfer
                 equity_infer_post,# EquityPostInfer
                 board_clusterer,  # same as you used to build datasets
                 params: dict      # knobs for blending/guardrails
                 ):
        self.pop = pop_infer
        self.expl = exploit_infer
        self.rng = range_infer
        self.eq_pre = equity_infer_pre
        self.eq_post = equity_infer_post
        self.clusterer = board_clusterer
        self.p = params

    def predict(self, req: dict) -> dict:
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
                "range":  rng_feats.get("summaries", {}),
                "exploit": ex,
                "population": pop,
                "blend": {
                    "alpha": self.p.get("alpha", 0.35),
                    "beta":  self.p.get("beta", 0.35),
                    "gamma": self.p.get("gamma", 0.30),
                },
                "guardrails": notes
            }
        }

    # ---- helpers ----
    def _range_features(self, req):
        # Preflop vs Postflop inputs differ; postflop needs board cluster
        if req["street"] == 0:
            y169 = self.rng.predict_pre(
                stack_bb=req["stack_bb"],
                hero_pos=req["hero_pos"],
                opener_pos=req.get("opener_pos",""),
                opener_action=req.get("opener_action",""),
            )
        else:
            board = req.get("board","")
            cluster_id = self.clusterer.predict([board])[0]
            y169 = self.rng.predict_post(
                stack_bb=req["stack_bb"],
                hero_pos=req["hero_pos"],
                villain_pos=req["villain_pos"],
                street=req["street"],
                board_cluster_id=int(cluster_id),
            )
        # Optional: derive quick summaries for blending (e.g., top-pair+, air mass)
        summaries = summarize_169(y169)   # your small utility
        return {"y169": y169, "summaries": summaries}

    def _equity(self, req, rng_feats):
        if req["street"] == 0:
            p_win, p_tie, p_lose = self.eq_pre.predict(
                stack_bb=req["stack_bb"],
                hero_pos=req["hero_pos"],
                opener_action=req.get("opener_action",""),
                hand_code=hand_to_169(req.get("hero_hand","")),  # or pass id
            )
        else:
            board = req.get("board","")
            p_win, p_tie, p_lose = self.eq_post.predict(
                stack_bb=req["stack_bb"],
                hero_pos=req["hero_pos"],
                opener_action=req.get("opener_action",""),
                board_cluster_id=self.clusterer.predict([board])[0],
                hand_id=hand_to_id(req.get("hero_hand","")),
            )
        return {"p_win": p_win, "p_tie": p_tie, "p_lose": p_lose}

    def _exploit(self, req):
        # Either use provided rolling stats or have ExploitInfer compute them
        ef = req.get("exploit_features")
        p_fold, p_call, p_raise, weight = self.expl.predict_features(ef, context=req)
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