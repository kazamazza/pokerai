from typing import Any, Dict, Union, List, Optional, Tuple
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.engines.postflop import PostflopBaselineProvider, PostflopMaskBuilder, \
    SignalsBundler, LogitShaper, PromotionApplier, DistributionBuilder, PostflopContextResolver
from ml.inference.policy.engines.preflop import PreflopEngine
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.inference.policy.projection import FCRProjector
from ml.inference.policy.signals import SignalCollector
from ml.inference.policy.types import PolicyRequest, PolicyResponse
import torch
from ml.inference.promotion.config import PromotionConfig
from ml.inference.promotion.gateway import PromotionGateway


class PolicyInfer:
    def __init__(self, deps: PolicyInferDeps, blend_cfg: PolicyBlendConfig | None = None):
        self.p: Dict[str, Any] = deps.params or {}
        self.blend = blend_cfg or PolicyBlendConfig.default()

        if deps.equity is None:
            raise ValueError("equity infer is required")
        if deps.range_pre is None:
            raise ValueError("range_pre (PreflopPolicy) is required")
        if deps.policy_post is None:
            raise ValueError("policy_post (PostflopPolicyRouter) is required")

        self.pol_post = deps.policy_post
        self.pop = deps.pop
        self.expl = deps.exploit
        self.eq = deps.equity
        self.range_pre = deps.range_pre
        self.clusterer = deps.clusterer
        self.ev_router = deps.ev

        if self.clusterer is not None and hasattr(self.pol_post, "set_clusterer"):
            try:
                self.pol_post.set_clusterer(self.clusterer)
            except Exception:
                pass

        self._signals = SignalCollector(
            eq_model=deps.equity,
            expl_store=deps.exploit,
            pop_model=deps.pop,
            router=self.pol_post,
            ev_router=self.ev_router
        )

        self._proj = FCRProjector()
        self._promoter = PromotionGateway(PromotionConfig(
            w_ev=0.60, w_eq=0.30, w_expl=0.10,
            eq_gate=0.55, expl_gate=0.05, ev_cap_bb=3.0,
            tau_min=0.12, tau_max=0.35,
            max_logit_boost=8.0,
            respect_fold_when_facing=True,
            min_temp=0.6, max_temp=1.2,
        ))

        try:
            from ml.models.policy_consts import ACTION_VOCAB as _VOC, VOCAB_INDEX as _VIX
            self.action_vocab = list(_VOC)
            self._vocab_index = dict(_VIX)
        except Exception:
            pass

        if not hasattr(self.pol_post, "predict"):
            raise TypeError(f"deps.policy_post lacks predict(). Got: {type(self.pol_post)!r}")
        if not hasattr(self.range_pre, "predict"):  # bugfix: was self.rng_pre
            raise TypeError(f"deps.range_pre lacks predict(). Got: {type(self.range_pre)!r}")
        if self.eq is not None and not (hasattr(self.eq, "predict_proba") or hasattr(self.eq, "predict")):
            raise TypeError(f"deps.equity lacks predict/predict_proba(). Got: {type(self.eq)!r}")

        # Engines & pipeline pieces

        self._ctx_resolver = PostflopContextResolver(self.pol_post)
        self._base = PostflopBaselineProvider(self.pol_post, self.action_vocab, self._vocab_index)
        self._masks = PostflopMaskBuilder()
        self._sig = SignalsBundler(self._signals, self.action_vocab, self._vocab_index)
        self._shape = LogitShaper(self.blend, self._proj); self._shape.bind_vocab(self.action_vocab, self._vocab_index)
        self._promo = PromotionApplier(self._promoter)
        self._pre = PreflopEngine(self.ev_router, self.range_pre, self._promo)
        self._dist = DistributionBuilder(self.blend, self.action_vocab)

    @torch.no_grad()
    def _predict_postflop(self, req: PolicyRequest, eq_sig=None) -> PolicyResponse:
        hero_is_ip, side, ctx, ctx_dbg = self._ctx_resolver.derive(req)
        actor = "ip" if hero_is_ip else "oop"

        base = self._base.get(req, actor=actor, side=side)
        masks = self._masks.build(req, base, side=side)
        sig = self._sig.collect(req, base, side=side, actor=actor, hero_is_ip=hero_is_ip, ctx=ctx, eq_sig=eq_sig)

        allow_allin = True if getattr(req, "allow_allin", None) is None else bool(req.allow_allin)
        z, promo_dbg = self._promo.postflop(base, masks, sig, allow_allin=allow_allin)
        promo_applied = bool(promo_dbg and promo_dbg.get("applied"))

        probs, best_idx_dist = self._dist.build(z, masks, req, promo_applied=promo_applied)

        # --- best action policy: promotion-first ---
        best_action_source = "distribution"
        best_action_dist = base.actions[best_idx_dist]
        best_action = best_action_dist
        promo_best = None

        if promo_applied:
            promo_best = promo_dbg.get("best_bucket")
            if isinstance(promo_best, str) and promo_best in base.actions:
                best_action = promo_best
                best_action_source = "promotion"

        # Debug
        debug = None
        if getattr(req, "debug", False):
            def _kept(tokens, m): return [t for i, t in enumerate(tokens) if bool(m[i] > 0.5)]

            kept_final = _kept(base.actions, masks.hero)
            bet_menu_pcts_masked = sorted({
                int(t.split("_", 1)[1]) for t in kept_final
                if t.startswith("BET_") and t.split("_", 1)[1].isdigit()
            }) if side == "root" else None

            debug = {
                "router_side": side,
                "invariants": {
                    "side": side, "ctx": ctx, "actor": actor,
                    "router_mask_sum": float(masks.router.sum().item()),
                    "role_mask_sum": float(masks.role.sum().item()),
                    "size_mask_sum": float(masks.size.sum().item()),
                    "final_mask_sum": float(masks.hero.sum().item()),
                    "kept_by_router": _kept(base.actions, masks.router),
                    "kept_by_role": _kept(base.actions, masks.role),
                    "kept_by_size": _kept(base.actions, masks.size),
                    "kept_final": kept_final,
                    "allow_allin": allow_allin,
                },
                "promotion": promo_dbg,
                "final_T": float(getattr(req, "_final_T", 1.0)),
                "final_eps": float(getattr(req, "_final_eps", 0.0)),
                "pot_bb": float(getattr(req, "pot_bb", 0.0) or 0.0),
                "faced_size_frac": float(getattr(req, "faced_size_frac", -1.0) or -1.0),
                "bet_menu_pcts": bet_menu_pcts_masked,
                "p_win": sig.p_win,
                # Selection trace
                "selection": {
                    "best_action": best_action,
                    "best_action_source": best_action_source,
                    "best_action_distribution": best_action_dist,
                    "promotion_best_bucket": promo_best,
                },
            }

        return PolicyResponse(
            actions=base.actions,
            probs=probs,
            evs=[sig.evs.get(a, 0.0) for a in base.actions],
            notes=[
                f"Postflop policy (router → generator legality → shaping → PromotionGateway[override]); hero_is_ip={hero_is_ip}"],
            debug=debug,
            best_action=best_action,
        )

    @torch.no_grad()
    def _predict_preflop(self, req: PolicyRequest, eq_sig=None) -> PolicyResponse:
        return self._pre.run(req, eq_sig)

    def predict(self, req_input: Union[Dict[str, Any], "PolicyRequest"]) -> "PolicyResponse":
        # -------- Parse input --------
        if isinstance(req_input, dict):
            req = PolicyRequest(**req_input)
        elif isinstance(req_input, PolicyRequest):
            req = req_input
        else:
            raise TypeError(f"PolicyInfer.predict expected dict or PolicyRequest, got {type(req_input)}")

        req.legalize()
        street = 1 if getattr(req, "street", None) is None else int(getattr(req, "street"))

        eq_sig = self._signals.collect_equity(req)

        # -------- Route --------
        if street == 0:
            return self._predict_preflop(req, eq_sig=eq_sig)
        else:
            return self._predict_postflop(req, eq_sig=eq_sig)