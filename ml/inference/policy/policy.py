from typing import Any, Dict, Union, List, Optional
import numpy as np
from ml.inference.action_context_classifier import ActionContextClassifier
from ml.inference.context_infer import ContextInferer
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.helpers import cap_allins, masked_softmax, equity_delta_vector, normalize_bet_sizes, \
    normalize_raise_buckets, update_vocab_cache, derive_side, soft_prior_blend, compute_temperature
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.inference.policy.projection import FCRProjector
from ml.inference.policy.signals import SignalCollector, EquitySig
from ml.inference.policy.types import PolicyRequest, PolicyResponse
import torch
from ml.inference.policy.utils import postflop_is_hero_ip, mix_ties_if_close, \
    epsilon_explore
from ml.inference.preflop_legal_action_generator import PreflopLegalActionGenerator
from ml.inference.promotion.config import PromotionConfig
from ml.inference.promotion.gateway import PromotionGateway


class PolicyInfer:

    def __init__(self, deps: "PolicyInferDeps", blend_cfg: PolicyBlendConfig | None = None):
        self.p: Dict[str, Any] = deps.params or {}
        self.blend = blend_cfg or PolicyBlendConfig.default()

        # Required deps
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
        self.rng_pre = deps.range_pre
        self.clusterer = deps.clusterer
        self.ev_router = deps.ev

        # Optional wiring to router
        if self.clusterer is not None and hasattr(self.pol_post, "set_clusterer"):
            try:
                self.pol_post.set_clusterer(self.clusterer)
            except Exception:
                pass

        # Vocab cache
        self.action_vocab: List[str] = []
        self._vocab_index: Dict[str, int] = {}
        self._P_fcr: Optional[torch.Tensor] = None  # [3,V], lazy
        self._signals = SignalCollector(
            eq_model=deps.equity,
            expl_store=deps.exploit,
            pop_model=deps.pop,
            router=self.pol_post,
            ev_router=self.ev_router
        )
        self._proj = FCRProjector()

        # ⬇️ swap out the old tuner for the PromotionGateway
        self._promoter = PromotionGateway(PromotionConfig(
            w_ev=0.60, w_eq=0.30, w_expl=0.10,
            eq_gate=0.55, expl_gate=0.05, ev_cap_bb=3.0,
            tau_min=0.12, tau_max=0.35,
            max_logit_boost=8.0,
            respect_fold_when_facing=True,
            min_temp=0.6, max_temp=1.2,
        ))
        self._tuner = None  # keep attribute for backward compatibility

        try:
            from ml.models.policy_consts import ACTION_VOCAB as _VOC, VOCAB_INDEX as _VIX  # type: ignore
            self.action_vocab = list(_VOC)
            self._vocab_index = dict(_VIX)
        except Exception:
            pass

        if not hasattr(self.pol_post, "predict"):
            raise TypeError(f"deps.policy_post lacks predict(). Got: {type(self.pol_post)!r}")
        if not hasattr(self.rng_pre, "predict"):
            raise TypeError(f"deps.range_pre lacks predict(). Got: {type(self.rng_pre)!r}")
        if self.eq is not None and not (hasattr(self.eq, "predict_proba") or hasattr(self.eq, "predict")):
            raise TypeError(f"deps.equity lacks predict/predict_proba(). Got: {type(self.eq)!r}")

    @torch.no_grad()
    def _predict_preflop(
            self,
            req: PolicyRequest,
            eq_sig: Optional[EquitySig] = None
    ) -> PolicyResponse:
        # ---- faced size normalization ----
        facing_bet = bool(getattr(req, "facing_bet", False))
        if facing_bet:
            if getattr(req, "faced_size_frac", None) is None and getattr(req, "faced_size_pct", None) is not None:
                try:
                    req.faced_size_frac = float(req.faced_size_pct) / 100.0
                except Exception:
                    pass
            if getattr(req, "faced_size_frac", None) is None:
                req.faced_size_frac = 0.33
        faced_frac = float(getattr(req, "faced_size_frac", 0.0) or 0.0)

        # ---- menu generation ----
        stack_bb = float(getattr(req, "eff_stack_bb", None) or getattr(req, "pot_bb", None) or 100.0)
        hero_pos = (getattr(req, "hero_pos", "") or "").upper()
        free_check = (not facing_bet) and (hero_pos == "BB")
        allow_allin = bool(getattr(req, "allow_allin", False))

        gen = PreflopLegalActionGenerator(
            open_sizes_cbb=(200, 250, 300),
            raise_totals_cbb=(600, 750, 900, 1200),
            allow_allin=allow_allin,
            max_open_cbb=None,
        )
        tokens = gen.generate(
            stack_bb=stack_bb,
            facing_bet=facing_bet,
            faced_frac=faced_frac,
            free_check=free_check,
        )
        if not tokens:
            tokens = ["FOLD"] + (["CHECK"] if free_check else [])

        # ---- EVs aligned to tokens ----
        if not hasattr(self, "ev_router") or self.ev_router is None:
            raise RuntimeError("EV router not attached (self.ev_router is missing). Wire it in PolicyInferFactory.")
        ev_out = self.ev_router.predict(req, side="preflop", tokens=tokens)
        ev_values = np.asarray(ev_out.evs, dtype=np.float32) if (ev_out and ev_out.available) else np.zeros(len(tokens),
                                                                                                            np.float32)
        ev_map = {t: float(ev_values[i]) for i, t in enumerate(tokens)}

        # ---- soft priors → baseline logits (numpy) ----
        base_logits = np.asarray(soft_prior_blend(tokens, req, evs=ev_map, eq_sig=eq_sig), dtype=np.float32)

        # ---- PromotionGateway ----
        promo_dbg = None
        if getattr(self, "_promoter", None) is not None:
            try:
                new_logits, promo_dbg = self._promoter.promote_preflop(
                    base_logits=base_logits,
                    tokens=tokens,
                    evs=ev_map,
                    p_win=(eq_sig.p_win if (eq_sig and eq_sig.available) else None),
                    facing_bet=facing_bet,
                    free_check=free_check,
                    allow_allin=allow_allin,
                )
                base_logits = new_logits.astype(np.float32)
            except Exception as e:
                promo_dbg = {"error": str(e)}

        # ---- temperature → softmax ----
        if base_logits.size:
            base_logits -= float(np.max(base_logits))
        T = float(compute_temperature(ev_values))
        probs = np.exp(base_logits / max(T, 1e-6))
        Z = probs.sum()
        probs = (probs / Z) if (Z > 0 and np.isfinite(Z)) else np.ones(len(tokens), np.float32) / float(len(tokens))
        best_action = tokens[int(np.argmax(probs))]

        debug = None
        if getattr(req, "debug", False):
            debug = {
                "tokens": list(tokens),
                "ev_values": ev_values.astype(float).tolist(),
                "temp": T,
                "facing_bet": facing_bet,
                "faced_frac": faced_frac,
                "free_check": free_check,
                "allow_allin": allow_allin,
                "stack_bb": stack_bb,
                "equity": (eq_sig.p_win if (eq_sig and eq_sig.available) else None),
                "promotion": promo_dbg,
                "ctx": getattr(req, "ctx", None),
            }

        return PolicyResponse(
            actions=list(tokens),
            probs=probs.astype(float).tolist(),
            evs=ev_values.astype(float).tolist(),
            best_action=best_action,
            notes=[f"preflop policy (range+EV router + promoter, T={T:.2f})"],
            debug=debug,
        )

    @torch.no_grad()
    def _predict_postflop(self, req: PolicyRequest, eq_sig: Optional[EquitySig] = None) -> PolicyResponse:
        hero_is_ip = postflop_is_hero_ip(req)
        actor = "ip" if hero_is_ip else "oop"
        side = derive_side(req, hero_is_ip=hero_is_ip, pol_post=self.pol_post)

        ctx, ctx_reason = ContextInferer.infer_with_reason(req)
        if ctx == "BLIND_VS_STEAL":
            ctx = "VS_OPEN"
        req.ctx = ctx

        if side == "facing":
            if getattr(req, "faced_size_frac", None) is None and getattr(req, "faced_size_pct", None) is not None:
                try:
                    req.faced_size_frac = float(req.faced_size_pct) / 100.0
                except Exception:
                    pass
            if getattr(req, "faced_size_frac", None) is None:
                req.faced_size_frac = 0.33

        router_resp = self.pol_post.predict(req, actor=actor, temperature=1.0, side=side)
        actions = list(getattr(router_resp, "actions", []) or [])
        if not actions:
            raise RuntimeError("Router returned empty actions")

        self.action_vocab, self._vocab_index, self._P_fcr = update_vocab_cache(self.action_vocab, actions, self._P_fcr)
        V = len(actions)

        if getattr(router_resp, "logits", None) is not None:
            z = torch.tensor(router_resp.logits, dtype=torch.float32).view(1, V)
        else:
            p_r = torch.tensor(router_resp.probs, dtype=torch.float32).view(1, V)
            z = torch.log(p_r.clamp_min(1e-8))

        # masks
        mask_router = getattr(router_resp, "mask", None)
        hero_mask_router = (torch.tensor(mask_router, dtype=torch.float32).view(1, V)[0]
                            if mask_router is not None else torch.ones(V, dtype=torch.float32))

        role_mask = torch.ones(V, dtype=torch.float32)
        if side == "root":
            for i, t in enumerate(actions):
                if t.startswith("DONK_") and actor == "ip":
                    role_mask[i] = 0.0
        else:
            for i, t in enumerate(actions):
                Ttok = t.upper()
                if not (Ttok in ("FOLD", "CALL") or Ttok.startswith("RAISE_") or Ttok == "ALLIN"):
                    role_mask[i] = 0.0

        allow_allin = True if getattr(req, "allow_allin", None) is None else bool(req.allow_allin)
        size_mask = torch.ones(V, dtype=torch.float32)

        if side == "facing":
            if req.raise_buckets is not None:
                rb = normalize_raise_buckets(req.raise_buckets)
                for i, t in enumerate(actions):
                    if t.startswith("RAISE_"):
                        try:
                            suf = int(t.split("_", 1)[1])
                            if suf not in rb:
                                size_mask[i] = 0.0
                        except Exception:
                            size_mask[i] = 0.0
            for i, t in enumerate(actions):
                if t == "ALLIN" and not allow_allin:
                    size_mask[i] = 0.0
        else:
            if req.bet_sizes is not None:
                bs = normalize_bet_sizes(req.bet_sizes)
                for i, t in enumerate(actions):
                    if t.startswith("BET_"):
                        try:
                            pct = int(t.split("_", 1)[1])
                            if pct not in bs:
                                size_mask[i] = 0.0
                        except Exception:
                            size_mask[i] = 0.0
            for i, t in enumerate(actions):
                if t == "ALLIN" and not allow_allin:
                    size_mask[i] = 0.0

        if side == "facing" and req.raise_buckets is None and req.bet_sizes is None:
            size_mask = torch.ones_like(size_mask)

        hero_mask = (hero_mask_router > 0.5).float() * (role_mask > 0.5).float() * (size_mask > 0.5).float()
        if hero_mask.sum() <= 0:
            hero_mask = (hero_mask_router > 0.5).float()

        # signals
        ex_sig = self._signals.collect_exploit(req)
        if side == "facing":
            facing_info = self._signals.collect_facing(req, hero_is_ip)
            root_info = None
        else:
            root_info = self._signals.collect_root(req, hero_is_ip)
            facing_info = None

        # optional shaping (unchanged)
        z = z.clone()
        if self.blend.lambda_eq > 0.0 and eq_sig and eq_sig.available:
            eq_margin = float(eq_sig.p_win) - 0.5
            d = equity_delta_vector(
                eq_margin=eq_margin,
                hero_is_ip=hero_is_ip,
                facing_bet=(side == "facing"),
                action_vocab=self.action_vocab,
                vocab_index=self._vocab_index,
            ).to(dtype=z.dtype, device=z.device)
            d = torch.clamp(d, -float(self.blend.eq_max_logit_delta), float(self.blend.eq_max_logit_delta))
            z = z + float(self.blend.lambda_eq) * d

        if ex_sig.available and self.blend.lambda_expl > 0.0 and ex_sig.raw is not None:
            delta = self._proj.lift(ex_sig.raw, self.action_vocab, z.dtype, z.device)
            z = z + float(self.blend.lambda_expl) * delta

        ev_sig = self._signals.collect_ev(req, tokens=actions, side=side)
        evs = dict(ev_sig.evs) if (ev_sig and ev_sig.available) else {a: 0.0 for a in actions}
        for k, v in list(evs.items()):
            if not (v == v) or v in (float("inf"), float("-inf")):
                evs[k] = 0.0

        action_ctx = ActionContextClassifier.from_request(req, side)

        root_bet_menu_pcts = None
        if root_info is not None:
            menu = getattr(root_info, "bet_menu", None)
            if menu:
                root_bet_menu_pcts = []
                for item in menu:
                    if isinstance(item, (int, float)):
                        root_bet_menu_pcts.append(int(round(item)))
                    elif isinstance(item, str) and item.upper().startswith("BET_"):
                        try:
                            root_bet_menu_pcts.append(int(item.split("_", 1)[1]))
                        except Exception:
                            pass
                if not root_bet_menu_pcts:
                    root_bet_menu_pcts = None

        # ---- PromotionGateway (patched signature) ----
        promo_dbg = None
        if getattr(self, "_promoter", None) is not None:
            try:
                spr = None
                pot = float(getattr(req, "pot_bb", 0.0) or 0.0)
                eff = float(getattr(req, "eff_stack_bb", 0.0) or 0.0)
                if pot > 0:
                    spr = eff / max(pot, 1e-9)

                out = self._promoter.promote_postflop(
                    tokens=actions,
                    base_logits=z,
                    hero_mask=hero_mask,
                    side=side,
                    actor=actor,
                    p_win=(eq_sig.p_win if (eq_sig and eq_sig.available) else None),
                    ex_probs=(list(ex_sig.probs) if ex_sig and ex_sig.probs else None),
                    evs=evs,
                    size_frac=(facing_info.size_frac if facing_info else getattr(req, "faced_size_frac",
                                                                                 None)) if side == "facing" else None,
                    bet_menu_pcts=root_bet_menu_pcts,
                    ctx=ctx,
                    spr=spr,
                    allow_allin=allow_allin,
                    action_ctx=action_ctx,
                )
                if isinstance(out, dict) and "logits" in out:
                    z = out["logits"]
                    promo_dbg = out.get("debug")
            except Exception as e:
                promo_dbg = {"error": str(e)}

        # ---- final distribution (unchanged) ----
        p = masked_softmax(
            z, hero_mask.view(1, -1),
            T=float(self.blend.temperature),
            eps=float(self.blend.min_legal_prob),
        )
        p = mix_ties_if_close(p, float(self.blend.tie_mix_threshold))
        p = epsilon_explore(p, float(self.blend.epsilon_explore), hero_mask.view(1, -1))
        p = cap_allins(
            p,
            eff_stack_bb=float(getattr(req, "eff_stack_bb", 0.0) or 0.0),
            action_vocab=self.action_vocab,
            max_allin_freq=float(self.blend.max_allin_freq),
            risk_floor_stack_bb=float(self.blend.risk_floor_stack_bb),
        )
        probs = p[0].tolist()
        best_idx = int(torch.argmax(p[0]))
        best_action = actions[best_idx]

        if req.debug:
            def _where(tokens, pred):
                return [t for i, t in enumerate(tokens) if bool(pred[i])]

            is_raise = torch.tensor([1.0 if t.startswith("RAISE_") else 0.0 for t in actions])
            is_bet = torch.tensor([1.0 if t.startswith("BET_") else 0.0 for t in actions])

            inv = {
                "side": side,
                "ctx": ctx,
                "ctx_reason": ctx_reason,
                "had_raises": bool((is_raise.sum().item()) > 0),
                "kept_raises_router": _where(actions, (is_raise > 0) * (hero_mask_router > 0.5)),
                "kept_raises_role": _where(actions, (is_raise > 0) * (role_mask > 0.5)),
                "kept_raises_size": _where(actions, (is_raise > 0) * (size_mask > 0.5)),
                "final_kept_raises": _where(actions, (is_raise > 0) * (hero_mask > 0.5)),
                "had_bets": bool((is_bet.sum().item()) > 0),
                "final_kept_bets": _where(actions, (is_bet > 0) * (hero_mask > 0.5)),
                "allow_allin": allow_allin,
                "menu_mask_sum": float(hero_mask.sum().item()),
                "router_mask_sum": float((hero_mask_router > 0.5).sum().item()),
                "role_mask_sum": float((role_mask > 0.5).sum().item()),
                "size_mask_sum": float((size_mask > 0.5).sum().item()),
            }
            debug = {
                "router_side": side,
                "invariants": inv,
                "promotion": promo_dbg,
                "pot_bb": float(getattr(req, "pot_bb", 0.0) or 0.0),
                "faced_size_frac": float(getattr(req, "faced_size_frac", -1.0) or -1.0),
            }
        else:
            debug = None

        return PolicyResponse(
            actions=actions,
            probs=probs,
            evs=[evs.get(a, 0.0) for a in actions],
            notes=[f"Postflop policy (GTO baseline + PromotionGateway); hero_is_ip={hero_is_ip}"],
            debug=debug,
            best_action=best_action,
        )

    def predict(self, req_input: Union[Dict[str, Any], "PolicyRequest"]) -> "PolicyResponse":
        # -------- Parse input --------
        if isinstance(req_input, dict):
            req = PolicyRequest(**req_input)
        elif isinstance(req_input, PolicyRequest):
            req = req_input
        else:
            raise TypeError(f"PolicyInfer.predict expected dict or PolicyRequest, got {type(req_input)}")

        req.legalize()

        _s = getattr(req, "street", None)
        street = 1 if _s is None else int(_s)

        eq_sig = self._signals.collect_equity(req)

        # -------- Route to street-specific policy --------
        if street == 0:
            return self._predict_preflop(req, eq_sig=eq_sig)
        else:
            return self._predict_postflop(req, eq_sig=eq_sig)