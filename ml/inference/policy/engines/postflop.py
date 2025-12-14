from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from ml.inference.policy.engines.types import Baseline, Masks, SignalsPack
from ml.inference.policy.helpers import update_vocab_cache, normalize_bet_sizes, equity_delta_vector, masked_softmax, \
    cap_allins
from ml.inference.policy.projection import FCRProjector
from ml.inference.policy.signals import SignalCollector
from ml.inference.policy.utils import mix_ties_if_close, epsilon_explore
from ml.inference.postflop_legal_action_generator import PostflopLegalActionGenerator
from ml.inference.promotion.gateway import PromotionGateway


@dataclass
class ActorContextDeriver:
    pol_post: Any
    @staticmethod
    def postflop_is_hero_ip(req) -> bool:
        # Expect your real util; fallback: IP if hero_pos in {BTN, CO}
        ip_pos = {"BTN","CO","HJ","IP"}
        return str(getattr(req, "hero_pos", "")).upper() in ip_pos
    def derive(self, req) -> Tuple[bool, str, str]:
        hero_is_ip = self.postflop_is_hero_ip(req)
        # Expect your real derive_side + ContextInferer; fallback heuristics
        side = "root" if not getattr(req, "facing_bet", False) else "facing"
        ctx = getattr(req, "ctx", None) or ("VS_OPEN" if getattr(req, "preflop_ctx", "") == "BLIND_VS_STEAL" else "GEN")
        req.ctx = ctx
        return hero_is_ip, side, ctx
    @staticmethod
    def normalize_faced_size_if_needed(req, side: str) -> None:
        if side != "facing": return
        if getattr(req, "faced_size_frac", None) is None and getattr(req, "faced_size_pct", None) is not None:
            try: req.faced_size_frac = float(req.faced_size_pct) / 100.0
            except Exception: ...
        if getattr(req, "faced_size_frac", None) is None:
            req.faced_size_frac = 0.33

class PostflopBaselineProvider:
    def __init__(self, pol_post: Any, action_vocab_ref: List[str], vocab_index_ref: Dict[str,int]):
        self.pol_post = pol_post
        self.action_vocab_ref = action_vocab_ref
        self.vocab_index_ref = vocab_index_ref
        self._P_fcr = None

    def get(self, req, *, actor: str, side: str) -> Baseline:
        router_resp = self.pol_post.predict(req, actor=actor, temperature=1.0, side=side)
        actions = list(getattr(router_resp, "actions", []) or [])
        if not actions:
            raise RuntimeError("Router returned empty actions")

        # ✅ unpack to temps, then update in-place (can’t assign to a function call)
        new_vocab, new_vix, new_P_fcr = update_vocab_cache(self.action_vocab_ref, actions, self._P_fcr)
        self.action_vocab_ref[:] = new_vocab  # keep list object identity
        self.vocab_index_ref.clear()
        self.vocab_index_ref.update(new_vix)
        self._P_fcr = new_P_fcr

        V = len(actions)
        if getattr(router_resp, "logits", None) is not None:
            z = torch.tensor(router_resp.logits, dtype=torch.float32).view(1, V)
        else:
            p_r = torch.tensor(router_resp.probs, dtype=torch.float32).view(1, V)
            z = torch.log(p_r.clamp_min(1e-8))

        mask_router = getattr(router_resp, "mask", None)
        hero_mask_router = (
            torch.tensor(mask_router, dtype=torch.float32).view(1, V)[0]
            if mask_router is not None else torch.ones(V, dtype=torch.float32)
        )
        return Baseline(actions=actions, logits=z, mask_router=hero_mask_router)

class PostflopMaskBuilder:
    def __init__(self): ...
    def build(self, req, base: Baseline, *, side: str) -> Masks:
        gen = PostflopLegalActionGenerator(
            bet_menu_pcts=normalize_bet_sizes(req.bet_sizes) if req.bet_sizes is not None else (33, 66),
            raise_pcts=(150, 200, 250, 300, 400, 500),
            allow_allin=True if getattr(req, "allow_allin", None) is None else bool(req.allow_allin),
        )
        gen_list = gen.generate(side=side, bet_sizes=req.bet_sizes, raise_pcts=None, allow_allin=req.allow_allin)
        gen_tokens = set(gen_list)
        actions = base.actions
        V = len(actions)
        role_mask = torch.ones(V, dtype=torch.float32)
        for i, t in enumerate(actions):
            if t.upper() not in gen_tokens:
                role_mask[i] = 0.0
        size_mask = torch.ones(V, dtype=torch.float32)
        hero_mask = (base.mask_router > 0.5).float() * (role_mask > 0.5).float() * (size_mask > 0.5).float()
        if hero_mask.sum() <= 0:
            hero_mask = (base.mask_router > 0.5).float()
        return Masks(
            router=(base.mask_router > 0.5).float(),
            role=(role_mask > 0.5).float(),
            size=(size_mask > 0.5).float(),
            hero=hero_mask,
        )

class SignalsBundler:
    def __init__(self, signals: SignalCollector, action_vocab_ref: List[str], vocab_index_ref: Dict[str,int]):
        self._signals = signals
        self.action_vocab_ref = action_vocab_ref
        self.vocab_index_ref = vocab_index_ref
    def collect(self, req, base: Baseline, *, side: str, actor: str, hero_is_ip: bool, ctx: str, eq_sig) -> SignalsPack:
        p_win = float(eq_sig.p_win) if (eq_sig and getattr(eq_sig, "available", False)) else None
        ex_sig = self._signals.collect_exploit(req)
        ex_probs = tuple(ex_sig.probs) if (ex_sig and getattr(ex_sig, "probs", None)) else None

        if side == "facing":
            facing_info = self._signals.collect_facing(req, hero_is_ip)
            size_frac = (facing_info.size_frac if facing_info else getattr(req, "faced_size_frac", None))
            bet_menu_pcts = None
        else:
            root_info = self._signals.collect_root(req, hero_is_ip)
            size_frac = None
            kept_bets = [
                int(a.split("_", 1)[1])
                for a in base.actions
                if a.upper().startswith("BET_") and "_" in a and a.split("_", 1)[1].isdigit()
            ]
            bet_menu_pcts = sorted(set(kept_bets)) or (
                list(normalize_bet_sizes(req.bet_sizes)) if req.bet_sizes else [33, 66])

        ev_sig = self._signals.collect_ev(req, tokens=base.actions, side=side)
        evs = dict(ev_sig.evs) if (ev_sig and getattr(ev_sig, "available", False)) else {a: 0.0 for a in base.actions}
        for k, v in list(evs.items()):
            if not (v == v) or v in (float("inf"), float("-inf")):
                evs[k] = 0.0

        pot = float(getattr(req, "pot_bb", 0.0) or 0.0)
        eff = float(getattr(req, "eff_stack_bb", 0.0) or 0.0)
        spr = (eff / max(pot, 1e-9)) if pot > 0 else None

        return SignalsPack(
            p_win=p_win,
            ex_probs=ex_probs,
            evs=evs,
            size_frac=size_frac,
            bet_menu_pcts=(list(bet_menu_pcts) if bet_menu_pcts is not None else None),
            spr=spr,
            ctx=ctx,
            side=side,
            actor=actor,
            hero_is_ip=hero_is_ip,
        )

class LogitShaper:
    def __init__(self, blend_cfg, projector: Optional[FCRProjector] = None):
        self.blend = blend_cfg
        self._proj = projector or FCRProjector()
        self._vocab_ref = None
        self._vix_ref = None
    def bind_vocab(self, action_vocab_ref: List[str], vix_ref: Dict[str,int]):
        self._vocab_ref = action_vocab_ref
        self._vix_ref = vix_ref
    def apply(self, z: torch.Tensor, sig: SignalsPack) -> torch.Tensor:
        out = z.clone()
        if self.blend.lambda_eq > 0.0 and sig.p_win is not None:
            eq_margin = float(sig.p_win) - 0.5
            d = equity_delta_vector(
                eq_margin=eq_margin,
                hero_is_ip=sig.hero_is_ip,
                facing_bet=(sig.side == "facing"),
                action_vocab=self._vocab_ref or [],
                vocab_index=self._vix_ref or {},
            ).to(dtype=out.dtype, device=out.device)
            d = torch.clamp(d, -float(self.blend.eq_max_logit_delta), float(self.blend.eq_max_logit_delta))
            out = out + float(self.blend.lambda_eq) * d
        if self.blend.lambda_expl > 0.0 and sig.ex_probs is not None:
            ex_raw = torch.tensor(sig.ex_probs, dtype=out.dtype).view(1, -1)
            delta = self._proj.lift(ex_raw[0].cpu().numpy(), self._vocab_ref or [], out.dtype, out.device)
            out = out + float(self.blend.lambda_expl) * delta
        return out

class PromotionApplier:
    def __init__(self, promoter: PromotionGateway): self.promoter = promoter
    def postflop(self, base: Baseline, masks: Masks, sig: SignalsPack, *, allow_allin: bool):
        kept = [t for i, t in enumerate(base.actions) if float(masks.hero[i]) > 0.5]
        legal_bets = sorted({
            int(t.split("_", 1)[1]) for t in kept
            if t.startswith("BET_") and t.split("_", 1)[1].isdigit()
        }) or None

        out = self.promoter.promote_postflop(
            tokens=base.actions,
            base_logits=base.logits,
            hero_mask=masks.hero,
            side=sig.side,
            actor=sig.actor,
            p_win=sig.p_win,
            ex_probs=sig.ex_probs,
            evs=sig.evs,
            size_frac=(sig.size_frac if sig.side == "facing" else None),
            bet_menu_pcts=legal_bets,
            ctx=sig.ctx,
            spr=sig.spr,
            allow_allin=allow_allin,
        )
        if isinstance(out, dict) and "logits" in out:
            dbg = out.get("debug") or {}
            dbg["bet_menu_pcts_used"] = legal_bets  # ✅ echo masked sizes actually used
            out["debug"] = dbg
            return out["logits"], out["debug"]
        return base.logits, None

    def preflop(self, **kw):
        return self.promoter.promote_preflop(**kw)

# DistributionBuilder
class DistributionBuilder:
    def __init__(self, blend_cfg, action_vocab_ref: List[str]):
        self.blend = blend_cfg
        self.action_vocab_ref = action_vocab_ref

    def build(self, z: torch.Tensor, masks: Masks, req, *, promo_applied: bool = False):
        T = 1.0 if promo_applied else float(self.blend.temperature)     # why: avoid washing out promo
        eps = 0.0 if promo_applied else float(self.blend.epsilon_explore)

        p = masked_softmax(z, masks.hero.view(1, -1), T=T, eps=float(self.blend.min_legal_prob))
        p = mix_ties_if_close(p, float(self.blend.tie_mix_threshold if not promo_applied else 0.0))
        p = epsilon_explore(p, eps, masks.hero.view(1, -1))
        p = cap_allins(
            p,
            eff_stack_bb=float(getattr(req, "eff_stack_bb", 0.0) or 0.0),
            action_vocab=self.action_vocab_ref,
            max_allin_freq=float(self.blend.max_allin_freq),
            risk_floor_stack_bb=float(self.blend.risk_floor_stack_bb),
        )

        # attach for upstream debug (so you can see it)
        setattr(req, "_final_T", T)
        setattr(req, "_final_eps", eps)

        probs = p[0].tolist()
        best_idx = int(torch.argmax(p[0]))
        return probs, best_idx