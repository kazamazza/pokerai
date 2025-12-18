from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from ml.inference.context_infer import ContextInferer
from ml.inference.policy.engines.types import Baseline, Masks, SignalsPack
from ml.inference.policy.helpers import update_vocab_cache, normalize_bet_sizes, equity_delta_vector, masked_softmax, \
    cap_allins
from ml.inference.policy.projection import FCRProjector
from ml.inference.policy.signals import SignalCollector
from ml.inference.policy.utils import mix_ties_if_close, epsilon_explore
from ml.inference.postflop_legal_action_generator import PostflopLegalActionGenerator
from ml.inference.promotion.gateway import PromotionGateway

class PostflopContextResolver:
    """Derives (hero_is_ip, side, ctx) with the same logic your models expect."""
    def __init__(self, pol_router):  # PostflopPolicyRouter
        self.router = pol_router

    @staticmethod
    def _hero_is_ip(hpos: str, vpos: str, street: int) -> bool:
        h = (hpos or "").upper(); v = (vpos or "").upper()
        if street == 0:  # preflop
            return False
        if h == "BTN" and v in ("SB", "BB"): return True
        if {h, v} == {"SB", "BB"}: return h == "BB"
        order = ["SB","BB","UTG","HJ","CO","BTN"]
        try: return order.index(h) > order.index(v)
        except ValueError: return True

    def derive(self, req):
        street = int(getattr(req, "street", 1) or 1)
        hero_is_ip = self._hero_is_ip(getattr(req, "hero_pos",""), getattr(req,"villain_pos",""), street)

        # side: prefer explicit flag, else infer via facing helper from the router
        if bool(getattr(req, "facing_bet", False)):
            side = "facing"
            size_frac = getattr(req, "faced_size_frac", None)
            if size_frac is None and getattr(req, "faced_size_pct", None) is not None:
                try: size_frac = float(req.faced_size_pct)/100.0
                except Exception: size_frac = None
            # default 1/3 if totally missing
            if size_frac is None: size_frac = 0.33
            req.faced_size = size_frac
        else:
            # ask the trained helper; falls back to history if flag is missing
            facing_flag, size_frac = self.router.infer_facing_and_size(req, hero_is_ip=hero_is_ip)
            side = "facing" if facing_flag else "root"
            if side == "facing" and size_frac is not None:
                req.faced_size = float(size_frac)

        # ctx: use your ContextInferer (action_seq / actions_hist / explicit / default)
        ctx, ctx_dbg = ContextInferer.infer_with_reason(req)
        req.ctx = ctx  # make it available to baseline & EV nets

        return hero_is_ip, side, ctx, ctx_dbg

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

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _coerce_pwin(x):
        try:
            v = float(x)
            # tolerate percentages 0..100 by auto-scaling
            if v > 1.0 and v <= 100.0:
                v = v / 100.0
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
        return None

    def _extract_pwin(self, eq_sig) -> tuple[float|None, str]:
        """Try multiple shapes/field names; return (p_win, source_tag)."""
        if eq_sig is None:
            return None, "none"
        # dict-like
        if isinstance(eq_sig, dict):
            for k in ("p_win","equity","win","p","p_hero","p_hero_win","value"):
                if k in eq_sig:
                    v = self._coerce_pwin(eq_sig[k])
                    if v is not None: return v, f"eq_sig.{k}"
            # respect 'available' if explicitly false
            if eq_sig.get("available") is False:
                return None, "eq_sig.unavailable"
        # attr-like
        for k in ("p_win","equity","win","p","p_hero","p_hero_win","value"):
            if hasattr(eq_sig, k):
                v = self._coerce_pwin(getattr(eq_sig, k))
                if v is not None: return v, f"eq_sig.{k}"
        # final: if it only has .available and it's False, say so
        if hasattr(eq_sig, "available") and not bool(getattr(eq_sig, "available")):
            return None, "eq_sig.unavailable"
        return None, "eq_sig.unknown_shape"

    # ------------------------------------------------------------------------

    def collect(self, req, base: Baseline, *, side: str, actor: str, hero_is_ip: bool, ctx: str, eq_sig) -> SignalsPack:
        # 1) equity
        p_win, eq_src = self._extract_pwin(eq_sig)

        if p_win is None:
            # try primary equity service via SignalCollector
            try:
                eq_sig2 = self._signals.collect_equity(req)
            except Exception:
                eq_sig2 = None
            p_win, eq_src2 = self._extract_pwin(eq_sig2)
            if p_win is None:
                eq_src = f"{eq_src}|collector:{eq_src2}"
            else:
                eq_src = f"collector:{eq_src2}"

        # 2) exploit
        ex_sig = self._signals.collect_exploit(req)
        ex_probs = tuple(ex_sig.probs) if (ex_sig and getattr(ex_sig, "probs", None)) else None

        # 3) sizing/menu (mirror router)
        if side == "facing":
            facing_info = self._signals.collect_facing(req, hero_is_ip)
            size_frac = (facing_info.size_frac if facing_info else getattr(req, "faced_size_frac", None))
            # harden size_frac if nonsense sneaks in
            if size_frac is not None:
                try:
                    size_frac = float(min(max(size_frac, 0.05), 1.5))
                except Exception:
                    size_frac = None
            bet_menu_pcts = None
        else:
            root_info = self._signals.collect_root(req, hero_is_ip)
            size_frac = None
            if root_info and getattr(root_info, "bet_menu", None):
                bet_menu_pcts = list(normalize_bet_sizes(root_info.bet_menu))
            else:
                kept_bets = [
                    int(a.split("_", 1)[1]) for a in base.actions
                    if a.upper().startswith("BET_") and a.split("_", 1)[1].isdigit()
                ]
                bet_menu_pcts = (
                    sorted(set(kept_bets))
                    or (list(normalize_bet_sizes(getattr(req, "bet_sizes", None))) if getattr(req, "bet_sizes", None) else [33, 66])
                )

        # 4) EVs (ensure ctx is on req so EV nets see same categorical)
        req.ctx = ctx
        ev_sig = self._signals.collect_ev(req, tokens=base.actions, side=side)
        evs = dict(ev_sig.evs) if (ev_sig and getattr(ev_sig, "available", False)) else {a: 0.0 for a in base.actions}
        # safety: non-finite to 0; FOLD to 0
        for k, v in list(evs.items()):
            if not (v == v) or v in (float("inf"), float("-inf")):
                evs[k] = 0.0
        if "FOLD" in evs:
            evs["FOLD"] = 0.0

        pot = float(getattr(req, "pot_bb", 0.0) or 0.0)
        eff = float(getattr(req, "eff_stack_bb", 0.0) or 0.0)
        spr = (eff / max(pot, 1e-9)) if pot > 0 else None

        pack = SignalsPack(
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
        # if your SignalsPack has a debug slot, you can annotate source:
        try:
            pack.debug = {"equity_src": eq_src}
        except Exception:
            pass
        return pack

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

import torch

import torch

class PromotionApplier:
    def __init__(self, promoter, shaper=None):
        self.promoter = promoter
        self.shaper = shaper

    def _to_tensor_1x(self, x, like: torch.Tensor | None = None):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.tensor(x, dtype=(like.dtype if like is not None else torch.float32),
                             device=(like.device if like is not None else None))
        if t.dim() == 1:
            t = t.view(1, -1)
        return t

    def _to_vector(self, x, like: torch.Tensor | None = None):
        """Ensure a 1-D vector (V,) on same device/dtype as `like`."""
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.tensor(x, dtype=(like.dtype if like is not None else torch.float32),
                             device=(like.device if like is not None else None))
        if t.dim() == 2 and t.shape[0] == 1:
            t = t.view(-1)
        return t

    def postflop(self, base, masks, sig, *, allow_allin: bool):
        actions = list(base.actions)

        # Build bet-menu actually legal after role mask
        kept = [t for i, t in enumerate(actions) if float(masks.hero[i]) > 0.5]
        legal_bets = sorted({
            int(t.split("_", 1)[1]) for t in kept
            if t.startswith("BET_") and t.split("_", 1)[1].isdigit()
        }) or None

        # Baseline logits [1,V]
        z = self._to_tensor_1x(base.logits)

        # ✅ Keep hero_mask as VECTOR [V] for the gateway
        hero_mask_vec = self._to_vector(masks.hero, like=z)
        hero_mask_1x = hero_mask_vec.view(1, -1)

        big_neg = torch.finfo(z.dtype).min / 4
        z_legal = torch.where(hero_mask_1x > 0.5, z, big_neg)

        # Optional shaping
        shaped_debug = {"applied": False}
        if self.shaper is not None:
            try:
                vix = {a: i for i, a in enumerate(actions)}
                self.shaper.bind_vocab(actions, vix)

                p_pre = torch.softmax(z_legal, dim=-1)
                z_shaped = self.shaper.apply(z_legal, sig)
                # re-mask to ensure no illegal resurrection
                z_shaped = torch.where(hero_mask_1x > 0.5, z_shaped, big_neg)
                p_post = torch.softmax(z_shaped, dim=-1)

                shaped_debug = {
                    "applied": True,
                    "pwin": (float(sig.p_win) if getattr(sig, "p_win", None) is not None else None),
                    "exploit_on": bool(getattr(sig, "ex_probs", None) is not None),
                    "delta_L1": float((p_post - p_pre).abs().sum().item()),
                }
                z_for_promo = z_shaped
            except Exception as e:
                shaped_debug = {"applied": False, "error": str(e)[:160]}
                z_for_promo = z_legal
        else:
            z_for_promo = z_legal

        # ✅ Pass the VECTOR hero_mask to the gateway (it expects [V])
        out = self.promoter.promote_postflop(
            tokens=actions,
            base_logits=z_for_promo,
            hero_mask=hero_mask_vec,           # <-- vector
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
            dbg["bet_menu_pcts_used"] = legal_bets
            dbg["shaping"] = shaped_debug
            try:
                eq_src = getattr(sig, "debug", {}).get("equity_src")
                if eq_src:
                    dbg.setdefault("signals", {})["equity_src"] = eq_src
            except Exception:
                pass
            out["debug"] = dbg
            return out["logits"], out["debug"]

        return z_for_promo, {"shaping": shaped_debug, "bet_menu_pcts_used": legal_bets}

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