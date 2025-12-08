import math
from typing import Any, Dict, Union, List, Optional
import numpy as np
from ml.inference.action_context_classifier import ActionContextClassifier
from ml.inference.ev_calculator import EVCalculator
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy_blend_config import PolicyBlendConfig, tuner_knobs_from_blend
from ml.inference.policy.projection import FCRProjector
from ml.inference.policy.signals import SignalCollector, EquitySig
from ml.inference.policy.tuner import PostflopTuner
from ml.inference.policy.types import PolicyRequest, PolicyResponse
import torch
from ml.inference.policy.utils import postflop_is_hero_ip, mix_ties_if_close, \
    epsilon_explore
import torch.nn.functional as F

from ml.inference.postflop_ctx import ensure_ctx_and_action_seq
from ml.inference.preflop_legal_action_generator import PreflopLegalActionGenerator
from ml.inference.villain_range_provider import VillainRangeProvider


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
            router=self.pol_post,  # pass the whole router; collector will use .facing if present
        )
        self._proj = FCRProjector()
        self._tuner = PostflopTuner(tuner_knobs_from_blend(self.blend))

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

    def _derive_side(self, req: "PolicyRequest", *, hero_is_ip: bool) -> str:
        """Return 'root' or 'facing' deterministically from the payload/history."""
        fb = getattr(req, "facing_bet", None)
        if isinstance(fb, bool):
            return "facing" if fb else "root"
        # explicit faced size implies facing
        if getattr(req, "faced_size_frac", None) is not None or getattr(req, "faced_size_pct", None) is not None:
            return "facing"
        # reuse facing single’s parser if available
        try:
            facing_flag, _ = self.pol_post.facing.infer_facing_and_size(req, hero_is_ip=hero_is_ip)
            return "facing" if facing_flag else "root"
        except Exception:
            return "root"

    def _update_vocab(self, actions: List[str]) -> None:
        """Rebuild index and invalidate F/C/R projection if actions changed."""
        if actions == self.action_vocab:
            return
        self.action_vocab = list(actions)
        self._vocab_index = {a: i for i, a in enumerate(self.action_vocab)}
        self._P_fcr = None  # invalidate

    def _equity_delta_vector(self, *, eq_margin: float, hero_is_ip: bool, facing_bet: bool) -> torch.Tensor:
        """Map equity margin into a [V]-logit delta; gentle shaping."""
        V = len(self.action_vocab)
        delta = torch.zeros(V, dtype=torch.float32)
        m = max(-0.25, min(0.25, float(eq_margin)))
        base = m

        def bump(tok: str, amt: float):
            j = self._vocab_index.get(tok)
            if j is not None:
                delta[j] += amt

        if facing_bet:
            bump("CALL", +1.00 * base)
            for tok in ("RAISE_150", "RAISE_200", "RAISE_300"):
                bump(tok, +0.50 * base)
            bump("FOLD", -1.25 * base)
        else:
            for tok in ("BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100"):
                bump(tok, +0.80 * base)
            if not hero_is_ip:
                bump("DONK_33", +0.80 * base)
            bump("CHECK", -1.00 * base)
        return delta.view(1, -1)

    def _cap_allins(self, p: torch.Tensor, eff_stack_bb: float) -> torch.Tensor:
        """Cap ALLIN mass when deep; preserves distribution over legal others."""
        if eff_stack_bb <= self.blend.risk_floor_stack_bb:
            return p
        try:
            idx = self.action_vocab.index("ALLIN")
        except ValueError:
            return p
        p = p if p.dim() == 2 else p.view(1, -1)
        if p[0, idx] <= self.blend.max_allin_freq:
            return p
        over = p[0, idx] - self.blend.max_allin_freq
        p = p.clone()
        p[0, idx] = self.blend.max_allin_freq
        rest = [i for i in range(len(self.action_vocab)) if i != idx]
        denom = float(p[0, rest].sum().item()) or 1e-12
        scale = 1.0 + (over / denom)
        p[0, rest] *= scale
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return p

    def _as_batch(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.dim() == 2 else x.view(1, -1)

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor, T: float, eps: float) -> torch.Tensor:
        logits = self._as_batch(logits)
        mask = self._as_batch(mask).to(logits.dtype)
        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits / max(T, 1e-6), big_neg)
        p = F.softmax(masked, dim=-1)
        p = p * (1 - eps) + (eps / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)) * (mask > 0.5).to(p.dtype)
        p = p * mask
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return p

    def _menu_mask(self, actions: List[str], *, actor: str, facing_bet: bool,
                  bet_sizes: Optional[List[float]],
                  raise_buckets: Optional[List[int]],
                  allow_allin: Optional[bool]) -> torch.Tensor:
        """Menu-aware legalities aligned to actions."""
        allow_ai = True if allow_allin is None else bool(allow_allin)
        m = torch.zeros(len(actions), dtype=torch.float32)
        raise_set = set(int(x) for x in (raise_buckets or []))
        bet_percents = None
        if bet_sizes is not None:
            bet_percents = set(
                int(round((s if s > 1 else (s * 100.0) + 1e-9)))
                for s in bet_sizes
            )

        for i, tok in enumerate(actions):
            T = tok.upper()
            legal = False

            if not facing_bet:
                # ROOT
                if T == "CHECK":
                    legal = True
                elif T.startswith("BET_"):
                    if bet_percents is None:
                        legal = True
                    else:
                        try:
                            pct = int(T.split("_", 1)[1])
                            legal = pct in bet_percents
                        except Exception:
                            legal = False
                elif T.startswith("DONK_"):
                    legal = (actor == "oop")
                elif T == "ALLIN":
                    legal = allow_ai
            else:
                if T in ("FOLD", "CALL"):
                    legal = True
                elif T.startswith("RAISE_"):
                    if not raise_set:
                        legal = True
                    else:
                        try:
                            mult = int(T.split("_", 1)[1])
                            legal = mult in raise_set
                        except Exception:
                            legal = False
                elif T == "ALLIN":
                    legal = allow_ai

            if legal:
                m[i] = 1.0

        if m.sum().item() == 0:
            m.fill_(1.0)

        return m

    @torch.no_grad()
    def _predict_postflop(self, req: PolicyRequest, eq_sig: Optional[EquitySig] = None,
                          villain_range: Optional[np.ndarray] = None) -> PolicyResponse:
        hero_is_ip = postflop_is_hero_ip(req)
        actor = "ip" if hero_is_ip else "oop"
        side = self._derive_side(req, hero_is_ip=hero_is_ip)

        # ---------- Router model ----------
        router_resp = self.pol_post.predict(req, actor=actor, temperature=1.0, side=side)
        actions = list(getattr(router_resp, "actions", []) or [])
        if not actions:
            raise RuntimeError("Router returned empty actions")
        self._update_vocab(actions)
        V = len(actions)

        # ---------- Logits ----------
        if getattr(router_resp, "logits", None) is not None:
            z = torch.tensor(router_resp.logits, dtype=torch.float32).view(1, V)
        else:
            p_r = torch.tensor(router_resp.probs, dtype=torch.float32).view(1, V)
            z = torch.log(p_r.clamp_min(1e-8))

        # ---------- Legal mask ----------
        mask_router = getattr(router_resp, "mask", None)
        hero_mask_router = (
            torch.tensor(mask_router, dtype=torch.float32).view(1, V)[0]
            if mask_router is not None else torch.ones(V, dtype=torch.float32)
        )
        menu_mask_t = self._menu_mask(
            actions=actions,
            actor=actor,
            facing_bet=(side == "facing"),
            bet_sizes=req.bet_sizes,
            raise_buckets=req.raise_buckets,
            allow_allin=req.allow_allin,
        )
        hero_mask = (hero_mask_router > 0.5).float() * (menu_mask_t > 0.5).float()
        if hero_mask.sum() <= 0:
            hero_mask = torch.ones_like(menu_mask_t)

        # ---------- Signals ----------
        ex_sig = self._signals.collect_exploit(req)
        facing_info = root_info = None
        if side == "facing":
            facing_info = self._signals.collect_facing(req, hero_is_ip)
        elif side == "root":
            root_info = self._signals.collect_root(req, hero_is_ip)

        # ---------- Equity adjustment ----------
        z = z.clone()
        if self.blend.lambda_eq > 0.0 and eq_sig and eq_sig.available:
            eq_margin = float(eq_sig.p_win) - 0.5
            d = self._equity_delta_vector(
                eq_margin=eq_margin,
                hero_is_ip=hero_is_ip,
                facing_bet=(side == "facing"),
            ).to(dtype=z.dtype, device=z.device)
            d = torch.clamp(d, -float(self.blend.eq_max_logit_delta), float(self.blend.eq_max_logit_delta))
            z = z + float(self.blend.lambda_eq) * d

        # ---------- Exploit adjustment ----------
        if ex_sig.available and self.blend.lambda_expl > 0.0 and ex_sig.raw is not None:
            delta = self._proj.lift(ex_sig.raw, self.action_vocab, z.dtype, z.device)
            z = z + float(self.blend.lambda_expl) * delta

        # ---------- EV Calculation ----------
        ctx = ActionContextClassifier.from_request(req, side=side)
        ev_calc = EVCalculator()
        ev_sig = ev_calc.compute(req, tokens=actions, villain_range=villain_range)

        evs = ev_sig.evs if ev_sig and ev_sig.available else {}
        best_ev = ev_sig.best_ev if ev_sig and ev_sig.available else None
        print("evs:", evs)

        # ---------- Tuner ----------
        if side == "facing":
            tuner_dbg = self._tuner.apply_facing_raise(
                z=z,
                actions=actions,
                hero_mask=hero_mask,
                p_win=(eq_sig.p_win if eq_sig and eq_sig.available else None),
                ex_probs=(list(ex_sig.probs) if ex_sig and ex_sig.probs else None),
                size_frac=(facing_info.size_frac if facing_info else None),
                evs=evs,
                action_ctx=ctx,
            )
        elif side == "root":
            tuner_dbg = self._tuner.apply_root_bet(
                z=z,
                actions=actions,
                hero_mask=hero_mask,
                p_win=(eq_sig.p_win if eq_sig and eq_sig.available else None),
                ex_probs=(list(ex_sig.probs) if ex_sig and ex_sig.probs else None),
                evs=evs,
                action_ctx=ctx,
            )

        # ---------- Final output distribution ----------
        p = self._masked_softmax(
            z, hero_mask.view(1, -1),
            T=float(self.blend.temperature),
            eps=float(self.blend.min_legal_prob),
        )
        p = mix_ties_if_close(p, float(self.blend.tie_mix_threshold))
        p = epsilon_explore(p, float(self.blend.epsilon_explore), hero_mask.view(1, -1))
        p = self._cap_allins(p, eff_stack_bb=float(req.eff_stack_bb or 0.0))
        probs = p[0].tolist()

        best_idx = int(torch.argmax(p[0]))
        best_action = actions[best_idx]

        debug = {
            "router_side": side,
            "menu_mask_sum": float(menu_mask_t.sum().item()),
            "signals": {
                "equity": {
                    "available": eq_sig.available if eq_sig else False,
                    "p_win": eq_sig.p_win if eq_sig else None,
                    "err": eq_sig.err if eq_sig else None,
                },
                "exploit": {
                    "available": ex_sig.available,
                    "counts_total": ex_sig.counts_total,
                    "probs": ex_sig.probs,
                    "err": ex_sig.err,
                },
                "facing": {
                    "is_facing": facing_info.is_facing if facing_info else False,
                    "size_frac": facing_info.size_frac if facing_info else None,
                },
                "root": {
                    "is_root": root_info.is_root if root_info else False,
                    "bet_menu": root_info.bet_menu if root_info else None,
                }
            },
            "tuner_debug": tuner_dbg,
            "blend_cfg": self.blend.to_dict(),
        } if req.debug else None

        return PolicyResponse(
            actions=actions,
            probs=probs,
            evs=[evs.get(a, 0.0) for a in actions],
            notes=[f"Postflop policy; hero_is_ip={hero_is_ip}"],
            debug=debug,
            best_action=best_action,
        )

    def _soft_prior_blend(
            self,
            hero_range: np.ndarray,
            tokens: list[str],
            req: PolicyRequest,
            evs: Optional[Dict[str, float]] = None,
            eq_sig: Optional["EquitySig"] = None,
    ) -> list[float]:

        stack = float(req.eff_stack_bb or req.pot_bb or 100.0)
        facing = bool(req.facing_bet)

        # --- Equity normalization ---
        eq_boost = 1.0
        if eq_sig and eq_sig.available and eq_sig.p_win is not None:
            # p_win in [0,1] → scale to [0.8, 1.3]
            # strong hands get +30% prior, weak hands get -20%
            eq_boost = 0.8 + 0.5 * float(eq_sig.p_win)

        priors = []
        for a in tokens:
            if a == "FOLD":
                base = 0.2 if facing else 0.01
                # If equity is poor, folding slightly more likely
                if eq_sig and eq_sig.available:
                    base *= (1.2 - eq_boost)  # invert equity
                priors.append(base)

            elif a == "CALL":
                base = 0.35 if facing else 0.0
                base *= eq_boost  # strong hands like calling more
                priors.append(base)

            elif a == "CHECK":
                base = 0.35 if not facing else 0.0
                priors.append(base)

            elif a.startswith("RAISE_") or a.startswith("OPEN_"):
                try:
                    amt = float(a.split("_")[1])
                    frac = amt / stack
                    base = 1.0 - abs(frac - 0.5)

                    # EV boost
                    if evs and a in evs:
                        base *= max(evs[a], 0.0) + 1.0

                    # EQUITY boost → aggressive actions
                    base *= eq_boost
                    priors.append(base)

                except:
                    priors.append(0.01)
            else:
                priors.append(0.01)

        return priors

    def _compute_temperature(self, evs: np.ndarray) -> float:
        """
        Computes softmax temperature from EV spread.
        - High spread → lower T (confident)
        - Low spread → higher T (uncertain)
        """
        if len(evs) == 0 or np.allclose(evs, evs[0]):
            return 2.5  # uniform EVs → use high T

        spread = float(np.max(evs) - np.min(evs))
        # Normalize spread into [0, 1]
        norm = np.clip(spread / 0.5, 0.0, 1.0)

        # Invert: high spread → low T
        return 2.5 - 2.0 * norm  # T ∈ [0.5, 2.5]

    @torch.no_grad()
    def _predict_preflop(self, req: PolicyRequest, eq_sig: Optional[EquitySig] = None,
                         villain_range: Optional[np.ndarray] = None) -> PolicyResponse:
        # Step 1: Compute hero hand range (169-dim)
        hero_range = self.rng_pre.predict(req)
        print("Hero range:", hero_range)

        # Step 2: Legal actions
        action_gen = PreflopLegalActionGenerator()
        tokens = action_gen.generate(
            stack_bb=req.eff_stack_bb or req.pot_bb or 100.0,
            facing_bet=req.facing_bet,
            faced_frac=req.faced_size_frac,
        )

        # Step 3: EV estimation (supports villain range)
        ev_calc = EVCalculator()
        ev_sig = ev_calc.compute(req, tokens=tokens, villain_range=villain_range)
        print(f"EV sig: {ev_sig}")
        evs = ev_sig.evs if ev_sig and ev_sig.available else {}
        ev_values = np.array([evs.get(tok, 0.0) for tok in tokens], dtype=np.float32)

        # Step 4: Soft prior blend + temperature scaling
        base_logits = self._soft_prior_blend(hero_range, tokens, req, evs, eq_sig)
        base_logits = np.array(base_logits, dtype=np.float32)
        base_logits = base_logits - np.max(base_logits)  # for numerical stability

        temp = self._compute_temperature(ev_values)
        probs = np.exp(base_logits / max(temp, 1e-6))
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(tokens)) / len(tokens)

        # Final output
        best_action = tokens[int(np.argmax(probs))]

        debug = {
            "hero_range": hero_range.tolist(),
            "equity": eq_sig.p_win if eq_sig else None,
            "ev_values": ev_values.tolist(),
            "temp": temp,
            "tokens": tokens,
        } if req.debug else None

        return PolicyResponse(
            actions=tokens,
            probs=probs.tolist(),
            evs=ev_values.tolist(),
            best_action=best_action,
            notes=[f"preflop policy (range + EV + equity priors T={temp:.2f})"],
            debug=debug,
        )

    def predict(self, req_input: Union[Dict[str, Any], "PolicyRequest"]) -> "PolicyResponse":
        # Parse input
        if isinstance(req_input, dict):
            req = PolicyRequest(**req_input)
        elif isinstance(req_input, PolicyRequest):
            req = req_input
        else:
            raise TypeError(f"PolicyInfer.predict expected dict or PolicyRequest, got {type(req_input)}")

        req.legalize()

        # Ensure ctx & action_seq BEFORE ranges/EV (postflop only)
        if int(getattr(req, "street", 1) or 1) == 0:
            if not hasattr(req, "action_seq") or req.action_seq is None:
                req.action_seq = []
        else:
            ensure_ctx_and_action_seq(req)

        # Villain range provider (stake-cached on the instance)
        stake_token = str(getattr(req, "stakes", "NL10")).upper().replace(" ", "")
        provider_path = f"data/datasets/rangenet_preflop_from_flop_{stake_token}.parquet"
        if not hasattr(self, "_range_providers"):
            self._range_providers = {}
        if provider_path not in self._range_providers:
            self._range_providers[provider_path] = VillainRangeProvider(provider_path)
        range_provider = self._range_providers[provider_path]

        # Load villain range vector (exact/nearest inside provider)
        try:
            villain_range_vec = range_provider.get_range_vector(
                hero_pos=req.hero_pos,
                villain_pos=req.villain_pos,
                stack=req.eff_stack_bb or req.pot_bb or 100.0,
                action_seq=getattr(req, "action_seq", []),
            )
        except Exception:
            villain_range_vec = None

        # Equity signal
        eq_sig = self._signals.collect_equity(req)

        # Route to street-specific policy logic
        if int(getattr(req, "street", 1) or 1) == 0:
            return self._predict_preflop(req, eq_sig=eq_sig, villain_range=villain_range_vec)
        else:
            return self._predict_postflop(req, eq_sig=eq_sig, villain_range=villain_range_vec)