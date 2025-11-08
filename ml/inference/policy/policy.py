import math
from typing import Any, Dict, Union, List, Optional, Tuple

import numpy as np

from ml.features.hands import hand_to_169_label
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy_blend_config import PolicyBlendConfig, tuner_knobs_from_blend
from ml.inference.policy.projection import FCRProjector
from ml.inference.policy.signals import SignalCollector
from ml.inference.policy.tuner import PostflopTuner
from ml.inference.policy.types import PolicyRequest, PolicyResponse
import torch
from ml.inference.policy.utils import postflop_is_hero_ip, mix_ties_if_close, \
    epsilon_explore
import torch.nn.functional as F


class PolicyInfer:
    """
    Thin, deterministic orchestrator over your existing inference classes.
    Fixes:
    - No mask loss (router/menu masks merged, never discarded).
    - Single temperature application (router called with T=1.0).
    - Vocab/index/projection refresh per call.
    - Equity/Exploit deltas applied in logit-space, then masked softmax.
    """

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
        self._signals = SignalCollector(deps.equity, deps.exploit, deps.pop,
                                        router_facing=getattr(deps.policy_post, "facing", None))
        self._proj = FCRProjector()
        self._tuner = PostflopTuner(tuner_knobs_from_blend(self.blend))

        # Optional external vocab constants (best-effort)
        try:
            from ml.models.policy_consts import ACTION_VOCAB as _VOC, VOCAB_INDEX as _VIX  # type: ignore
            self.action_vocab = list(_VOC)
            self._vocab_index = dict(_VIX)
        except Exception:
            # stays empty; will update after first router call
            pass

        # Interface sanity
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

    def _infer_equity_postflop(self, req: "PolicyRequest") -> Tuple[Optional[Dict[str, float]], str]:
        """Best-effort equity inference. Returns (equity_dict or None, reason)."""
        if self.eq is None:
            return None, "no_equity_model"
        if not req.hero_hand:
            return None, "no_hero_hand"
        if not req.board:
            return None, "no_board"

        # optional: reject overlap hero↔board for safety
        try:
            board_cards = [req.board[i:i + 2] for i in range(0, len(req.board), 2)]
            if any(c in (req.hero_hand or "") for c in board_cards):
                return None, "hand_board_overlap"
        except Exception:
            pass

        # hand→169 mapping (optional)
        try:

            hand169 = int(hand_to_169_label(req.hero_hand))
        except Exception:
            return None, "no_169_mapping"

        payload = {"street": int(req.street), "hand_id": hand169}
        try:
            if hasattr(self.eq, "predict_proba"):
                out = self.eq.predict_proba([payload])
            else:
                out = self.eq.predict([payload])
            if not out or len(out[0]) != 3:
                return None, "eq_model_empty_or_bad_shape"
            p_win, p_tie, p_lose = map(float, out[0])
            return {"p_win": p_win, "p_tie": p_tie, "p_lose": p_lose}, ""
        except Exception as e:
            return None, f"eq_predict_error:{e}"

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
        # ensure tiny mass stays on legal set
        p = p * (1 - eps) + (eps / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)) * (mask > 0.5).to(p.dtype)
        p = p * mask
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return p

    def _fcr_projection(self) -> torch.Tensor:
        """Return [3,V] projection from F/C/R deltas → vocab deltas."""
        if self._P_fcr is not None and self._P_fcr.shape[1] == len(self.action_vocab):
            return self._P_fcr
        V = len(self.action_vocab)
        P = torch.zeros(3, V, dtype=torch.float32)
        if "FOLD" in self._vocab_index:
            P[0, self._vocab_index["FOLD"]] = 1.0
        if "CALL" in self._vocab_index:
            P[1, self._vocab_index["CALL"]] = 1.0
        raise_like_idx = []
        for i, tok in enumerate(self.action_vocab):
            T = tok.upper()
            if T.startswith("BET_") or T.startswith("RAISE_") or T.startswith("DONK_") or T == "ALLIN":
                raise_like_idx.append(i)
        if raise_like_idx:
            w = 1.0 / float(len(raise_like_idx))
            for i in raise_like_idx:
                P[2, i] = w
        self._P_fcr = P
        return self._P_fcr

    def _lift_exploit_signal(self, sig_fcr: "np.ndarray | torch.Tensor", device, dtype) -> torch.Tensor:
        """Lift shape-(3,) F/C/R logit deltas into [1,V] aligned to current vocab."""
        if not isinstance(sig_fcr, torch.Tensor):
            sig_fcr = torch.tensor(sig_fcr, dtype=dtype, device=device)
        sig_fcr = sig_fcr.view(3)
        P = self._fcr_projection().to(device=device, dtype=dtype)  # [3,V]
        delta_v = torch.matmul(sig_fcr, P)  # [V]
        return delta_v.view(1, -1)

    def _ruler_group_delta(
            self,
            *,
            z: torch.Tensor,  # [1, V] logits (pre-temperature)
            legal_idx: List[int],  # indices of currently-legal actions
            group_idx: List[int],  # indices of the group to boost (e.g., all RAISE_*)
            tau: float,  # desired total share for the group (0..1)
            max_boost: float = 8.0  # safety cap on the uniform logit bump
    ) -> float:
        """
        Compute a single uniform delta 'd' to add to all group logits so that
        softmax mass over the group ≈ tau (within the legal set).
        If current mass ≥ tau, returns 0.0.
        """
        if not legal_idx or not group_idx or tau <= 0.0:
            return 0.0

        l = z[0]  # [V]
        # log-sum-exp over legal and group sets
        L_logsum = torch.logsumexp(l[legal_idx], dim=0)  # log( sum_{k ∈ L} exp(l_k) )
        G_logsum = torch.logsumexp(l[group_idx], dim=0)  # log( sum_{g ∈ G} exp(l_g) )
        # current group share within legal set
        g_share = torch.exp(G_logsum - L_logsum).item()
        if tau <= g_share + 1e-9:
            return 0.0

        # Notation: Sg = sum_{g∈G} exp(l_g), So = sum_{o∈L\G} exp(l_o)
        Sg = torch.exp(G_logsum).item()
        So = max(torch.exp(L_logsum).item() - Sg, 1e-12)

        # Want: (Sg * exp(d)) / (So + Sg * exp(d)) = tau
        # -> exp(d) = (tau / (1 - tau)) * (So / Sg)
        # -> d = log( (tau/(1 - tau)) * So / Sg )
        ratio = max(tau / max(1.0 - tau, 1e-6), 1e-6)
        d = math.log(ratio * (So / max(Sg, 1e-12)))
        d = max(0.0, min(d, float(max_boost)))
        return float(d)

    def _menu_mask(self, actions: List[str], *, actor: str, facing_bet: bool,
                  bet_sizes: Optional[List[float]],
                  raise_buckets: Optional[List[int]],
                  allow_allin: Optional[bool]) -> torch.Tensor:
        """Menu-aware legalities aligned to actions."""
        allow_ai = True if allow_allin is None else bool(allow_allin)
        m = torch.zeros(len(actions), dtype=torch.float32)

        # Precompute lookups
        raise_set = set(int(x) for x in (raise_buckets or []))
        bet_percents = None
        if bet_sizes is not None:
            # Accept [0.33, 0.66] or [33, 66]
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
                # FACING
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
            # Failsafe: if we somehow zeroed the entire row, make all legal.
            # (Optional) print/log once here so you can spot this during smoke tests.
            m.fill_(1.0)

        return m

    @torch.no_grad()
    def _predict_postflop(self, req: "PolicyRequest") -> "PolicyResponse":
        hero_is_ip = postflop_is_hero_ip(req)
        side = self._derive_side(req, hero_is_ip=hero_is_ip)
        router_resp = self.pol_post.predict(req, actor=("ip" if hero_is_ip else "oop"),
                                            temperature=1.0, side=side)

        actions = list(getattr(router_resp, "actions", []));
        self._update_vocab(actions)
        logits = (torch.tensor(router_resp.logits, dtype=torch.float32).view(1, -1)
                  if getattr(router_resp, "logits", None) is not None
                  else torch.log(torch.tensor(router_resp.probs, dtype=torch.float32).view(1, -1).clamp_min(1e-8)))
        mask_router = getattr(router_resp, "mask", None)
        hero_mask_router = torch.tensor(mask_router, dtype=torch.float32).view(1, -1)[
            0] if mask_router is not None else torch.ones(len(actions))
        mm = self._menu_mask(actions,
                       actor=("ip" if hero_is_ip else "oop"),
                       facing_bet=(side == "facing"),
                       bet_sizes=getattr(req, "bet_sizes", None),
                       raise_buckets=getattr(req, "raise_buckets", None),
                       allow_allin=getattr(req, "allow_allin", None))
        hero_mask = ((hero_mask_router > 0.5).float() * (mm > 0.5).float())
        if hero_mask.sum() <= 0: hero_mask = torch.ones_like(mm)

        # --- collect signals (always) ---
        eq_sig = self._signals.collect_equity(req)
        ex_sig = self._signals.collect_exploit(req)
        facing = self._signals.collect_facing(req, hero_is_ip)

        z = logits

        # --- apply exploit logit deltas only if lambda_expl>0 ---
        if ex_sig.available and self.blend.lambda_expl > 0.0 and ex_sig.raw is not None:
            z = z + float(self.blend.lambda_expl) * self._proj.lift(ex_sig.raw, actions, z.dtype, z.device)

        # --- tuner (facing raises) uses signals irrespective of lambdas ---
        tuner_dbg = None
        if side == "facing":
            tuner_dbg = self._tuner.apply_facing_raise(
                z=z, actions=actions, hero_mask=hero_mask,
                p_win=(eq_sig.p_win if eq_sig.available else None),
                ex_probs=(list(ex_sig.probs) if ex_sig.probs else None),
                size_frac=facing.size_frac,
            )

        # --- finalize ---
        p = self._masked_softmax(z, hero_mask.view(1, -1), T=float(self.blend.temperature),
                                 eps=float(self.blend.min_legal_prob))
        p = mix_ties_if_close(p, float(self.blend.tie_mix_threshold))
        p = epsilon_explore(p, float(self.blend.epsilon_explore), hero_mask.view(1, -1))
        p = self._cap_allins(p, eff_stack_bb=float(getattr(req, "eff_stack_bb", 0.0)))
        probs = p[0].tolist()

        debug = {
            "router_side": side,
            "menu_mask_sum": float(mm.sum().item()),
            "signals": {
                "equity": {"available": eq_sig.available, "p_win": eq_sig.p_win, "err": eq_sig.err},
                "exploit": {"available": ex_sig.available, "counts_total": ex_sig.counts_total,
                            "probs": ex_sig.probs, "err": ex_sig.err},
                "facing": {"is_facing": facing.is_facing, "size_frac": facing.size_frac},
            },
            "tuner_debug": tuner_dbg,
            "blend_cfg": self.blend.to_dict(),
        }
        return PolicyResponse(actions=actions, probs=probs, evs=[0.0] * len(actions),
                              notes=[f"Postflop policy; hero_is_ip={hero_is_ip}"], debug=debug)

    # -----------------------
    # Public entrypoint
    # -----------------------
    def predict(self, req_input: Union[Dict[str, Any], "PolicyRequest"]) -> "PolicyResponse":
        # Normalize request
        if isinstance(req_input, dict):
            req = PolicyRequest(**req_input)
        elif hasattr(req_input, "__dict__"):
            req = req_input  # PolicyRequest
        else:
            raise TypeError(f"PolicyInfer.predict expected dict or PolicyRequest, got {type(req_input)}")

        req.legalize()  # raises on inconsistency

        street = int(getattr(req, "street", 0))
        if street == 0:
            # Preflop: keep your original behavior; optional equity nudge
            equity = None
            if self.eq and getattr(req, "hero_hand", None):
                try:
                    # If your equity supports preflop hand_id only; safe best-effort
                    hand169 = int(hand_to_169_label(req.hero_hand))  # may raise; that's fine
                    out = self.eq.predict([{"street": 0, "hand_id": hand169}])
                    if out:
                        p_win, p_tie, p_lose = out[0]
                        equity = {"p_win": float(p_win), "p_tie": float(p_tie), "p_lose": float(p_lose)}
                except Exception:
                    equity = None

            eq_nudge = float(self.blend.equity_nudge_pre)
            try:
                req_nudge = float(getattr(req, "raw", {}).get("equity_nudge"))
                if math.isfinite(req_nudge):
                    eq_nudge = req_nudge
            except Exception:
                pass

            return self.rng_pre.predict(
                req,
                equity=equity,
                temperature=float(self.blend.temperature),
                equity_nudge=eq_nudge,
            )

        # Postflop path
        return self._predict_postflop(req)