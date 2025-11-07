from __future__ import annotations
import hashlib
import math
from typing import Any, Dict, Union, List, Optional, Tuple
import numpy as np
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.inference.policy.types import Action, PolicyRequest, PolicyResponse
import torch
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

        try:
            bcf = getattr(self.pol_post, "board_cluster_feat", None)
            if bcf:
                print(f"[policy] postflop expects board cluster feature: {bcf}")
        except Exception:
            pass

        # Vocab cache
        self.action_vocab: List[str] = []
        self._vocab_index: Dict[str, int] = {}
        self._P_fcr: Optional[torch.Tensor] = None  # [3,V], lazy

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

    # -----------------------
    # Vocab & projection
    # -----------------------
    def _update_vocab(self, actions: List[str]) -> None:
        """Rebuild index and invalidate F/C/R projection if actions changed."""
        if actions == self.action_vocab:
            return
        self.action_vocab = list(actions)
        self._vocab_index = {a: i for i, a in enumerate(self.action_vocab)}
        self._P_fcr = None  # invalidate

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

    # -----------------------
    # Masks & blend ops
    # -----------------------
    @staticmethod
    def _as_batch(x: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def _mix_ties_if_close(p: torch.Tensor, thresh: float) -> torch.Tensor:
        p = p if p.dim() == 2 else p.view(1, -1)
        k = min(2, p.size(-1))
        if k < 2:
            return p
        top2 = torch.topk(p, k=k, dim=-1)
        close = (top2.values[:, 0] - top2.values[:, 1]).abs() <= thresh
        if not close.any():
            return p
        for b in torch.nonzero(close).view(-1):
            i1, i2 = int(top2.indices[b, 0]), int(top2.indices[b, 1])
            mass = p[b, i1] + p[b, i2]
            p[b, :] *= 0.0
            p[b, i1] = 0.6 * mass
            p[b, i2] = 0.4 * mass
        return p

    @staticmethod
    def _epsilon_explore(p: torch.Tensor, eps: float, mask: torch.Tensor) -> torch.Tensor:
        if eps <= 0:
            return p
        p = p if p.dim() == 2 else p.view(1, -1)
        mask = mask if mask.dim() == 2 else mask.view(1, -1)
        uni = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        p = (1 - eps) * p + eps * uni
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return p

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

    @staticmethod
    def _menu_mask(actions: List[str], *, actor: str, facing_bet: bool,
                   bet_sizes: Optional[List[float]],
                   raise_buckets: Optional[List[int]],
                   allow_allin: Optional[bool]) -> torch.Tensor:
        """Menu-aware legalities aligned to actions."""
        allow_ai = True if allow_allin is None else bool(allow_allin)
        m = torch.zeros(len(actions), dtype=torch.float32)
        for i, tok in enumerate(actions):
            T = tok.upper()
            legal = False
            if not facing_bet:
                if T == "CHECK":
                    legal = True
                elif T.startswith("BET_"):
                    if bet_sizes is None:
                        legal = True
                    else:
                        try:
                            pct = int(T.split("_", 1)[1])
                            legal = any(int(round(s * 100)) == pct for s in bet_sizes)
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
                    if raise_buckets is None:
                        legal = True
                    else:
                        try:
                            mult = int(T.split("_", 1)[1])
                            legal = mult in set(raise_buckets)
                        except Exception:
                            legal = False
                elif T == "ALLIN":
                    legal = allow_ai
            if legal:
                m[i] = 1.0
        if m.sum().item() == 0:
            m.fill_(1.0)
        return m

    # -----------------------
    # Equity & exploit deltas
    # -----------------------
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

    def _lift_exploit_signal(self, sig_fcr: "np.ndarray | torch.Tensor", device, dtype) -> torch.Tensor:
        """Lift shape-(3,) F/C/R logit deltas into [1,V] aligned to current vocab."""
        if not isinstance(sig_fcr, torch.Tensor):
            sig_fcr = torch.tensor(sig_fcr, dtype=dtype, device=device)
        sig_fcr = sig_fcr.view(3)
        P = self._fcr_projection().to(device=device, dtype=dtype)  # [3,V]
        delta_v = torch.matmul(sig_fcr, P)  # [V]
        return delta_v.view(1, -1)

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
            from ml.poker.handmap import hand_to_169_label  # type: ignore
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

    # -----------------------
    # Postflop path
    # -----------------------
    def _predict_postflop(self, req: "PolicyRequest") -> "PolicyResponse":
        # Actor decision from positions
        hero_is_ip = PolicyRequest.is_hero_ip(req.hero_pos or "", req.villain_pos or "")
        actor = "ip" if hero_is_ip else "oop"

        # Router inference: **temperature-free**; try to force side=actor
        router_resp = self.pol_post.predict(
            req,
            actor=actor,
            temperature=1.0,     # ensure logits/probs are T=1
            side=actor,          # keep router's internal view aligned
        )

        # Actions & logits standardization
        actions = list(getattr(router_resp, "actions", []))
        if not actions:
            raise RuntimeError("Router returned empty actions")
        self._update_vocab(actions)
        V = len(actions)

        if hasattr(router_resp, "logits") and router_resp.logits is not None:
            logits = torch.tensor(router_resp.logits, dtype=torch.float32).view(1, V)
        else:
            eps = 1e-8
            p = torch.tensor(router_resp.probs, dtype=torch.float32).view(1, V)
            logits = torch.log(torch.clamp(p, min=eps))

        # Router-provided mask (optional)
        mask_router = getattr(router_resp, "mask", None)
        if mask_router is None:
            hero_mask_router = torch.ones(V, dtype=torch.float32)
        else:
            hero_mask_router = torch.tensor(mask_router, dtype=torch.float32).view(1, V)[0]

        # Menu mask (request-level menus)
        menu_mask = self._menu_mask(
            actions,
            actor=actor,
            facing_bet=bool(getattr(req, "facing_bet", False)),
            bet_sizes=req.bet_sizes,
            raise_buckets=req.raise_buckets,
            allow_allin=req.allow_allin,
        )

        # Final mask = router ∧ menu; never all-zero
        hero_mask = (hero_mask_router > 0.5).to(torch.float32) * (menu_mask > 0.5).to(torch.float32)
        if hero_mask.sum() <= 0:
            hero_mask = torch.ones_like(menu_mask)

        # Equity delta
        equity_debug: Dict[str, Any] = {"lambda_eq": float(self.blend.lambda_eq)}
        z = logits  # start in logit-space
        if self.blend.lambda_eq > 0.0:
            equity, eq_err = self._infer_equity_postflop(req)
            if eq_err:
                equity_debug.update({"applied": False, "reason": eq_err})
            elif not equity:
                equity_debug.update({"applied": False, "reason": "no_equity_dict"})
            else:
                eq_margin = float(equity.get("p_win", 0.5)) - 0.5
                gate = float(self.blend.eq_min_abs_margin)
                if abs(eq_margin) < gate:
                    equity_debug.update({"applied": False, "reason": "small_margin", "p_win": float(equity.get("p_win", 0.5))})
                else:
                    d = self._equity_delta_vector(eq_margin=eq_margin, hero_is_ip=hero_is_ip, facing_bet=bool(req.facing_bet))
                    mx = float(self.blend.eq_max_logit_delta)
                    d = torch.clamp(d, -mx, mx).to(z.dtype).to(z.device)
                    z = z + float(self.blend.lambda_eq) * d
                    equity_debug.update({"applied": True, "p_win": float(equity.get("p_win", 0.5)), "margin": float(eq_margin), "sum_abs": float(d.abs().sum().item())})

        # Exploit delta via PopNet/Store
        exploit_debug: Dict[str, Any] | None = None
        try:
            if self.expl is not None and getattr(req, "villain_id", None) and (self.pop is not None):
                sig3 = self.expl.get_signal_from_request(str(req.villain_id), req, self.pop)  # expected shape (3,) in logit space
                if sig3 is not None:
                    t_delta = self._lift_exploit_signal(sig3, device=z.device, dtype=z.dtype)  # [1, V]
                    z = z + float(self.blend.lambda_expl) * t_delta
                    exploit_debug = {"applied": True, "scale": float(self.blend.lambda_expl), "sum_abs": float(t_delta.abs().sum().item())}
                else:
                    exploit_debug = {"applied": False, "reason": "insufficient_data_or_small_kl"}
        except Exception as e:
            exploit_debug = {"error": str(e)}

        # Blend → probs (apply **temperature once** here)
        p = self._masked_softmax(z, hero_mask.view(1, -1), T=float(self.blend.temperature), eps=float(self.blend.min_legal_prob))
        p = self._mix_ties_if_close(p, float(self.blend.tie_mix_threshold))
        p = self._epsilon_explore(p, float(self.blend.epsilon_explore), hero_mask.view(1, -1))
        p = self._cap_allins(p, eff_stack_bb=float(getattr(req, "eff_stack_bb", 0.0)))

        probs = p[0].tolist()

        # Debug payload
        bcf = getattr(self.pol_post, "board_cluster_feat", None)
        debug = {
            "actor": actor,
            "hero_is_ip": hero_is_ip,
            "router_mask_sum": float(hero_mask_router.sum().item()),
            "menu_mask_sum": float(menu_mask.sum().item()),
            "blend_cfg": self.blend.__dict__,
            "exploit_debug": exploit_debug,
            "equity_debug": equity_debug,
            "vocab_sha": hashlib.sha256("".join(actions).encode("utf-8")).hexdigest()[:12],
            "board_cluster_id": getattr(router_resp, "meta", {}).get("board_cluster_id") if hasattr(router_resp, "meta") else None,
        }

        return PolicyResponse(
            actions=actions,
            probs=probs,
            evs=[0.0] * len(actions),
            notes=[f"Postflop policy; hero_is_ip={hero_is_ip}, temp={float(self.blend.temperature):.2f}"],
            debug=debug,
        )

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
                    from ml.poker.handmap import hand_to_169_label  # type: ignore
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