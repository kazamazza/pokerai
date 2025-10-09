from __future__ import annotations
from typing import Any, Dict
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.inference.policy.types import Action, PolicyRequest, PolicyResponse
from ml.inference.preflop import PreflopPolicy, PreflopDeps
import torch
import torch.nn.functional as F
from ml.utils.board_mask import make_board_mask_52


class PolicyInfer:
    def __init__(self, deps: PolicyInferDeps, blend_cfg: PolicyBlendConfig | None = None):
        self.p = deps.params or {}
        self.blend = blend_cfg or PolicyBlendConfig.default()
        if deps.exploit is None:
            raise ValueError("exploit infer is required")
        if deps.equity is None:
            raise ValueError("equity infer is required")
        if deps.range_pre is None:
            raise ValueError("range_pre (RangeNetPreflopInfer) is required")
        if deps.policy_post is None:
            raise ValueError("policy_post (PostflopPolicyInfer) is required")

        # deps
        self.pol_post   = deps.policy_post
        self.pop        = deps.pop
        self.expl       = deps.exploit
        self.eq         = deps.equity
        self.rng_pre    = deps.range_pre
        self.clusterer  = deps.clusterer
        self.p          = deps.params or {}

        # preflop module (stateless facade)
        self._preflop = PreflopPolicy(PreflopDeps(range_pre=self.rng_pre, equity=self.eq))

        # action vocab (shared with postflop model)
        try:
            from ml.models.postflop_policy_net import ACTION_VOCAB as _VOC, VOCAB_INDEX as _VIX
            self.action_vocab = list(_VOC)
            self._vocab_index = dict(_VIX)
        except Exception:
            self.action_vocab = ["FOLD","CHECK","CALL","BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
                                 "DONK_33","RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"]
            self._vocab_index = {a:i for i,a in enumerate(self.action_vocab)}

    def _encode_action(self, a: Action) -> str:
        k = a.kind.upper()
        if k in ("FOLD", "CHECK", "CALL", "ALLIN"):
            return k
        if k == "BET":
            if a.size_pct is not None:  # 33 -> BET_33
                return f"BET_{int(round(a.size_pct))}"
            return "BET_33"
        if k == "RAISE":
            if a.size_mult is not None: # 2.0 -> RAISE_200
                return f"RAISE_{int(round(float(a.size_mult)*100))}"
            return "RAISE_200"
        return k

    @staticmethod
    def _decode_action(token: str) -> Action:
        t = token.upper()
        if t in ("FOLD","CHECK","CALL","ALLIN"):
            return Action(kind=t)
        if t.startswith("BET_"):
            try:
                pct = float(t.split("_",1)[1])
            except Exception:
                pct = 33.0
            return Action(kind="BET", size_pct=pct)
        if t.startswith("RAISE_"):
            try:
                mult = float(t.split("_",1)[1]) / 100.0
            except Exception:
                mult = 2.0
            return Action(kind="RAISE", size_mult=mult)
        # fallback
        return Action(kind="CHECK")

    # --- helper: build one row for postflop model ---
    def _build_postflop_row(self, req: "PolicyRequest") -> dict:
        # Seats: prefer explicit fields, then raw payload, then safe defaults
        ip = (getattr(req, "ip_pos", None) or req.raw.get("ip_pos") or "IP").upper()
        oop = (getattr(req, "oop_pos", None) or req.raw.get("oop_pos") or "OOP").upper()

        # Context + street (keep as integer)
        ctx = (getattr(req, "ctx", None) or req.raw.get("ctx") or "VS_OPEN").upper()
        s = int(getattr(req, "street", 1) or 1)

        # Numbers
        pot = float(getattr(req, "pot_bb", None) or req.raw.get("pot_bb") or 0.0)
        stack = float(
            getattr(req, "eff_stack_bb", None) or req.raw.get("eff_stack_bb") or getattr(req, "stack_bb", 0.0) or 0.0
        )

        # Board mask
        board = getattr(req, "board", None) or req.raw.get("board")
        bm = req.raw.get("board_mask_52") or (make_board_mask_52(board) if board else [0.0] * 52)

        row = {
            "ip_pos": ip,
            "oop_pos": oop,
            "ctx": ctx,
            "street": s,  # ✅ integer street
            "board_mask_52": bm,
            "pot_bb": pot,
            "eff_stack_bb": stack,
        }

        # ✅ Optional board cluster id (if model expects it)
        if hasattr(self, "clusterer") and "board_cluster_id" in getattr(self.pol_post, "feature_order", []):
            try:
                cid = int(self.clusterer.predict_one(board)) if board else 0
            except Exception:
                cid = 0
            row["board_cluster_id"] = cid

        return row

    def _as_batch(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.dim() == 2 else x.view(1, -1)

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor, T: float = 1.0,
                        eps: float = 1e-6) -> torch.Tensor:
        logits = self._as_batch(logits)
        mask = self._as_batch(mask).to(logits.dtype)
        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits / max(T, 1e-6), big_neg)
        p = F.softmax(masked, dim=-1)
        # minimum mass on legal actions (avoid exact zeros)
        p = p * (1 - eps) + (eps / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)) * (mask > 0.5).to(p.dtype)
        # renormalize within legal set
        p = p * mask
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return p

    def _mix_ties_if_close(self, p: torch.Tensor, thresh: float) -> torch.Tensor:
        p = self._as_batch(p)
        top2 = torch.topk(p, k=min(2, p.size(-1)), dim=-1)  # values, indices
        if top2.values.size(-1) < 2:
            return p
        close = (top2.values[:, 0] - top2.values[:, 1]).abs() <= thresh
        if not close.any():
            return p
        # mix 60/40 between top-1 and top-2 when close
        B, V = p.shape
        for b in torch.nonzero(close).view(-1):
            i1, i2 = int(top2.indices[b, 0]), int(top2.indices[b, 1])
            mass = p[b, i1] + p[b, i2]
            p[b, :] *= 0.0
            p[b, i1] = 0.6 * mass
            p[b, i2] = 0.4 * mass
        return p

    def _epsilon_explore(self, p: torch.Tensor, eps: float, mask: torch.Tensor) -> torch.Tensor:
        if eps <= 0:
            return p
        p = self._as_batch(p)
        mask = self._as_batch(mask).to(p.dtype)
        V = p.size(-1)
        uni = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        p = (1 - eps) * p + eps * uni
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return p

    def _cap_allins(self, p: torch.Tensor, row: dict, mask: torch.Tensor, blend) -> torch.Tensor:
        # Example: if eff_stack is deep, don’t allow high all-in freq
        try:
            eff_stack = float(row.get("eff_stack_bb", row.get("stack_bb", 0.0)) or 0.0)
        except Exception:
            eff_stack = 0.0
        if eff_stack <= blend.risk_floor_stack_bb:
            return p
        # find ALLIN index
        try:
            allin_idx = self.action_vocab.index("ALLIN")
        except ValueError:
            return p
        p = self._as_batch(p).clone()
        M = self._as_batch(mask)
        # If ALLIN is illegal or already under cap, no change
        if M[0, allin_idx] < 0.5 or p[0, allin_idx] <= blend.max_allin_freq:
            return p
        # cap and re-normalize remaining legal mass
        over = p[0, allin_idx] - blend.max_allin_freq
        p[0, allin_idx] = blend.max_allin_freq
        legal = (M[0] > 0.5).nonzero().view(-1).tolist()
        rest = [i for i in legal if i != allin_idx]
        if rest:
            scale = 1.0 + (over / (p[0, rest].sum().clamp_min(1e-8)))
            p[0, rest] *= scale
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return p

    def _build_legality_masks(
            self,
            *,
            req: "PolicyRequest",
            vocab: list[str],
            actor: str = "both",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Return (mask_ip, mask_oop) where each is a [V] float tensor in {0,1}.
        Very simple legality:
          - not facing bet: CHECK, BET_*, DONK_* (donk only meaningful for OOP)
          - facing bet    : FOLD, CALL, RAISE_*, ALLIN
        """

        def mask_for_side(side: str, facing_bet: bool) -> torch.Tensor:
            m = torch.zeros(len(vocab), dtype=torch.float32)
            for i, a in enumerate(vocab):
                A = a.upper()
                legal = False
                if not facing_bet:
                    if A == "CHECK":
                        legal = True
                    elif A.startswith("BET_"):
                        legal = True
                    elif A.startswith("DONK_"):
                        legal = (side == "oop")  # donk only for OOP
                    elif A == "ALLIN":
                        legal = True  # allow shove
                else:
                    if A in ("FOLD", "CALL", "ALLIN"):
                        legal = True
                    elif A.startswith("RAISE_"):
                        legal = True
                if legal:
                    m[i] = 1.0
            if m.sum().item() == 0:
                m.fill_(1.0)
            return m

        facing = bool(getattr(req, "facing_bet", False))

        if actor.lower() == "ip":
            return mask_for_side("ip", facing), None
        elif actor.lower() == "oop":
            return None, mask_for_side("oop", facing)
        else:
            # both: same facing flag applied to both (simple; you can refine later)
            return mask_for_side("ip", facing), mask_for_side("oop", facing)

    def _apply_exploit_adjustment(self, z: torch.Tensor, exploit_signal: Any) -> torch.Tensor:
        """
        z: [B, V] logits
        exploit_signal:
          - None -> no change
          - torch.Tensor / ndarray / list length V -> added elementwise
          - dict[action_token -> delta] -> map tokens -> idx using self._vocab_index and add
        """
        if exploit_signal is None:
            return z
        if isinstance(exploit_signal, torch.Tensor):
            delta = exploit_signal.to(z.device).view(1, -1)
            if delta.size(-1) == z.size(-1):
                return z + delta
            # else ignore if shape mismatch
            return z

        # ndarray / list-like
        try:
            import numpy as np
            if isinstance(exploit_signal, (list, tuple, np.ndarray)):
                arr = np.asarray(exploit_signal, dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] == z.size(-1):
                    delta = torch.tensor(arr, dtype=z.dtype, device=z.device).view(1, -1)
                    return z + delta
        except Exception:
            pass

        # dict mapping token -> delta
        if isinstance(exploit_signal, dict):
            delta = torch.zeros_like(z)
            for tok, val in exploit_signal.items():
                idx = self._vocab_index.get(str(tok).upper())
                if idx is None:
                    continue
                try:
                    delta[0, int(idx)] = float(val)
                except Exception:
                    continue
            return z + delta

        # fallback: no change
        return z

    def _predict_postflop(self, req_dict: Dict[str, Any]) -> PolicyResponse:
        req = PolicyRequest(**req_dict)
        actor = (getattr(req, "actor", None) or getattr(req, "to_act", None) or "ip").lower()
        if actor not in ("ip", "oop"):
            actor = "ip"

        # Build input row and legality masks
        row = self._build_postflop_row(req)
        mask_ip, mask_oop = self._build_legality_masks(req=req, vocab=self.action_vocab, actor=actor)

        # Run model inference (ask for logits so we can apply exploit deltas pre-softmax)
        out = self.pol_post.predict_proba([row], actor=actor, return_logits=True)
        li, lo = out["logits_ip"], out["logits_oop"]  # [1, V]
        # start from model logits
        z_ip, z_oop = li.clone(), lo.clone()

        # --- Exploit signal integration (if available) ---
        exploit_debug = None
        try:
            if self.expl and getattr(req, "villain_id", None):
                vid = str(req.villain_id)

                # Build a simple scenario key that matches how observations are keyed in live usage.
                scenario_key = f"{row.get('street')}:{row.get('ctx') or ''}:{row.get('ip_pos') or ''}:{row.get('oop_pos') or ''}"

                # Try a couple of calling conventions to be tolerant to small API differences:
                # 1) get_signal(player_id, scenario_key, pop_probs=None)
                # 2) get_signal(player_id, row)  (legacy)
                exploit_signal_raw = None
                pop_probs = None  # optional: if you have population probs, plug them here

                try:
                    # preferred form (player_id, scenario, pop_probs)
                    exploit_signal_raw = self.expl.get_signal(vid, scenario_key, pop_probs)
                except TypeError:
                    # fallback to legacy signature (player_id, row)
                    try:
                        exploit_signal_raw = self.expl.get_signal(vid, row)
                    except Exception:
                        exploit_signal_raw = None
                except Exception:
                    exploit_signal_raw = None

                # No signal → skip
                if exploit_signal_raw is None:
                    exploit_debug = {"signal": None}
                else:
                    # Normalize returned signal to a length-V numpy array
                    V = z_ip.shape[-1]
                    sig_vec = None

                    # dict mapping token/index -> delta
                    if isinstance(exploit_signal_raw, dict):
                        # If keys are tokens, map to indices if possible
                        sig = np.zeros(V, dtype=float)
                        for k, v in exploit_signal_raw.items():
                            try:
                                # token -> index via vocab index if available
                                if isinstance(k, str) and hasattr(self, "_vocab_index"):
                                    idx = int(self._vocab_index.get(k, -1))
                                else:
                                    idx = int(k)
                                if 0 <= idx < V:
                                    sig[idx] = float(v)
                            except Exception:
                                continue
                        sig_vec = sig
                    else:
                        # list/ndarray-like
                        try:
                            arr = np.asarray(exploit_signal_raw, dtype=float).reshape(-1)
                            if arr.shape[0] == V:
                                sig_vec = arr
                            else:
                                # mismatched length -> ignore
                                sig_vec = None
                        except Exception:
                            sig_vec = None

                    if sig_vec is None:
                        exploit_debug = {"signal_parsed": None, "raw": exploit_signal_raw}
                    else:
                        # Convert to torch [1, V] with same dtype/device as logits
                        dev = z_ip.device
                        dtype = z_ip.dtype
                        t_signal = torch.tensor(sig_vec, dtype=dtype, device=dev).view(1, -1)

                        # scale by blend.lambda_expl (blend weight)
                        scale = float(self.blend.lambda_expl)

                        # Add the deltas to logits (additive logit-space adjustment)
                        z_ip = z_ip + scale * t_signal
                        z_oop = z_oop + scale * t_signal

                        exploit_debug = {
                            "applied": True,
                            "scale": scale,
                            "signal_shape": list(t_signal.shape),
                            "signal_norm": float(t_signal.abs().sum().cpu().item()),
                        }
        except Exception as e:
            # don't break inference on exploit errors; keep a debug note
            exploit_debug = {"error": str(e)}

        blend = self.blend

        # Apply other component deltas placeholders (EQ/POP/RISK) if/when you compute them
        # For now those are zero tensors (kept for clarity)
        # z_ip/z_oop already contain model logits + exploit deltas

        # Masked softmax (respects legality and temperature)
        p_ip = self._masked_softmax(z_ip, mask_ip, T=blend.temperature, eps=blend.min_legal_prob)
        p_oop = self._masked_softmax(z_oop, mask_oop, T=blend.temperature, eps=blend.min_legal_prob)

        # Tie-mix, exploration, all-in capping
        p_ip = self._mix_ties_if_close(p_ip, blend.tie_mix_threshold)
        p_oop = self._mix_ties_if_close(p_oop, blend.tie_mix_threshold)

        p_ip = self._epsilon_explore(p_ip, blend.epsilon_explore, mask_ip)
        p_oop = self._epsilon_explore(p_oop, blend.epsilon_explore, mask_oop)

        p_ip = self._cap_allins(p_ip, row, mask_ip, blend)
        p_oop = self._cap_allins(p_oop, row, mask_oop, blend)

        # Select actor-specific probabilities
        probs = p_ip[0].tolist() if actor == "ip" else p_oop[0].tolist()

        return PolicyResponse(
            actions=self.action_vocab,
            probs=probs,
            evs=[0.0] * len(self.action_vocab),
            notes=[f"Postflop policy; actor={actor}, temp={blend.temperature:.2f}"],
            debug={
                "actor": actor,
                "ctx": row.get("ctx"),
                "street": row.get("street"),
                "board_cluster_id": row.get("board_cluster_id"),
                "blend_cfg": blend.to_dict(),
                "exploit_debug": exploit_debug,
            },
        )

    def predict(self, req_dict: Dict[str, Any]) -> PolicyResponse:
        """Unified entrypoint for both preflop and postflop policies."""
        req = PolicyRequest(**req_dict)

        try:
            street = int(req.street or 0)
        except Exception:
            street = 0

        if street not in (0, 1, 2, 3):
            board = (req.board or "").strip()
            n = len(board)
            street = 3 if n >= 10 else 2 if n >= 8 else 1 if n >= 6 else 0

        # === Preflop ===
        if street == 0:
            out: PolicyResponse = self._preflop.predict(req)
            return out

        # === Postflop ===
        return self._predict_postflop(dict(req_dict, street=street))