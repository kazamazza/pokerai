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

    def _predict_postflop(self, req_dict: Dict[str, Any]) -> PolicyResponse:
        req = PolicyRequest(**req_dict)
        actor = (getattr(req, "actor", None) or getattr(req, "to_act", None) or "ip").lower()
        if actor not in ("ip", "oop"):
            actor = "ip"

        # Build input row and legality masks
        row = self._build_postflop_row(req)
        mask_ip, mask_oop = self._build_legality_masks(req=req, vocab=self.action_vocab, actor=actor)

        # Run model inference
        out = self.pol_post.predict_proba([row], actor=actor, return_logits=False)
        pi, po = out["probs_ip"], out["probs_oop"]

        blend = self.blend

        # Apply blending corrections
        p_ip = self._mix_ties_if_close(pi, blend.tie_mix_threshold)
        p_oop = self._mix_ties_if_close(po, blend.tie_mix_threshold)

        p_ip = self._epsilon_explore(p_ip, blend.epsilon_explore, mask_ip)
        p_oop = self._epsilon_explore(p_oop, blend.epsilon_explore, mask_oop)

        p_ip = self._cap_allins(p_ip, row, mask_ip, blend)
        p_oop = self._cap_allins(p_oop, row, mask_oop, blend)

        # Select actor-specific probabilities
        if actor == "ip":
            probs = p_ip[0].tolist()
        else:  # "oop"
            probs = p_oop[0].tolist()

        # ✅ Return a proper PolicyResponse
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
            },
        )

    def predict(self, req_dict: Dict[str, Any]) -> PolicyResponse:
        """Unified entrypoint for both preflop and postflop policies."""
        req = PolicyRequest(**req_dict)

        # Infer street if missing
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
            # Already returns a PolicyResponse
            return out

        # === Postflop ===
        return self._predict_postflop(dict(req_dict, street=street))