from __future__ import annotations
from typing import Any, Dict
import numpy as np

from ml.features.hands import hand_to_169_label
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.inference.policy.types import Action, PolicyRequest, PolicyResponse
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
            raise ValueError("range_pre (PreflopPolicy) is required")
        if deps.policy_post is None:
            raise ValueError("policy_post (PostflopPolicyInfer) is required")

        # deps
        self.pol_post   = deps.policy_post
        self.pop        = deps.pop
        self.expl       = deps.exploit
        self.eq         = deps.equity
        self.rng_pre    = deps.range_pre     # <- keep the original object
        self.clusterer  = deps.clusterer

        # if the postflop infer supports an explicit setter, pass it through
        if self.clusterer is not None and hasattr(self.pol_post, "set_clusterer"):
            try:
                self.pol_post.set_clusterer(self.clusterer)
            except Exception:
                pass

        # (optional) quick visibility on what the postflop model expects
        try:
            bcf = getattr(self.pol_post, "board_cluster_feat", None)
            if bcf:
                print(f"[policy] postflop expects board cluster feature: {bcf}")
        except Exception:
            pass


        self.p          = deps.params or {}

        # ✅ preflop module: use what we were passed; do NOT re-wrap
        self._preflop = self.rng_pre

        # action vocab (shared with postflop model)
        try:
            from ml.models.postflop_policy_net import ACTION_VOCAB as _VOC, VOCAB_INDEX as _VIX
            self.action_vocab = list(_VOC)
            self._vocab_index = dict(_VIX)
        except Exception:
            self.action_vocab = ["FOLD","CHECK","CALL","BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
                                 "DONK_33","RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"]
            self._vocab_index = {a:i for i,a in enumerate(self.action_vocab)}

        # 🔒 Fail fast on wiring mistakes
        if not hasattr(self.pol_post, "predict_proba"):
            raise TypeError(
                "deps.policy_post is not a PostflopPolicyInfer (missing predict_proba). "
                f"Got: {type(self.pol_post)!r}"
            )
        if not hasattr(self._preflop, "predict"):
            raise TypeError(
                "deps.range_pre is not a PreflopPolicy (missing predict). "
                f"Got: {type(self._preflop)!r}"
            )
        if self.eq is not None and not hasattr(self.eq, "predict_proba"):
            raise TypeError(
                "deps.equity does not expose predict_proba(). "
                f"Got: {type(self.eq)!r}"
            )

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

    def _build_postflop_row(self, req: "PolicyRequest") -> dict:
        """
        Build the feature row expected by PostflopPolicyInfer.

        Accepts PolicyRequest with hero_pos/villain_pos (preferred) or ip_pos/oop_pos.
        Deduces ip_pos/oop_pos using a small canonical seat order. Produces:
          - hero_pos, ip_pos, oop_pos, ctx, street (int)
          - board_mask_52 (list/np.array length 52)
          - pot_bb, eff_stack_bb
          - optional board_cluster feature (board_cluster or board_cluster_id)
        """

        # --- canonicalize seats / simple seat order for relative position ---
        def _canon_seat(v):
            if v is None:
                return ""
            return str(v).strip().upper()

        # prefer explicit hero/villain labels; fallback to ip_pos/oop_pos (legacy)
        hero = _canon_seat(getattr(req, "hero_pos", None) or req.raw.get("hero_pos"))
        villain = _canon_seat(getattr(req, "villain_pos", None) or req.raw.get("villain_pos"))

        # minimal seat ordering for deciding who is "left of" whom (6-max style)
        # (higher index = more to the left; if hero index > villain index => hero is IP)
        SEAT_ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

        hero_idx = SEAT_ORDER.index(hero) if hero in SEAT_ORDER else None
        vill_idx = SEAT_ORDER.index(villain) if villain in SEAT_ORDER else None

        # If both known, decide by index; else try raw.actor hint; else default hero as IP
        if hero_idx is not None and vill_idx is not None:
            hero_is_ip = hero_idx > vill_idx
        else:
            raw_actor = (req.raw.get("actor") or "").lower()
            if raw_actor in ("ip", "oop"):
                hero_is_ip = (raw_actor == "ip")
            else:
                # fallback: assume hero is in position (safe default for many HU tests)
                hero_is_ip = True

        # assign ip_pos / oop_pos
        if hero_is_ip:
            ip_pos = hero or ""
            oop_pos = villain or ""
        else:
            ip_pos = villain or ""
            oop_pos = hero or ""

        # --- context & numeric features ---
        ctx = _canon_seat(getattr(req, "ctx", None) or req.raw.get("ctx") or "VS_OPEN")
        try:
            street = int(getattr(req, "street", 1) or 1)
        except Exception:
            street = 1

        pot = float(getattr(req, "pot_bb", None) or req.raw.get("pot_bb") or 0.0)
        stack = float(
            getattr(req, "eff_stack_bb", None)
            or req.raw.get("eff_stack_bb")
            or getattr(req, "stack_bb", 0.0)
            or 0.0
        )

        # --- board / board mask (52-d) ---
        board = getattr(req, "board", None) or req.raw.get("board")
        bm = req.raw.get("board_mask_52")
        if bm is None:
            # make_board_mask_52 should return list/array len 52; provide safe fallback
            try:
                bm = make_board_mask_52(board) if board else [0.0] * 52
            except Exception:
                bm = [0.0] * 52

        # ensure mask is length 52 (pad/truncate)
        try:
            bm_arr = np.asarray(bm, dtype=float).reshape(-1)
        except Exception:
            bm_arr = np.zeros(52, dtype=float)
        if bm_arr.size < 52:
            tmp = np.zeros(52, dtype=float)
            tmp[: bm_arr.size] = bm_arr
            bm_arr = tmp
        elif bm_arr.size > 52:
            bm_arr = bm_arr[:52]

        # --- assemble row with canonical keys that PostflopPolicyInfer expects ---
        row = {
            "hero_pos": hero,
            "ip_pos": ip_pos,
            "oop_pos": oop_pos,
            "ctx": ctx,
            "street": street,
            "board_mask_52": bm_arr.tolist() if not isinstance(bm_arr, list) else bm_arr,
            "pot_bb": float(pot),
            "eff_stack_bb": float(stack),
        }

        # --- optional board-cluster feature (respect PostflopPolicyInfer.board_cluster_feat) ---
        bcf = getattr(self.pol_post, "board_cluster_feat", None)  # "board_cluster" or "board_cluster_id" or None
        if bcf is not None:
            cid = 0
            if self.clusterer is not None and board:
                try:
                    cid = int(self.clusterer.predict_one(board))
                except Exception:
                    cid = 0
            row[bcf] = int(cid)

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

    def _build_fcr_projection(self) -> torch.Tensor:
        """
        Returns P [3, V] that maps deltas over F/C/R into your action vocab logits.
        Row 0 -> FOLD bucket(s), Row 1 -> CALL bucket(s), Row 2 -> RAISE-family (bet/raise/donk/all-in).
        We spread raise-delta uniformly across all raise-like tokens that are legal in the vocab.
        """
        V = len(self.action_vocab)
        P = torch.zeros(3, V, dtype=torch.float32)

        # Indices for exact tokens if they exist
        idx_fold = self._vocab_index.get("FOLD")
        idx_call = self._vocab_index.get("CALL")

        if idx_fold is not None:
            P[0, idx_fold] = 1.0
        if idx_call is not None:
            P[1, idx_call] = 1.0

        # Anything that represents aggression goes into the R row
        raise_like = []
        for i, tok in enumerate(self.action_vocab):
            T = tok.upper()
            if T.startswith("BET_") or T.startswith("RAISE_") or T.startswith("DONK_") or T == "ALLIN":
                raise_like.append(i)

        if raise_like:
            w = 1.0 / float(len(raise_like))
            for i in raise_like:
                P[2, i] = w

        return P

    def _lift_exploit_signal(self, sig_fcr: "np.ndarray | torch.Tensor",
                             device, dtype) -> torch.Tensor:
        """
        sig_fcr: shape (3,) in logit space for [FOLD, CALL, RAISE]
        returns: [1, V] vocab-aligned deltas
        """
        if not hasattr(self, "_P_fcr"):
            self._P_fcr = self._build_fcr_projection().to(device)
        if not isinstance(sig_fcr, torch.Tensor):
            sig_fcr = torch.tensor(sig_fcr, dtype=dtype, device=device)
        sig_fcr = sig_fcr.view(3)
        # [3] @ [3,V] -> [V]
        delta_v = torch.matmul(sig_fcr, self._P_fcr)  # [V]
        return delta_v.view(1, -1)

    def _predict_postflop(self, req_dict: Dict[str, Any]) -> PolicyResponse:
        req = PolicyRequest(**req_dict)

        # 1) Build row from hero/villain; row includes hero_pos, ip_pos, oop_pos, etc.
        row = self._build_postflop_row(req)

        # Decide which side is the hero
        hero_is_ip = (str(row.get("hero_pos", "")).upper() == str(row.get("ip_pos", "")).upper())
        actor = "ip" if hero_is_ip else "oop"

        # 2) Legality masks for the hero side
        mask_ip, mask_oop = self._build_legality_masks(req=req, vocab=self.action_vocab, actor=actor)
        V = len(self.action_vocab)
        if mask_ip is None:  mask_ip = torch.ones(V, dtype=torch.float32)
        if mask_oop is None: mask_oop = torch.ones(V, dtype=torch.float32)
        hero_mask = mask_ip if hero_is_ip else mask_oop

        # 3) Model inference (logits for both sides)
        out = self.pol_post.predict_proba([row], actor=actor, return_logits=True)
        z_ip, z_oop = out["logits_ip"], out["logits_oop"]  # [1, V]
        z_hero = z_ip.clone() if hero_is_ip else z_oop.clone()  # start from hero side

        # 4) Exploit integration (optional)
        exploit_debug = None
        try:
            if self.expl is not None and getattr(req, "villain_id", None) and (self.pop is not None):
                sig3 = self.expl.get_signal_from_request(str(req.villain_id), req, self.pop)  # -> np.ndarray | None
                if sig3 is not None:
                    t_delta = self._lift_exploit_signal(sig3, device=z_hero.device, dtype=z_hero.dtype)  # [1, V]
                    scale = float(self.blend.lambda_expl)
                    z_hero = z_hero + scale * t_delta
                    exploit_debug = {"applied": True, "scale": scale, "sum_abs": float(t_delta.abs().sum().item())}
                else:
                    exploit_debug = {"applied": False, "reason": "insufficient_data_or_small_kl"}
        except Exception as e:
            exploit_debug = {"error": str(e)}

        # 5) Convert to probabilities with masking & blending niceties
        blend = self.blend
        p_hero = self._masked_softmax(z_hero, hero_mask, T=blend.temperature, eps=blend.min_legal_prob)
        p_hero = self._mix_ties_if_close(p_hero, blend.tie_mix_threshold)
        p_hero = self._epsilon_explore(p_hero, blend.epsilon_explore, hero_mask)
        p_hero = self._cap_allins(p_hero, row, hero_mask, blend)

        probs = p_hero[0].tolist()

        # 6) Debug helpers
        bcf = getattr(self.pol_post, "board_cluster_feat", None)  # "board_cluster" or "board_cluster_id" or None
        debug = {
            "hero_is_ip": hero_is_ip,
            "ctx": row.get("ctx"),
            "street": row.get("street"),
            "board_cluster_id": row.get(bcf) if bcf else None,
            "blend_cfg": blend.to_dict(),
            "exploit_debug": exploit_debug,
        }

        return PolicyResponse(
            actions=self.action_vocab,
            probs=probs,
            evs=[0.0] * len(self.action_vocab),
            notes=[f"Postflop policy; hero_is_ip={hero_is_ip}, temp={blend.temperature:.2f}"],
            debug=debug,
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
            equity = None
            if self.eq and req.hero_hand:
                try:
                    out = self.eq.predict([{"street": 0, "hand_id": hand_to_169_label(req.hero_hand)}])
                    if out:
                        p_win, p_tie, p_lose = out[0]
                        equity = {"p_win": float(p_win), "p_tie": float(p_tie), "p_lose": float(p_lose)}
                except Exception:
                    equity = None

            return self._preflop.predict(
                req,
                equity=equity,
                temperature=self.blend.temperature,
                equity_nudge=getattr(self.blend, "equity_nudge_pre", 0.02),  # tiny, configurable
            )

        # === Postflop ===
        return self._predict_postflop(dict(req_dict, street=street))