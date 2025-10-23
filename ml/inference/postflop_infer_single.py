from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import torch
import torch.nn as nn
from ml.features.boards import BoardClusterer
from ml.inference.policy.types import PolicyResponse, PolicyRequest
from ml.inference.postflop_ctx import infer_postflop_ctx
from ml.inference.preflop import _to_device
from ml.models.postflop_policy_side_net import PostflopPolicySideLit
from ml.utils.board_mask import make_board_mask_52
from ml.utils.sidecar import load_sidecar


class PostflopPolicyInferSingle:
    """
    Inference wrapper for a *single-side* postflop policy checkpoint
    (either ROOT or FACING). The side is implicit from the sidecar's
    action_vocab (root tokens contain 'CHECK', facing tokens contain 'FOLD'/'CALL').

    The Lightning module must expose:
        forward_single(x_cat: Dict[str, LongTensor[B]],
                       x_cont: Dict[str, FloatTensor[B,·]]) -> FloatTensor[B, V]
    where V == len(action_vocab).
    """

    # -------------------- lifecycle --------------------

    def __init__(
        self,
        *,
        model: nn.Module,
        feature_order: Sequence[str],
        cards: Mapping[str, int],
        id_maps: Optional[Mapping[str, Mapping[str, int]]] = None,
        cont_features: Sequence[str] = ("board_mask_52", "pot_bb", "eff_stack_bb"),
        action_vocab: Sequence[str],
        device: Optional[torch.device] = None,
        clusterer: Optional[BoardClusterer] = None,
    ):
        self.model = model.eval()
        self.device = device or _to_device("auto")
        self.model.to(self.device)

        # categoricals schema
        self.feature_order: List[str] = [str(c) for c in (feature_order or [])]
        if not self.feature_order:
            raise ValueError("sidecar missing 'feature_order'")

        self.cards: Dict[str, int] = {str(k): int(v) for k, v in (cards or {}).items()}
        missing = [c for c in self.feature_order if c not in self.cards]
        if missing:
            raise ValueError(f"sidecar/cards missing entries for: {missing}")

        # optional token->id maps (categoricals)
        self.id_maps: Dict[str, Dict[str, int]] = {
            str(k): {str(a): int(b) for a, b in (m or {}).items()}
            for k, m in (id_maps or {}).items()
        }

        # continuous schema
        self.cont_features: List[str] = [str(c) for c in (cont_features or ["board_mask_52", "pot_bb", "eff_stack_bb"])]

        # action vocab for this single model
        self.action_vocab: List[str] = list(action_vocab or [])
        if not self.action_vocab:
            raise ValueError("sidecar missing 'action_vocab'")
        self.vocab_size = len(self.action_vocab)

        # optional board clusterer
        self.clusterer = clusterer
        self.board_cluster_feat = (
            "board_cluster"
            if "board_cluster" in self.feature_order
            else ("board_cluster_id" if "board_cluster_id" in self.feature_order else None)
        )

        # sanity: head width
        self._assert_model_width_matches_vocab()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        sidecar_path: Union[str, Path],
        device: str = "auto",
    ) -> "PostflopPolicyInferSingle":
        dev = _to_device(device)
        sc = load_sidecar(sidecar_path)
        # load your single-head Lightning module
        lit = PostflopPolicySideLit.load_from_checkpoint(str(checkpoint_path), map_location=dev).eval().to(dev)

        feature_order = sc.get("feature_order") or sc.get("cat_feature_order") or []
        cards = sc.get("cards") or sc.get("card_sizes") or {}
        id_maps = sc.get("id_maps") or {}
        cont_features = sc.get("cont_features") or ["board_mask_52", "pot_bb", "eff_stack_bb"]
        action_vocab = sc.get("action_vocab") or []

        return cls(
            model=lit,
            feature_order=feature_order,
            cards=cards,
            id_maps=id_maps,
            cont_features=cont_features,
            action_vocab=action_vocab,
            device=dev,
        )

    @classmethod
    def from_dir(
        cls,
        ckpt_dir: Union[str, Path],
        ckpt_name: Optional[str] = None,
        device: str = "auto",
    ) -> "PostflopPolicyInferSingle":
        ckpt_dir = Path(ckpt_dir)
        if ckpt_name is None:
            cands = sorted(ckpt_dir.glob("postflop_policy-*-*.ckpt"))
            checkpoint_path = cands[0] if cands else ckpt_dir / "last.ckpt"
        else:
            checkpoint_path = ckpt_dir / ckpt_name
        sidecar_path = ckpt_dir / "best_sidecar.json"
        return cls.from_checkpoint(checkpoint_path, sidecar_path, device=device)

    @staticmethod
    def _parse_bet_size_from_token(tok: str) -> Optional[float]:
        """
        Accepts a variety of bet tokens, returns fraction in {0.25,0.33,0.50,0.66,0.75,1.00} (best-effort).
        Examples accepted:
          "BET 33", "BET_33", "BET33", "BET 0.33", "DONK_33", "DONK 50", "BET 100%"
        """
        if not tok:
            return None
        s = tok.strip().upper().replace("%", "")
        # keep only last numeric-ish part
        import re
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)$", s.replace("_", " "))
        if not m:
            return None
        val = float(m.group(1))
        # if looks like percent, convert
        if val > 1.5:
            val = val / 100.0
        # snap to our known menu
        CANDS = [0.25, 0.33, 0.50, 0.66, 0.75, 1.00]
        best = min(CANDS, key=lambda c: abs(c - val))
        return best if abs(best - val) <= 0.05 else None  # tolerant

    @staticmethod
    def _is_bet_like(tok: str) -> bool:
        s = (tok or "").strip().upper()
        return s.startswith("BET") or s.startswith("DONK")

    def infer_facing_and_size(self, req: "PolicyRequest", *, hero_is_ip: bool) -> tuple[bool, Optional[float]]:
        """
        Heuristic: if the last action in actions_hist is a bet (by opponent), we are facing.
        We don't require actor tags; we just detect if a bet was made most recently on this street
        and there hasn't been a call/raise from hero after it.
        If facing, we extract the bet size fraction from that last token.
        """
        hist = list(req.actions_hist or [])
        if not hist:
            # fallback to raw payload hints (optional)
            raw = req.raw or {}
            f = raw.get("size_frac")
            p = raw.get("size_pct")
            frac = None
            try:
                if f is not None:
                    frac = float(f)
                elif p is not None:
                    frac = float(p) / 100.0
            except Exception:
                frac = None
            return (False, frac)

        last = str(hist[-1])
        # If last token is a bet-like action, assume hero is *facing* that bet.
        if self._is_bet_like(last):
            frac = self._parse_bet_size_from_token(last)
            return (True, frac)

        # If the last token is CALL/RAISE/CHECK from anyone, assume not currently facing.
        return (False, None)

    def _assert_model_width_matches_vocab(self) -> None:
        B = 1
        x_cat = {k: torch.zeros(B, dtype=torch.long, device=self.device) for k in self.feature_order}
        x_cont = {
            "board_mask_52": torch.zeros(B, 52, device=self.device),
            "pot_bb": torch.zeros(B, 1, device=self.device),
            "eff_stack_bb": torch.zeros(B, 1, device=self.device),
        }
        logits = self.model.forward_single(x_cat, x_cont)  # must exist on your Lit module
        if logits.shape[-1] != self.vocab_size:
            raise ValueError(f"model head width {logits.shape[-1]} != action_vocab size {self.vocab_size}")

    # -------------------- encoding helpers --------------------

    def _encode_cat_value(self, col: str, v: Any) -> int:
        mapping = self.id_maps.get(col) or {}
        if mapping:
            key = "__NA__" if v is None else str(v)
            if key in mapping:
                return mapping[key]
            vocab = int(self.cards.get(col, 0))
            return vocab - 1 if vocab > 0 else 0
        try:
            return int(v) if v is not None else 0
        except Exception:
            return 0

    def _encode_x_cat(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, List[int]] = {c: [] for c in self.feature_order}
        need_cluster = ("board_cluster_id" in self.feature_order) or ("board_cluster" in self.feature_order)

        for r in rows:
            rr = dict(r)
            if need_cluster and self.board_cluster_feat and self.board_cluster_feat not in rr:
                if self.clusterer and rr.get("board"):
                    try:
                        cid = int(self.clusterer.predict_one(rr["board"]))
                        rr[self.board_cluster_feat] = cid
                    except Exception:
                        # leave missing; maps to unknown
                        pass
            for c in self.feature_order:
                out[c].append(self._encode_cat_value(c, rr.get(c)))

        return {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in out.items()}

    def _as_52_mask(self, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            t = v.to(dtype=torch.float32, device=self.device).view(-1)
        else:
            import numpy as np
            t = torch.tensor(np.asarray(v, dtype=np.float32).reshape(-1), dtype=torch.float32, device=self.device)
        if t.numel() < 52:
            pad = torch.zeros(52, dtype=torch.float32, device=self.device)
            pad[: t.numel()] = t
            t = pad
        elif t.numel() > 52:
            t = t[:52]
        return t

    def _encode_x_cont(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(rows)
        x_cont: Dict[str, torch.Tensor] = {}
        if "board_mask_52" in self.cont_features:
            masks = [self._as_52_mask(r.get("board_mask_52", [0.0]*52)) for r in rows]
            x_cont["board_mask_52"] = torch.stack(masks, dim=0)
        for k in ("pot_bb", "eff_stack_bb"):
            if k in self.cont_features:
                vals = [float(r.get(k, 0.0) or 0.0) for r in rows]
                x_cont[k] = torch.tensor(vals, dtype=torch.float32, device=self.device).view(B, 1)
        return x_cont

    def _build_row_dict(self, req: PolicyRequest) -> Dict[str, Any]:
        # street (only used for categoricals if present)
        try:
            street = int(getattr(req, "street", 1) or 1)
        except Exception:
            street = 1
        street = max(1, min(3, street))

        hero = (getattr(req, "hero_pos", "") or "").strip().upper()
        vill = (getattr(req, "villain_pos", "") or "").strip().upper()
        if not hero and vill:
            hero = "BTN" if vill != "BTN" else "CO"
        if not vill and hero:
            vill = "BB" if hero != "BB" else "SB"
        if not hero and not vill:
            hero, vill = "BTN", "BB"

        # derive ip/oop if you have a helper; otherwise leave as provided
        try:
            from ml.inference.policy.types import PolicyRequest as PR  # optional
            hero_is_ip = PR.is_hero_ip(hero, vill)
        except Exception:
            # simple fallback (BTN usually IP in SRP BTN vs BB)
            hero_is_ip = True

        ip_pos = hero if hero_is_ip else vill
        oop_pos = vill if hero_is_ip else hero

        ctx = infer_postflop_ctx(req)

        row: Dict[str, Any] = {
            "hero_pos": hero,
            "ip_pos": ip_pos,
            "oop_pos": oop_pos,
            "ctx": ctx,
            "street": street,
            "board": getattr(req, "board", None) or "",
            "pot_bb": float(getattr(req, "pot_bb", 0.0) or 0.0),
            "eff_stack_bb": float(getattr(req, "eff_stack_bb", 0.0) or 0.0),
        }

        # embed mask if requested
        if "board_mask_52" in self.cont_features and row["board"]:
            row["board_mask_52"] = make_board_mask_52(row["board"])

        # optional pass-through bet sizes for root legality
        raw = getattr(req, "raw", None)
        if isinstance(raw, dict) and "bet_sizes" in raw:
            try:
                row["bet_sizes"] = list(raw["bet_sizes"])
            except Exception:
                pass

            # --- infer facing + size from actions_hist/raw (no payload changes required) ---
        try:
            hero_is_ip = PolicyRequest.is_hero_ip(hero, vill)  # if you have this util
        except Exception:
            hero_is_ip = True

        facing, size_frac = self.infer_facing_and_size(req, hero_is_ip=hero_is_ip)
        row["size_frac"] = float(size_frac) if (facing and size_frac is not None) else 0.0

        # (optional) pass menu for legality gating if present
        if isinstance(req.raw, dict) and "bet_sizes" in req.raw:
            try:
                row["bet_sizes"] = list(req.raw["bet_sizes"])
            except:
                pass

        return row

    def _legal_mask_root(self, actor: str, bet_menu: Optional[Sequence[float]]) -> torch.Tensor:
        names = self.action_vocab
        legal = {"CHECK", "BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100", "DONK_33"}
        if actor.lower() != "oop":
            legal.discard("DONK_33")

        if bet_menu:
            want = set()
            def has(x: float) -> bool:
                return any(abs(float(s) - x) < 1e-3 for s in bet_menu)
            if has(0.25): want.add("BET_25")
            if has(0.33): want.update({"BET_33", "DONK_33"})
            if has(0.50): want.add("BET_50")
            if has(0.66): want.add("BET_66")
            if has(0.75): want.add("BET_75")
            if has(1.00): want.add("BET_100")

            for b in {"BET_25","BET_33","BET_50","BET_66","BET_75","BET_100"}:
                if b not in want:
                    legal.discard(b)
            if "DONK_33" not in want:
                legal.discard("DONK_33")

        m = torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device)
        for i, a in enumerate(names):
            if a in legal:
                m[i] = 1.0
        return m

    def _legal_mask_facing(self) -> torch.Tensor:
        names = self.action_vocab
        legal = {"FOLD", "CALL", "RAISE_150", "RAISE_200", "RAISE_300", "RAISE_400", "RAISE_500", "ALLIN"}
        m = torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device)
        for i, a in enumerate(names):
            if a in legal:
                m[i] = 1.0
        return m

    # -------------------- predict --------------------

    @torch.no_grad()
    def predict(
        self,
        req: "PolicyRequest",
        *,
        actor: str = "ip",
        temperature: float = 1.0,
    ) -> "PolicyResponse":
        """
        High-level API for 1 request. Selects legality based on the model's vocab:
          - if 'CHECK' ∈ vocab ⇒ ROOT model (uses bet_sizes if provided)
          - else ⇒ FACING model
        """
        row = self._build_row_dict(req)
        x_cat = self._encode_x_cat([row])
        x_cont = self._encode_x_cont([row])

        logits = self.model.forward_single(x_cat, x_cont)  # [1, V]
        is_root_model = ("CHECK" in self.action_vocab)

        if is_root_model:
            bet_menu = row.get("bet_sizes", None)
            mask = self._legal_mask_root(actor=actor, bet_menu=bet_menu).view(1, -1)
        else:
            mask = self._legal_mask_facing().view(1, -1)

        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits, big_neg)
        if temperature and temperature != 1.0:
            masked = masked / float(temperature)

        probs = torch.softmax(masked, dim=-1)[0].tolist()
        return PolicyResponse(
            actions=self.action_vocab,
            probs=[float(p) for p in probs],
            evs=[0.0] * len(self.action_vocab),
            notes=[f"postflop single-side; root={is_root_model}; temp={temperature:.3f}"],
            debug={"mask_nz": int(mask.sum().item())},
        )