from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.inference.policy.helpers import normalize_bet_sizes
from ml.inference.policy.types import PolicyRequest, PolicyResponse  # noqa: E402
from ml.inference.postflop_single.facing import is_hero_ip, infer_facing_and_size
from ml.inference.postflop_single.features import compute_cluster_id, encode_cats, encode_cont
from ml.inference.postflop_single.legality import mask_root, mask_facing
from ml.models.postflop_policy_side_net import PostflopPolicySideLit
from ml.utils.device import to_device
from ml.utils.sidecar import load_sidecar


class PostflopPolicyInferSingle:
    """
    Clean, single-side inferencer (ROOT or FACING).
    Loads sidecar schema; builds features strictly per sidecar; no silent ctx defaults.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        feature_order: Sequence[str],
        cards: Mapping[str, int],
        id_maps: Optional[Mapping[str, Mapping[str, int]]] = None,
        cont_features: Sequence[str] = ("board_mask_52", "pot_bb", "stack_bb"),
        action_vocab: Sequence[str],
        device: Optional[torch.device] = None,
        clusterer: Optional[Any] = None,  # BoardClusterer
    ):
        self.model = model.eval()
        self.device = device or to_device("auto")
        self.model.to(self.device)

        self.feature_order: List[str] = [str(c) for c in (feature_order or [])]
        if not self.feature_order:
            raise ValueError("sidecar missing 'feature_order'")

        self.cards: Dict[str, int] = {str(k): int(v) for k, v in (cards or {}).items()}
        missing = [c for c in self.feature_order if c not in self.cards]
        if missing:
            raise ValueError(f"sidecar/cards missing entries for: {missing}")

        self.id_maps: Dict[str, Dict[str, int]] = {
            str(k): {str(a): int(b) for a, b in (m or {}).items()}
            for k, m in (id_maps or {}).items()
        }

        self.cont_features: List[str] = [str(c) for c in (cont_features or ["board_mask_52", "pot_bb", "eff_stack_bb"])]

        self.action_vocab: List[str] = list(action_vocab or [])
        if not self.action_vocab:
            raise ValueError("sidecar missing 'action_vocab'")
        self.vocab_size = len(self.action_vocab)

        self.clusterer = clusterer
        self.board_cluster_feat = (
            "board_cluster" if "board_cluster" in self.feature_order
            else ("board_cluster_id" if "board_cluster_id" in self.feature_order else None)
        )

        self._assert_model_width_matches_vocab()

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: Union[str, Path],
            sidecar_path: Union[str, Path],
            device: str = "auto",
    ) -> "PostflopPolicyInferSingle":
        import torch

        dev = to_device(device)
        sc = load_sidecar(sidecar_path)

        feature_order = sc.get("feature_order") or sc.get("cat_feature_order") or []
        card_sizes_raw = sc.get("card_sizes") or sc.get("cards") or {}
        id_maps = sc.get("id_maps") or {}
        cont_features = sc.get("cont_features") or ["board_mask_52", "pot_bb", "eff_stack_bb"]
        sidecar_vocab = sc.get("action_vocab") or []

        if not feature_order or not card_sizes_raw or not sidecar_vocab:
            raise ValueError(f"incomplete sidecar metadata in {sidecar_path}")

        card_sizes = {str(k): int(v) for k, v in card_sizes_raw.items()}
        lit_side = sc.get("side") or "ip"

        lit = None
        try:
            lit = PostflopPolicySideLit.load_from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                map_location=dev,
                side=lit_side,
                card_sizes=card_sizes,
                cat_feature_order=feature_order,
                action_vocab=sidecar_vocab,
            ).eval().to(dev)
        except TypeError:
            try:
                lit = PostflopPolicySideLit.load_from_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    map_location=dev,
                    side=lit_side,
                    card_sizes=card_sizes,
                    cat_feature_order=feature_order,
                ).eval().to(dev)
            except Exception:
                state = torch.load(str(checkpoint_path), map_location=dev)
                state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
                try:
                    lit = PostflopPolicySideLit(
                        side=lit_side,
                        card_sizes=card_sizes,
                        cat_feature_order=feature_order,
                        action_vocab=sidecar_vocab,
                    )
                except TypeError:
                    lit = PostflopPolicySideLit(
                        side=lit_side,
                        card_sizes=card_sizes,
                        cat_feature_order=feature_order,
                    )
                lit.load_state_dict(state_dict, strict=True)
                lit.eval().to(dev)

        model_vocab = None
        try:
            mv = getattr(lit, "vocab", None)
            if mv is not None:
                model_vocab = list(mv)
        except Exception:
            pass
        action_vocab = model_vocab if model_vocab else list(sidecar_vocab)

        out_dim = None
        try:
            out_dim = int(getattr(lit, "head").out_features)
        except Exception:
            try:
                out_dim = int(getattr(lit, "vocab_size"))
            except Exception:
                out_dim = None
        if out_dim is not None and out_dim != len(action_vocab):
            raise RuntimeError(
                f"Model head width ({out_dim}) != len(action_vocab) ({len(action_vocab)}). "
                f"Check sidecar/ckpt alignment.\n  sidecar={sidecar_path}\n  ckpt={checkpoint_path}"
            )

        return cls(
            model=lit,
            feature_order=feature_order,
            cards=card_sizes,
            id_maps=id_maps,
            cont_features=cont_features,
            action_vocab=action_vocab,
            device=dev,
            clusterer=None,
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

    # ---------- helpers ----------
    def _assert_model_width_matches_vocab(self) -> None:
        B = 1
        x_cat = {k: torch.zeros(B, dtype=torch.long, device=self.device) for k in self.feature_order}
        x_cont: Dict[str, torch.Tensor] = {}
        for name in self.cont_features:
            if name == "board_mask_52":
                x_cont[name] = torch.zeros(B, 52, device=self.device)
            else:
                x_cont[name] = torch.zeros(B, 1, device=self.device)
        logits = self.model(x_cat, x_cont)
        if logits.shape[-1] != self.vocab_size:
            raise ValueError(f"model head width {logits.shape[-1]} != action_vocab size {self.vocab_size}")


    @torch.no_grad()
    def predict(
            self,
            req: "PolicyRequest",
            *,
            actor: str = "ip",
            temperature: float = 1.0,
    ) -> "PolicyResponse":
        import torch
        import torch.nn.functional as F

        # ---------- 1) Positions / street / who is IP ----------
        hpos = (getattr(req, "hero_pos", "") or "").upper()
        vpos = (getattr(req, "villain_pos", "") or "").upper()
        try:
            street = int(getattr(req, "street", 1) or 1)
        except Exception:
            street = 1

        def _is_hero_ip_pf(h: str, v: str) -> bool:
            # Preflop ordering for IP/OOP resolution
            order = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
            try:
                return order.index(h) > order.index(v)
            except ValueError:
                return True  # safe default

        # Prefer library helper if available
        try:

            hero_is_ip = bool(is_hero_ip(hpos, vpos)) if street > 0 else bool(is_hero_ip(hpos, vpos))
        except Exception:
            if street == 0:
                hero_is_ip = _is_hero_ip_pf(hpos, vpos)
            else:
                if hpos == "BTN" and vpos in ("SB", "BB"):
                    hero_is_ip = True
                elif hpos in ("SB", "BB") and vpos in ("SB", "BB"):
                    hero_is_ip = (hpos == "BB")
                else:
                    hero_is_ip = _is_hero_ip_pf(hpos, vpos)

        # ---------- 2) Board / pot / stacks / cluster ----------
        board_in = getattr(req, "board", None) or ""
        if isinstance(board_in, (list, tuple)):
            board = "".join(board_in)
        else:
            board = str(board_in).replace(" ", "")

        pot_bb = float(getattr(req, "pot_bb", 0.0) or 0.0)
        eff_stack_bb = float(getattr(req, "eff_stack_bb", 0.0) or 0.0)

        def _compute_cluster_id():
            if not self.board_cluster_feat:
                return None
            if self.clusterer is not None:
                try:
                    return int(self.clusterer.predict(board))
                except Exception:
                    pass
            return 0

        cluster_id = _compute_cluster_id()

        # Categorical row (sidecar-driven; keys must match self.feature_order/id_maps)
        row: Dict[str, Any] = {
            "hero_pos": ("IP" if hero_is_ip else "OOP"),
            "ip_pos": "IP",
            "oop_pos": "OOP",
            "ctx": getattr(req, "ctx", None),
            "street": "1",  # flop-only categorical for postflop sidecars
            "board": board,
            # cont added via cont_features below
        }
        if self.board_cluster_feat:
            row[self.board_cluster_feat] = int(cluster_id)

        if "stakes_id" in self.feature_order:
            sid_map = self.id_maps.get("stakes_id", {})
            raw = getattr(getattr(req, "stakes", None), "value", None)
            key = str(raw if raw is not None else getattr(req, "stakes", "2"))
            if key not in sid_map and sid_map:
                key = next(iter(sid_map.keys()))
            row["stakes_id"] = key

        # ---------- 3) Root bet menu & Facing raise buckets ----------
        # Root: prefer req.bet_sizes; else derive from vocab; finally default [33, 66]
        bet_menu: Optional[List[float]] = None
        if isinstance(getattr(req, "bet_sizes", None), (list, tuple)):
            try:
                bet_menu = [float(x) for x in getattr(req, "bet_sizes")]
            except Exception:
                bet_menu = None
        if bet_menu is None and "CHECK" in self.action_vocab:
            try:
                bet_menu = sorted({
                    float(int(tok.split("_", 1)[1]))
                    for tok in self.action_vocab
                    if tok.startswith("BET_") and tok.split("_", 1)[1].isdigit()
                }) or [33.0, 66.0]
            except Exception:
                bet_menu = [33.0, 66.0]
        # Write back so downstream (bundler/promo) sees the same menu
        try:
            if bet_menu is not None:
                req.bet_sizes = list(bet_menu)
        except Exception:
            pass

        # Facing: ensure req.raise_buckets present for mask_facing
        if "CALL" in self.action_vocab:
            if not hasattr(req, "raise_buckets") or not getattr(req, "raise_buckets"):
                try:
                    req.raise_buckets = sorted({
                        int(tok.split("_", 1)[1])
                        for tok in self.action_vocab
                        if tok.startswith("RAISE_") and tok.split("_", 1)[1].isdigit()
                    }) or [150, 200, 300, 400, 500]
                except Exception:
                    req.raise_buckets = [150, 200, 300, 400, 500]

        # ---------- 4) Facing flag & size_frac ----------
        try:
            # Use your central helper if available in the codebase
            from ml.inference.postflop_single.facing import infer_facing_and_size
            facing_flag, size_frac, _ = infer_facing_and_size(req, hero_is_ip=hero_is_ip)
            facing = bool(facing_flag)
        except Exception:
            facing = bool(getattr(req, "facing_bet", False))
            size_frac = getattr(req, "faced_size_frac", None)

        # Continuous features (ONLY those declared in sidecar)
        cont: Dict[str, float] = {}
        if "pot_bb" in self.cont_features:
            cont["pot_bb"] = float(pot_bb)
        if "stack_bb" in self.cont_features:
            cont["stack_bb"] = float(eff_stack_bb)
        if "eff_stack_bb" in self.cont_features:
            cont["eff_stack_bb"] = float(eff_stack_bb)
        if "size_frac" in self.cont_features:
            cont["size_frac"] = float(size_frac) if (facing and size_frac is not None) else 0.0

        # ---------- 5) Encode & forward ----------
        x_cat = encode_cats(self.feature_order, self.cards, self.id_maps, [row], self.device)
        x_cont = encode_cont(self.cont_features, [cont], self.device)
        logits = self.model(x_cat, x_cont)

        # ---------- 6) Mask & softmax ----------
        is_root_model = ("CHECK" in self.action_vocab)
        if is_root_model:
            mask = mask_root(self.action_vocab, actor=actor, bet_menu=bet_menu, ctx=getattr(req, "ctx", None))
        else:
            mask = mask_facing(self.action_vocab, raise_buckets=getattr(req, "raise_buckets", None))
        mask = mask.view(1, -1).to(logits.device)

        temp = max(float(temperature), 1e-6)
        big_neg = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        masked = torch.where(mask > 0.5, logits / temp, big_neg)
        probs = F.softmax(masked, dim=-1)[0].tolist()

        # ---------- 7) Debug ----------
        dbg = {
            "mask_nz": int(mask.sum().item()),
            "hero_is_ip": hero_is_ip,
            "cluster_feat": self.board_cluster_feat,
            "board_cluster_id": int(row.get(self.board_cluster_feat)) if self.board_cluster_feat else None,
            "facing": facing,
            "size_frac": (float(size_frac) if size_frac is not None else None),
            "encoded_pos": row["hero_pos"],
            "raw_positions": [hpos, vpos],
            "ctx": getattr(req, "ctx", None),
            "bet_menu_used": (list(bet_menu) if is_root_model else None),
            "raise_buckets_used": (list(getattr(req, "raise_buckets", [])) if not is_root_model else None),
        }
        logits_out = logits[0].tolist()

        # ---------- 8) Response ----------
        return PolicyResponse(
            actions=self.action_vocab,
            probs=[float(p) for p in probs],
            evs=[0.0] * len(self.action_vocab),  # EVs are supplied by the EV router, not this policy head
            notes=[f"postflop single; root={is_root_model}; temp={float(temperature):.3f}"],
            debug=dbg,
            logits=logits_out,
        )