
# =========================================
# File: ml/inference/postflop_single/single.py
# =========================================
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml.inference.policy.types import PolicyRequest, PolicyResponse  # noqa: E402
from ml.inference.postflop_single.facing import is_hero_ip, infer_facing_and_size
from ml.inference.postflop_single.fetaures import compute_cluster_id, encode_cats, encode_cont
from ml.inference.postflop_single.legality import mask_root, mask_facing
from ml.models.postflop_policy_side_net import PostflopPolicySideLit
from ml.utils.device import to_device
from ml.utils.sidecar import load_sidecar


class PostflopPolicyInferSingle:
    """
    Clean, single-side inferencer (ROOT or FACING).
    - Loads with sidecar schema (cards, feature_order, cont_features, action_vocab).
    - Derives facing & faced size if not explicit.
    - Builds features and legality strictly from sidecar.
    - Returns temperature-free softmax over masked logits (temperature applied once here).
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

        self.cont_features: List[str] = [str(c) for c in (cont_features or ["board_mask_52","pot_bb","eff_stack_bb"])]

        self.action_vocab: List[str] = list(action_vocab or [])
        if not self.action_vocab:
            raise ValueError("sidecar missing 'action_vocab'")
        self.vocab_size = len(self.action_vocab)

        self.clusterer = clusterer
        self.board_cluster_feat = (
            "board_cluster" if "board_cluster" in self.feature_order
            else ("board_cluster_id" if "board_cluster_id" in self.feature_order else None)
        )

        # sanity: model head matches vocab
        self._assert_model_width_matches_vocab()

    # ---------- loaders ----------
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

        # --- schema from sidecar ---
        feature_order = sc.get("feature_order") or sc.get("cat_feature_order") or []
        card_sizes_raw = sc.get("card_sizes") or sc.get("cards") or {}
        id_maps = sc.get("id_maps") or {}
        cont_features = sc.get("cont_features") or ["board_mask_52", "pot_bb", "eff_stack_bb"]
        action_vocab = sc.get("action_vocab") or []

        if not feature_order or not card_sizes_raw or not action_vocab:
            raise ValueError(f"incomplete sidecar metadata in {sidecar_path}")

        # Normalize card sizes to {str: int}
        card_sizes = {str(k): int(v) for k, v in card_sizes_raw.items()}

        # side is only for logging/meta in your Lit; default safely if not persisted
        lit_side = sc.get("side") or "ip"

        # --- try Lightning load with sidecar-driven init (matches ctor signature) ---
        lit = None
        last_err = None
        try:
            lit = PostflopPolicySideLit.load_from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                map_location=dev,
                side=lit_side,
                card_sizes=card_sizes,
                cat_feature_order=feature_order,
                action_vocab=action_vocab,  # preferred to set head width
            ).eval().to(dev)
        except TypeError as e:
            last_err = e
            # older checkpoints may not accept action_vocab in __init__
            try:
                lit = PostflopPolicySideLit.load_from_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    map_location=dev,
                    side=lit_side,
                    card_sizes=card_sizes,
                    cat_feature_order=feature_order,
                    # no action_vocab
                ).eval().to(dev)
            except Exception as e2:
                last_err = e2
        except Exception as e:
            last_err = e

        # --- final fallback: manual init + strict state load ---
        if lit is None:
            state = torch.load(str(checkpoint_path), map_location=dev)
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
            try:
                lit = PostflopPolicySideLit(
                    side=lit_side,
                    card_sizes=card_sizes,
                    cat_feature_order=feature_order,
                    action_vocab=action_vocab,
                )
            except TypeError:
                lit = PostflopPolicySideLit(
                    side=lit_side,
                    card_sizes=card_sizes,
                    cat_feature_order=feature_order,
                )
            lit.load_state_dict(state_dict, strict=True)
            lit.eval().to(dev)

        # --- sanity: head width must match vocab size if observable ---
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

        # --- wrap & return inferencer ---
        return cls(
            model=lit,
            feature_order=feature_order,
            cards=card_sizes,  # wrapper expects 'cards' (categorical sizes)
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
        # CHANGE HERE: forward_single -> forward
        logits = self.model(x_cat, x_cont)
        if logits.shape[-1] != self.vocab_size:
            raise ValueError(f"model head width {logits.shape[-1]} != action_vocab size {self.vocab_size}")

    # ---------- public utilities ----------
    def infer_facing_and_size(self, req: "PolicyRequest", *, hero_is_ip: bool) -> tuple[bool, Optional[float]]:
        facing, size_frac, _dbg = infer_facing_and_size(req, hero_is_ip=hero_is_ip)
        return facing, size_frac

    # ---------- core predict ----------
    @torch.no_grad()
    def predict(
        self,
        req: "PolicyRequest",
        *,
        actor: str = "ip",
        temperature: float = 1.0,
    ) -> "PolicyResponse":
        # Positions → ip/oop
        hpos = (getattr(req, "hero_pos", "") or "").upper()
        vpos = (getattr(req, "villain_pos", "") or "").upper()
        hero_is_ip = is_hero_ip(hpos, vpos)
        ip_pos = hpos if hero_is_ip else vpos
        oop_pos = vpos if hero_is_ip else hpos

        # Minimal row with sidecar-aligned fields
        street = max(1, min(3, int(getattr(req, "street", 1) or 1)))
        ctx = getattr(req, "ctx", None) or getattr(getattr(req, "raw", {}), "get", lambda *_: None)("ctx") or "VS_OPEN"
        board = getattr(req, "board", None) or ""
        pot_bb = float(getattr(req, "pot_bb", 0.0) or 0.0)
        eff_stack_bb = float(getattr(req, "eff_stack_bb", 0.0) or 0.0)

        # Board cluster (optional feature)
        cluster_id = compute_cluster_id(board, self.clusterer, self.id_maps, self.board_cluster_feat)
        row: Dict[str, Any] = {
            "hero_pos": hpos, "ip_pos": ip_pos, "oop_pos": oop_pos,
            "ctx": ctx, "street": street, "board": board,
            "pot_bb": pot_bb, "eff_stack_bb": eff_stack_bb,
        }
        if cluster_id is not None and self.board_cluster_feat:
            row[self.board_cluster_feat] = cluster_id

        # Facing and bet menu
        bet_menu = None
        if isinstance(getattr(req, "bet_sizes", None), (list, tuple)):
            try: bet_menu = [float(x) for x in getattr(req, "bet_sizes")]
            except Exception: bet_menu = None

        facing, size_frac, _dbg = infer_facing_and_size(req, hero_is_ip=hero_is_ip)
        # If facing, include size_frac; else keep 0.0
        if "size_frac" in self.cont_features:
            row["size_frac"] = float(size_frac) if (facing and size_frac is not None) else 0.0

        # Encode features
        x_cat = encode_cats(self.feature_order, self.cards, self.id_maps, [row], self.device)
        x_cont = encode_cont(self.cont_features, [row], self.device)

        # Forward
        logits = self.model.forward(x_cat, x_cont)  # [1, V]
        is_root_model = ("CHECK" in self.action_vocab)

        # Legal mask
        if is_root_model:
            mask = mask_root(self.action_vocab, actor=actor, bet_menu=bet_menu).view(1, -1).to(logits.device)
        else:
            mask = mask_facing(self.action_vocab).view(1, -1).to(logits.device)

        # Temperature (apply once) + masked softmax
        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits / max(float(temperature), 1e-6), big_neg)
        probs = F.softmax(masked, dim=-1)[0].tolist()

        return PolicyResponse(
            actions=self.action_vocab,
            probs=[float(p) for p in probs],
            evs=[0.0] * len(self.action_vocab),
            notes=[f"postflop single; root={is_root_model}; temp={float(temperature):.3f}"],
            debug={
                "mask_nz": int(mask.sum().item()),
                "hero_is_ip": hero_is_ip,
                "cluster_feat": self.board_cluster_feat,
                "cluster_id": cluster_id,
                "facing": facing,
                "size_frac": float(size_frac) if size_frac is not None else None,
            },
        )

def _has_param(cls, name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(cls.__init__).parameters
    except Exception:
        return False