from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def infer_facing_and_size(self, req: "PolicyRequest", *, hero_is_ip: bool) -> tuple[bool, Optional[float]]:
        facing, size_frac, _dbg = infer_facing_and_size(req, hero_is_ip=hero_is_ip)
        return facing, size_frac

    def infer_root_menu(self, req: "PolicyRequest", *, hero_is_ip: bool) -> tuple[bool, Optional[List[float]]]:
        is_facing, _, _ = infer_facing_and_size(req, hero_is_ip=hero_is_ip)
        is_root = not is_facing
        bet_menu = None
        if is_root and hasattr(req, "bet_sizes") and isinstance(req.bet_sizes, (list, tuple)):
            try:
                bet_menu = [float(x) for x in req.bet_sizes]
            except Exception:
                pass
        return is_root, bet_menu

    @torch.no_grad()
    def predict(
            self,
            req: "PolicyRequest",
            *,
            actor: str = "ip",
            temperature: float = 1.0,
    ) -> "PolicyResponse":
        hpos = (getattr(req, "hero_pos", "") or "").upper()
        vpos = (getattr(req, "villain_pos", "") or "").upper()
        try:
            street = int(getattr(req, "street", 1) or 1)
        except Exception:
            street = 1

        if street == 0:
            hero_is_ip = is_hero_ip(hpos, vpos)
        else:
            if hpos == "BTN" and vpos in ("SB", "BB"):
                hero_is_ip = True
            elif hpos in ("SB", "BB") and vpos in ("SB", "BB"):
                hero_is_ip = (hpos == "BB")
            else:
                POST = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
                try:
                    hero_is_ip = POST.index(hpos) > POST.index(vpos)
                except ValueError:
                    hero_is_ip = True

        # board canonicalization
        board_in = getattr(req, "board", None) or ""
        if isinstance(board_in, (list, tuple)):
            board = "".join(board_in)
        else:
            board = str(board_in).replace(" ", "")

        pot_bb = float(getattr(req, "pot_bb", 0.0) or 0.0)
        eff_stack_bb = float(getattr(req, "eff_stack_bb", 0.0) or 0.0)
        cluster_id = compute_cluster_id(board, self.clusterer, self.id_maps, self.board_cluster_feat)

        row: Dict[str, Any] = {
            "hero_pos": ("IP" if hero_is_ip else "OOP"),
            "ip_pos": "IP",
            "oop_pos": "OOP",
            "ctx": req.ctx,
            "street": "1",  # flop-only categorical
            "board": board,
            "pot_bb": pot_bb,
            "eff_stack_bb": eff_stack_bb
        }

        if self.board_cluster_feat:
            safe_cluster = int(cluster_id) if cluster_id is not None else 0
            row[self.board_cluster_feat] = safe_cluster

        if "stakes_id" in self.feature_order:
            sid_map = self.id_maps.get("stakes_id", {})
            raw = getattr(getattr(req, "stakes", None), "value", None)
            key = str(raw if raw is not None else getattr(req, "stakes", "2"))
            if key not in sid_map and sid_map:
                key = next(iter(sid_map.keys()))
            row["stakes_id"] = key

        bet_menu = None
        if isinstance(getattr(req, "bet_sizes", None), (list, tuple)):
            try:
                bet_menu = [float(x) for x in getattr(req, "bet_sizes")]
            except Exception:
                bet_menu = None

        facing, size_frac, _ = infer_facing_and_size(req, hero_is_ip=hero_is_ip)

        if "stack_bb" in self.cont_features and "stack_bb" not in row:
            row["stack_bb"] = eff_stack_bb
        if "size_frac" in self.cont_features:
            row["size_frac"] = float(size_frac) if (facing and size_frac is not None) else 0.0

        x_cat = encode_cats(self.feature_order, self.cards, self.id_maps, [row], self.device)
        x_cont = encode_cont(self.cont_features, [row], self.device)

        logits = self.model(x_cat, x_cont)
        is_root_model = ("CHECK" in self.action_vocab)

        if is_root_model:
            mask = mask_root(self.action_vocab, actor=actor, bet_menu=bet_menu,ctx=req.ctx).view(1, -1).to(logits.device)
        else:
            mask = mask_facing(self.action_vocab, raise_buckets=req.raise_buckets).view(1, -1).to(logits.device)

        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits / max(float(temperature), 1e-6), big_neg)
        probs = F.softmax(masked, dim=-1)[0].tolist()

        dbg = {
            "mask_nz": int(mask.sum().item()),
            "hero_is_ip": hero_is_ip,
            "cluster_feat": self.board_cluster_feat,
            "board_cluster_id": int(row.get(self.board_cluster_feat)) if self.board_cluster_feat else None,
            "facing": facing,
            "size_frac": float(size_frac) if size_frac is not None else None,
            "encoded_pos": row["hero_pos"],
            "raw_positions": [hpos, vpos],
            "ctx": req.ctx,
        }
        logits_out = logits[0].tolist()

        return PolicyResponse(
            actions=self.action_vocab,
            probs=[float(p) for p in probs],
            evs=[0.0] * len(self.action_vocab),
            notes=[f"postflop single; root={is_root_model}; temp={float(temperature):.3f}"],
            debug=dbg,
            logits=logits_out
        )