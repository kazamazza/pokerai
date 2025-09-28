from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F

from ml.models.policy_consts import ACTION_VOCAB
from ml.models.postflop_policy_net import PostflopPolicyLit
from ml.utils.sidecar import load_sidecar

DeviceLike = Union[str, torch.device]

def _to_device(dev: DeviceLike = "auto") -> torch.device:
    if isinstance(dev, torch.device):
        return dev
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

class PostflopPolicyInfer:
    """
    Inference for PostflopPolicyLit (Lightning IS the model).
    - Loads Lightning checkpoint exactly (no inner .model).
    - Uses sidecar to know categorical feature order, vocab sizes, optional id_maps,
      and continuous feature names.
    """

    def __init__(
        self,
        *,
        model: PostflopPolicyLit,
        feature_order: Sequence[str],
        cards: Mapping[str, int],
        id_maps: Optional[Mapping[str, Mapping[str, int]]] = None,
        cont_features: Sequence[str] = ("board_mask_52", "pot_bb", "eff_stack_bb"),
        action_vocab: Sequence[str] = ACTION_VOCAB,
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.device = device or _to_device("auto")
        self.model.to(self.device)

        self.feature_order = list(feature_order)
        self.cards = {str(k): int(v) for k, v in cards.items()}
        self.id_maps = {k: {str(a): int(b) for a, b in (m or {}).items()} for k, m in (id_maps or {}).items()}
        self.cont_features = list(cont_features)

        self.action_vocab = list(action_vocab)
        self.vocab_size = len(self.action_vocab)

        # sanity
        missing = [c for c in self.feature_order if c not in self.cards]
        if missing:
            raise ValueError(f"Sidecar/cards missing entries for categorical features: {missing}")

    # -------- constructors --------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        sidecar_path: Union[str, Path],
        device: DeviceLike = "auto",
    ) -> "PostflopPolicyInfer":
        dev = _to_device(device)

        sc = load_sidecar(sidecar_path)  # must contain feature_order + cards; optional id_maps, cont_features, action_vocab
        feature_order = sc.get("feature_order") or sc.get("cat_feature_order") or []
        cards = sc.get("cards") or sc.get("card_sizes") or {}
        id_maps = sc.get("id_maps") or {}
        cont_features = sc.get("cont_features") or ["board_mask_52", "pot_bb", "eff_stack_bb"]
        action_vocab = sc.get("action_vocab") or ACTION_VOCAB

        lit = PostflopPolicyLit.load_from_checkpoint(str(checkpoint_path), map_location=dev)
        lit.eval().to(dev)

        return cls(
            model=lit,
            feature_order=feature_order,
            cards=cards,
            id_maps=id_maps,
            cont_features=cont_features,
            action_vocab=action_vocab,
            device=dev,
        )

    # -------- encoding helpers --------

    def _encode_cat_value(self, col: str, v: Any) -> int:
        mapping = self.id_maps.get(col) or {}
        if mapping:
            key = str(v) if v is not None else "__NA__"
            return mapping.get(key, max(mapping.values()))  # unknown → last bucket
        try:
            return int(v)
        except Exception:
            return 0

    def _encode_x_cat(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, List[int]] = {c: [] for c in self.feature_order}
        for r in rows:
            for c in self.feature_order:
                if c not in r:
                    raise KeyError(f"Missing categorical feature '{c}' in row keys {list(r.keys())}")
                out[c].append(self._encode_cat_value(c, r[c]))
        return {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in out.items()}

    def _as_52_mask(self, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            t = v.float().view(-1)
        else:
            import numpy as np
            t = torch.tensor(np.asarray(v, dtype=float)).view(-1)
        if t.numel() < 52:
            pad = torch.zeros(52, dtype=torch.float32)
            pad[: t.numel()] = t
            t = pad
        elif t.numel() > 52:
            t = t[:52]
        return t

    def _encode_x_cont(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(rows)
        x_cont: Dict[str, torch.Tensor] = {}
        if "board_mask_52" in self.cont_features:
            masks = [self._as_52_mask(r.get("board_mask_52", [0]*52)) for r in rows]
            x_cont["board_mask_52"] = torch.stack(masks, dim=0).to(self.device)
        for k in ("pot_bb", "eff_stack_bb"):
            if k in self.cont_features:
                vals = [float(r.get(k, 0.0) or 0.0) for r in rows]
                x_cont[k] = torch.tensor(vals, dtype=torch.float32, device=self.device).view(B, 1)
        return x_cont

    def _role_masks(
        self, actor: str, B: int, mask_override: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_override is not None:
            m = mask_override.to(self.device).float()
            if m.dim() == 1:
                m = m.view(1, -1).repeat(B, 1)
            return m, m
        ones = torch.ones(B, self.vocab_size, dtype=torch.float32, device=self.device)
        zeros = torch.zeros_like(ones)
        return (ones, zeros) if (actor or "ip").lower() == "ip" else (zeros, ones)

    # -------- public API --------

    @torch.no_grad()
    def predict_proba(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        actor: str = "ip",
        mask: Optional[torch.Tensor] = None,  # optional [B,V] or [V]
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if not rows:
            B = 0
            empty = torch.empty(B, self.vocab_size, device=self.device)
            out = {"probs_ip": empty, "probs_oop": empty}
            if return_logits:
                out["logits_ip"], out["logits_oop"] = empty, empty
            return out

        B = len(rows)
        x_cat = self._encode_x_cat(rows)
        x_cont = self._encode_x_cont(rows)

        # Call the Lightning model directly
        logits_ip, logits_oop = self.model(x_cat, x_cont)

        m_ip, m_oop = self._role_masks(actor, B, mask_override=mask)

        big_neg = torch.finfo(logits_ip.dtype).min / 4
        li_masked = torch.where(m_ip > 0.5, logits_ip, big_neg)
        lo_masked = torch.where(m_oop > 0.5, logits_oop, big_neg)

        probs_ip = F.softmax(li_masked, dim=-1)
        probs_oop = F.softmax(lo_masked, dim=-1)

        out: Dict[str, torch.Tensor] = {"probs_ip": probs_ip, "probs_oop": probs_oop}
        if return_logits:
            out["logits_ip"], out["logits_oop"] = logits_ip, logits_oop
        return out

    @torch.no_grad()
    def predict_one(
        self,
        row: Mapping[str, Any],
        *,
        actor: str = "ip",
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[float]]:
        res = self.predict_proba([row], actor=actor, mask=mask, return_logits=False)
        return {
            "actions": self.action_vocab,
            "probs_ip":  res["probs_ip"][0].tolist(),
            "probs_oop": res["probs_oop"][0].tolist(),
        }