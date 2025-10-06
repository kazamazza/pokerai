from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from ml.models.policy_consts import ACTION_VOCAB as DEFAULT_VOCAB
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
    Inference wrapper for PostflopPolicyLit.
    - Loads a Lightning checkpoint (no nested .model).
    - Uses sidecar to align categorical feature order, cards (vocab sizes),
      optional id_maps, continuous features, and action vocab.
    - Provides batched predict_proba with per-side masks and temperature.
    """

    def __init__(
        self,
        *,
        model: PostflopPolicyLit,
        feature_order: Sequence[str],
        cards: Mapping[str, int],
        id_maps: Optional[Mapping[str, Mapping[str, int]]] = None,
        cont_features: Sequence[str] = ("board_mask_52", "pot_bb", "eff_stack_bb"),
        action_vocab: Sequence[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.device = device or _to_device("auto")
        self.model.to(self.device)

        # Categorical setup
        self.feature_order = [str(c) for c in (feature_order or [])]
        if not self.feature_order:
            raise ValueError("Sidecar missing 'feature_order' (categorical feature order).")
        self.cards = {str(k): int(v) for k, v in (cards or {}).items()}
        missing = [c for c in self.feature_order if c not in self.cards]
        if missing:
            raise ValueError(f"Sidecar/cards missing entries for categorical features: {missing}")

        # Optional id maps (string -> int ids used in training)
        self.id_maps = {str(k): {str(a): int(b) for a, b in (m or {}).items()}
                        for k, m in (id_maps or {}).items()}

        # Continuous features (in forward we always provide board_mask_52, pot_bb, eff_stack_bb)
        self.cont_features = [str(c) for c in (cont_features or ["board_mask_52", "pot_bb", "eff_stack_bb"])]

        # Action vocab

        self.action_vocab = list(action_vocab) if action_vocab is not None else list(DEFAULT_VOCAB)
        self.vocab_size = len(self.action_vocab)

        # Sanity: verify head sizes match vocab
        self._assert_model_heads_match_vocab()

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: Union[str, Path],
            sidecar_path: Union[str, Path],
            device: DeviceLike = "auto",
    ) -> "PostflopPolicyInfer":
        """
        Load a Lightning checkpoint + sidecar.
        Be tolerant to either schema:
          - feature_order + cards
          - cat_feature_order + card_sizes
        We do NOT modify the global load_sidecar(); we just fall back locally.
        """
        dev = _to_device(device)

        # Try the strict global loader first; if it rejects, fall back to raw JSON.
        try:
            sc_raw = load_sidecar(sidecar_path)  # may raise if keys differ
        except Exception:
            import json
            p = Path(sidecar_path)
            if not p.exists():
                raise FileNotFoundError(f"Sidecar missing: {p}")
            try:
                sc_raw = json.loads(p.read_text())
            except Exception as e:
                raise ValueError(f"Failed to parse sidecar JSON at {p}: {e}")

        # Normalize keys (support both naming variants)
        feature_order = sc_raw.get("feature_order") or sc_raw.get("cat_feature_order") or []
        cards_raw = sc_raw.get("cards") or sc_raw.get("card_sizes") or {}

        if not feature_order or not isinstance(feature_order, list):
            raise ValueError(
                f"Sidecar {sidecar_path} missing 'feature_order'/'cat_feature_order' list"
            )
        if not isinstance(cards_raw, dict) or not cards_raw:
            raise ValueError(
                f"Sidecar {sidecar_path} missing 'cards'/'card_sizes' dict"
            )

        # Cast to the shapes the model expects
        cards = {str(k): int(v) for k, v in cards_raw.items()}
        id_maps = sc_raw.get("id_maps") or {}
        cont_features = sc_raw.get("cont_features") or ["board_mask_52", "pot_bb", "eff_stack_bb"]

        # Action vocab: prefer sidecar, else model default
        try:
            from ml.models.postflop_policy_net import ACTION_VOCAB as DEFAULT_VOCAB
        except Exception:
            DEFAULT_VOCAB = None
        action_vocab = sc_raw.get("action_vocab") or DEFAULT_VOCAB

        # Load Lightning model
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

    @classmethod
    def from_dir(
        cls,
        ckpt_dir: Union[str, Path],
        ckpt_name: Optional[str] = None,
        device: DeviceLike = "auto",
    ) -> "PostflopPolicyInfer":
        """Convenience: load {dir}/best-or-last.ckpt + {dir}/sidecar.json"""
        ckpt_dir = Path(ckpt_dir)
        if ckpt_name is None:
            # Try best (pattern), else last.ckpt
            cands = sorted(ckpt_dir.glob("postflop_policy-*-*.ckpt"))
            checkpoint_path = cands[0] if cands else ckpt_dir / "last.ckpt"
        else:
            checkpoint_path = ckpt_dir / ckpt_name
        sidecar_path = ckpt_dir / "sidecar.json"
        return cls.from_checkpoint(checkpoint_path, sidecar_path, device=device)

    # -------- private helpers --------

    def _assert_model_heads_match_vocab(self) -> None:
        """Quick forward with zeros to verify head widths == action vocab."""
        # minimal fake batch B=1
        x_cat = {k: torch.zeros(1, dtype=torch.long, device=self.device) for k in self.feature_order}
        brd = torch.zeros(1, 52, device=self.device)
        x_cont = {
            "board_mask_52": brd,
            "pot_bb": torch.zeros(1, 1, device=self.device),
            "eff_stack_bb": torch.zeros(1, 1, device=self.device),
        }
        li, lo = self.model(x_cat, x_cont)
        if li.shape[-1] != self.vocab_size or lo.shape[-1] != self.vocab_size:
            raise ValueError(
                f"Model head width != action_vocab size: "
                f"ip={li.shape[-1]}, oop={lo.shape[-1]}, vocab={self.vocab_size}"
            )

    def _encode_cat_value(self, col: str, v: Any) -> int:
        """Encode one categorical value using id_maps if present; else int-cast fallback."""
        mapping = self.id_maps.get(col) or {}
        if mapping:
            key = "__NA__" if v is None else str(v)
            # unknown categories map to last bucket (stable fallback)
            return mapping.get(key, max(mapping.values()) if mapping else 0)
        try:
            return int(v) if v is not None else 0
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
            t = v.to(dtype=torch.float32, device=self.device).view(-1)
        else:
            import numpy as np
            arr = np.asarray(v, dtype=np.float32)  # force float32
            t = torch.tensor(arr, dtype=torch.float32, device=self.device).view(-1)
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
            masks = [self._as_52_mask(r.get("board_mask_52", [0] * 52)) for r in rows]
            x_cont["board_mask_52"] = torch.stack(masks, dim=0)  # already float32/device
        for k in ("pot_bb", "eff_stack_bb"):
            if k in self.cont_features:
                vals = [float(r.get(k, 0.0) or 0.0) for r in rows]
                x_cont[k] = torch.tensor(vals, dtype=torch.float32, device=self.device).view(B, 1)
        return x_cont

    def _role_masks(
        self,
        actor: str,
        B: int,
        mask_ip: Optional[torch.Tensor],
        mask_oop: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _norm(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if m is None:
                return None
            m = m.to(self.device).float()
            if m.dim() == 1:  # [V] -> [B,V]
                m = m.view(1, -1).repeat(B, 1)
            return m

        mi = _norm(mask_ip)
        mo = _norm(mask_oop)

        if mi is None or mo is None:
            ones = torch.ones(B, self.vocab_size, dtype=torch.float32, device=self.device)
            zeros = torch.zeros_like(ones)
            if (actor or "ip").lower() == "ip":
                mi = mi if mi is not None else ones
                mo = mo if mo is not None else zeros
            else:
                mi = mi if mi is not None else zeros
                mo = mo if mo is not None else ones
        return mi, mo

    # -------- public API --------

    @torch.no_grad()
    def predict_proba(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        actor: str = "ip",
        mask_ip: Optional[torch.Tensor] = None,   # [V] or [B,V]
        mask_oop: Optional[torch.Tensor] = None,  # [V] or [B,V]
        temperature: float = 1.0,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict of:
          - probs_ip  : [B,V]
          - probs_oop : [B,V]
          - (optional) logits_ip/logits_oop if return_logits=True
        """
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

        li, lo = self.model(x_cat, x_cont)  # [B,V], [B,V]
        if li.shape[-1] != self.vocab_size or lo.shape[-1] != self.vocab_size:
            raise RuntimeError(f"Model head width changed at runtime: ip={li.shape[-1]}, oop={lo.shape[-1]}, vocab={self.vocab_size}")

        mi, mo = self._role_masks(actor, B, mask_ip, mask_oop)

        big_neg = torch.finfo(li.dtype).min / 4
        li_masked = torch.where(mi > 0.5, li, big_neg)
        lo_masked = torch.where(mo > 0.5, lo, big_neg)

        if temperature and temperature != 1.0:
            t = float(temperature)
            li_masked = li_masked / t
            lo_masked = lo_masked / t

        probs_ip = F.softmax(li_masked, dim=-1)
        probs_oop = F.softmax(lo_masked, dim=-1)

        out: Dict[str, torch.Tensor] = {"probs_ip": probs_ip, "probs_oop": probs_oop}
        if return_logits:
            out["logits_ip"], out["logits_oop"] = li, lo
        return out

    @torch.no_grad()
    def predict_one(
        self,
        row: Mapping[str, Any],
        *,
        actor: str = "ip",
        mask_ip: Optional[torch.Tensor] = None,
        mask_oop: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, List[float]]:
        res = self.predict_proba(
            [row],
            actor=actor,
            mask_ip=mask_ip,
            mask_oop=mask_oop,
            temperature=temperature,
            return_logits=False,
        )
        return {
            "actions": self.action_vocab,
            "probs_ip":  res["probs_ip"][0].tolist(),
            "probs_oop": res["probs_oop"][0].tolist(),
        }