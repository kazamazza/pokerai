from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ml.models.postflop_policy_net import PostflopPolicyNetLit, ACTION_VOCAB, VOCAB_SIZE
from ml.utils.device import to_device            # your helper ("auto" -> cuda/cpu)
from ml.utils.sidecar import load_sidecar


class PostflopPolicyInfer:
    """
    Lightweight inference wrapper for PostflopPolicyNetLit.

    Expects sidecar JSON fields:
      - feature_order: list[str]  (categorical feature order used in training)
      - cards: dict[str,int]      (category cardinalities)
      - id_maps (or encoders): {col: {token_str: idx}}

    predict_proba(rows): rows are dicts with keys:
      CATEGORICAL (must match feature_order): hero_pos, ip_pos, oop_pos, ctx, street, ...
      CONTINUOUS: pot_bb, eff_stack_bb
      OPTIONAL: board_mask_52 (len 52, 0/1), actor ("ip"|"oop") to choose head
    """

    def __init__(
        self,
        *,
        model: PostflopPolicyNetLit,
        feature_order: Sequence[str],
        id_maps: Dict[str, Dict[str, int]],
        cards: Dict[str, int],
        device: torch.device | None = None,
    ):
        self.model = model.eval()
        self.feature_order = list(feature_order)
        # normalize id_maps to str keys / int vals
        self.id_maps = {
            col: {str(k): int(v) for k, v in (mapping or {}).items()}
            for col, mapping in (id_maps or {}).items()
        }
        self.cards = {k: int(v) for k, v in (cards or {}).items()}
        self.device = device or to_device("auto")
        self.model.to(self.device)

    # -------- construction --------
    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, sidecar_path: str | Path, device: str | torch.device = "auto"):
        dev = to_device(device)
        sc = load_sidecar(sidecar_path)  # must contain feature_order/cards/(id_maps|encoders)
        model = PostflopPolicyNetLit.load_from_checkpoint(str(ckpt_path), map_location=dev)
        model.eval().to(dev)
        id_maps = sc.get("id_maps") or sc.get("encoders") or {}
        return cls(
            model=model,
            feature_order=sc["feature_order"],
            id_maps=id_maps,
            cards=sc["cards"],
            device=dev,
        )

    # -------- encoding --------
    def _unknown_idx(self, col: str) -> int:
        C = int(self.cards.get(col, 1))
        return max(C - 1, 0)

    def _encode_column(self, col: str, values: Sequence[Any]) -> torch.Tensor:
        enc = self.id_maps.get(col, {})
        card = int(self.cards.get(col, max(len(enc), 1)))
        unk = min(self._unknown_idx(col), card - 1)
        ids: List[int] = []
        for v in values:
            key = "__NA__" if v is None else str(v)
            idx = int(enc.get(key, unk))
            if idx >= card:
                idx = card - 1
            ids.append(idx)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _encode_batch(self, rows: Sequence[Mapping[str, Any]]) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        if not rows:
            raise ValueError("predict_proba: empty rows")

        # categorical in sidecar-defined order
        cols: Dict[str, List[Any]] = {k: [] for k in self.feature_order}
        for r in rows:
            for k in self.feature_order:
                if k not in r:
                    raise KeyError(f"Missing categorical feature '{k}' in row: {r.keys()}")
                cols[k].append(r[k])
        x_cat = {k: self._encode_column(k, v) for k, v in cols.items()}

        # continuous
        B = len(rows)
        def _colf(name: str, default: float = 0.0) -> torch.Tensor:
            vals = [float(r.get(name, default) or 0.0) for r in rows]
            return torch.tensor(vals, dtype=torch.float32, device=self.device).view(B, 1)

        pot_bb = _colf("pot_bb", 0.0)
        eff_stack_bb = _colf("eff_stack_bb", float(rows[0].get("stack_bb", 0.0)))  # safe fallback

        bm = []
        for r in rows:
            v = r.get("board_mask_52", None)
            if v is None:
                bm.append(np.zeros(52, dtype=np.float32))
            else:
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                if arr.size != 52:
                    tmp = np.zeros(52, dtype=np.float32)
                    tmp[: min(52, arr.size)] = arr[: min(52, arr.size)]
                    arr = tmp
                bm.append(arr)
        board_mask_52 = torch.tensor(np.stack(bm, axis=0), dtype=torch.float32, device=self.device)

        x_cont = {
            "pot_bb": pot_bb,
            "eff_stack_bb": eff_stack_bb,
            "board_mask_52": board_mask_52,
        }

        # actor head selector
        actors = [str(r.get("actor", "ip")).lower() for r in rows]
        is_ip = torch.tensor([1.0 if a == "ip" else 0.0 for a in actors], dtype=torch.float32, device=self.device).view(B, 1)

        return x_cat, x_cont, is_ip

    # -------- inference --------
    @torch.no_grad()
    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """
        Returns np.ndarray [B, VOCAB_SIZE] — probs over ACTION_VOCAB, picking the correct head per row.
        """
        if not rows:
            return np.empty((0, VOCAB_SIZE), dtype=np.float32)

        x_cat, x_cont, is_ip = self._encode_batch(rows)
        li, lo = self.model(x_cat, x_cont)           # [B,V], [B,V]
        # select logits per row: ip -> li, oop -> lo
        logits = is_ip * li + (1.0 - is_ip) * lo     # [B,V]
        probs = F.softmax(logits, dim=-1)            # [B,V]
        return probs.detach().cpu().numpy()

    @torch.no_grad()
    def predict(self, rows: Sequence[Mapping[str, Any]]) -> List[List[float]]:
        return self.predict_proba(rows).tolist()