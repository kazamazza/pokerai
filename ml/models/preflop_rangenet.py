from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

HAND_COUNT = 169

class CatEmbedBlock(nn.Module):
    def __init__(self, cards: Dict[str, int], feature_order: Sequence[str] | None, emb_dim: int = 16):
        super().__init__()

        if cards is None or not isinstance(cards, dict) or not cards:
            raise ValueError(
                "CatEmbedBlock: `cards` is None/empty. "
                "Pass the dataset’s cards dict (e.g., ds.cards()) or load from sidecar."
            )

        # If feature_order is None, default to the cards’ key order
        if feature_order is None:
            feature_order = list(cards.keys())

        try:
            self.feature_order = list(feature_order)
        except TypeError:
            raise ValueError(
                "CatEmbedBlock: `feature_order` is None or not iterable. "
                "Pass ds.feature_order or use the sidecar’s feature_order."
            )

        if not self.feature_order:
            raise ValueError("CatEmbedBlock: empty feature_order after resolution.")

        self.embs = nn.ModuleDict()
        for name in self.feature_order:
            if name not in cards:
                raise KeyError(
                    f"CatEmbedBlock: feature '{name}' missing in `cards`. "
                    f"Available: {list(cards.keys())}"
                )
            c = int(cards[name])
            if c <= 0:
                raise ValueError(f"CatEmbedBlock: cardinality for '{name}' must be > 0 (got {c}).")

            # simple embedding size rule-of-thumb
            d = min(64, max(4, int(round(min(16.0, (c**0.5) * 4)))))
            self.embs[name] = nn.Embedding(c, d)

        self.out_dim = sum(e.embedding_dim for e in self.embs.values())

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        outs = []
        for name in self.feature_order:
            x = x_dict[name]
            # Expect LongTensor of indices; convert if needed and safe
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.dtype not in (torch.int64, torch.long):
                # If the dataset already index-encodes cateogries this should be long.
                # If you see strings here, fix the dataset/loader to encode them.
                x = x.to(torch.long)
            outs.append(self.embs[name](x))
        return torch.cat(outs, dim=-1)


class PreflopRangeNet(nn.Module):
    def __init__(self, cards: Dict[str, int] | None, feature_order: Sequence[str] | None,
                 hidden_dims=(128, 128), dropout=0.1):
        super().__init__()
        # allow defaulting feature order from cards if omitted
        if feature_order is None and cards:
            feature_order = list(cards.keys())
        self.embed = CatEmbedBlock(cards, feature_order)

        layers = []
        in_dim = self.embed.out_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, HAND_COUNT)

    def forward(self, x_dict):
        z = self.embed(x_dict)
        logits = self.head(self.mlp(z))   # [B,169]
        return logits

class RangeNetLit(pl.LightningModule):
    def __init__(
        self,
        cards: dict | None,
        feature_order: Sequence[str] | None,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        # Default feature order from cards if not provided
        if feature_order is None and cards:
            feature_order = list(cards.keys())

        # Fail fast on missing schema
        if not cards or not isinstance(cards, dict):
            raise ValueError("RangeNetLit: `cards` is None/empty or not a dict.")
        if not feature_order:
            raise ValueError("RangeNetLit: `feature_order` is None/empty.")

        # Save hyperparams (but exclude big objects)
        self.save_hyperparameters(ignore=["cards", "feature_order"])

        # Core model
        self.model = PreflopRangeNet(cards, feature_order, hidden_dims, dropout)

        # Optim/loss knobs
        self.lr = float(lr)
        self.wd = float(weight_decay)
        self.ls = float(label_smoothing)

        # (optional) keep for reference
        self.feature_order = list(feature_order)
        self.num_actions = HAND_COUNT

    def _kl_loss(self, logits, y, w):
        # optional label smoothing
        if self.ls > 0:
            y = (1 - self.ls) * y + self.ls / y.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        # KL(y || p) = sum y * (log y - log p)
        kl = torch.sum(y * (torch.log(y + 1e-8) - log_p), dim=-1)
        return torch.sum(w * kl) / (w.sum() + 1e-8)

    def training_step(self, batch, _):
        x, y, w = batch
        logits = self.model(x)
        loss = self._kl_loss(logits, y, w)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y, w = batch
        logits = self.model(x)
        # --- SAME metric as sweep ---
        eps = 1e-12
        ls = float(getattr(self, "ls", 0.0))
        p = torch.softmax(logits, dim=-1)
        y_n = (y + eps) / (y + eps).sum(dim=1, keepdim=True)
        if ls > 0:
            y_n = (1 - ls) * y_n + ls / y_n.size(-1)
        kl = (y_n * (torch.log(y_n + eps) - torch.log(p + eps))).sum(dim=1)  # [B]
        # weight-aware global mean
        val_kl = (kl * w).sum() / (w.sum() + 1e-8)

        self.log("val_kl", val_kl, on_step=False, on_epoch=True, prog_bar=True)
        # keep val_loss = val_kl so ModelCheckpoint can use it transparently
        self.log("val_loss", val_kl, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self, x_dict):
        """Return unnormalized logits [B,169] for eval/inference."""
        return self.model(x_dict)

    # (optional but handy for Lightning predict/test loops)
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, *_ = batch
        return self.forward(x)


def rangenet_preflop_collate_fn(batch: List[Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]]):
    """
    Collate for RangeNet-Preflop:
      batch item = (x_dict, y_soft[169], weight)
    - Stacks numeric tensors
    - Leaves string/categorical lists as-is (model can ignore or handle via embeddings)
    """
    # unzip
    x_list, y_list, w_list = zip(*batch)  # type: ignore

    # merge keys across x_dicts
    all_keys = set()
    for x in x_list:
        all_keys.update(x.keys())

    x_out: Dict[str, Any] = {}
    for k in all_keys:
        vals = [x.get(k) for x in x_list]

        # If already tensors and same shape per-sample → stack
        if isinstance(vals[0], torch.Tensor):
            try:
                x_out[k] = torch.stack(vals, dim=0)
                continue
            except Exception:
                # fall through to numeric conversion attempt
                pass

        # Try numeric to tensor
        if isinstance(vals[0], (int, float)) or (
            hasattr(vals[0], "shape") and getattr(vals[0], "dtype", None) is not None
        ):
            try:
                x_out[k] = torch.as_tensor(vals)
                continue
            except Exception:
                pass

        # Fallback: keep as a Python list (strings/categoricals etc.)
        x_out[k] = vals

    # labels & weights
    y = torch.stack([torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in y_list], dim=0)
    w = torch.stack([torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in w_list], dim=0).view(-1)

    return x_out, y, w