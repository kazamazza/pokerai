from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Any
from ml.models.constants import HAND_COUNT
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CatEmbedBlock(nn.Module):
    def __init__(self, cards: Dict[str, int], feature_order: Sequence[str] | None, emb_dim: int = 16):
        super().__init__()

        if cards is None or not isinstance(cards, dict) or not cards:
            raise ValueError(
                "CatEmbedBlock: `cards` is None/empty. "
                "Pass the dataset’s cards dict (e.g., ds.cards()) or load from sidecar."
            )

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

            d = min(64, max(4, int(round(min(16.0, (c**0.5) * 4)))))
            self.embs[name] = nn.Embedding(c, d, padding_idx=c-1)

        self.out_dim = sum(e.embedding_dim for e in self.embs.values())

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        outs = []
        B = None
        for name in self.feature_order:
            x = x_dict[name]
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.dtype != torch.long:
                x = x.to(torch.long)
            if x.dim() != 1:
                raise ValueError(f"Feature '{name}' must be 1D [B], got shape {tuple(x.shape)}")
            if B is None:
                B = x.size(0)
            elif x.size(0) != B:
                raise ValueError(f"Batch size mismatch for '{name}': {x.size(0)} vs {B}")
            outs.append(self.embs[name](x))
        return torch.cat(outs, dim=-1)


class PreflopRangeNet(nn.Module):
    def __init__(self, cards: Dict[str, int] | None, feature_order: Sequence[str] | None,
                 hidden_dims=(128, 128), dropout=0.1):
        super().__init__()

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

        if feature_order is None and cards:
            feature_order = list(cards.keys())

        if not cards or not isinstance(cards, dict):
            raise ValueError("RangeNetLit: `cards` is None/empty or not a dict.")
        if not feature_order:
            raise ValueError("RangeNetLit: `feature_order` is None/empty.")

        self.save_hyperparameters(ignore=["cards", "feature_order"])
        self.model = PreflopRangeNet(cards, feature_order, hidden_dims, dropout)
        self.lr = float(lr)
        self.wd = float(weight_decay)
        self.ls = float(label_smoothing)
        self.feature_order = list(feature_order)
        self.num_actions = HAND_COUNT

    def _kl_loss(self, logits, y, w):
        if self.ls > 0:
            y = (1 - self.ls) * y + self.ls / y.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
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
        val_kl = self._kl_loss(logits, y, w)
        self.log("val_kl", val_kl, prog_bar=True)
        self.log("val_loss", val_kl)  # for checkpoint
        self.log("val_loss", val_kl, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        use_sched = True
        if use_sched:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-5
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val_kl",
                    "strict": False,  # don’t crash if monitor missing
                },
            }
        return opt

    def forward(self, x_dict):
        """Return unnormalized logits [B,169] for eval/inference."""
        return self.model(x_dict)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, *_ = batch
        return self.forward(x)


def rangenet_preflop_collate_fn(batch: List[Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]]):
    x_list, y_list, w_list = zip(*batch)

    all_keys = set()
    for x in x_list:
        all_keys.update(x.keys())

    x_out: Dict[str, Any] = {}
    for k in all_keys:
        vals = [x.get(k) for x in x_list]

        if isinstance(vals[0], torch.Tensor):
            try:
                x_out[k] = torch.stack(vals, dim=0)
                continue
            except Exception:
                pass

        if isinstance(vals[0], (int, float)) or (
            hasattr(vals[0], "shape") and getattr(vals[0], "dtype", None) is not None
        ):
            try:
                x_out[k] = torch.as_tensor(vals)
                continue
            except Exception:
                pass

        x_out[k] = vals

    y = torch.stack([torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in y_list], dim=0)
    w = torch.stack([torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in w_list], dim=0).view(-1)

    return x_out, y, w