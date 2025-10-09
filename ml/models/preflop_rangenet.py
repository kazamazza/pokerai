import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Sequence, List, Tuple, Any

HAND_COUNT = 169


class RangeNetLit(pl.LightningModule):
    """
    Unified Preflop Lightning model — no nested wrappers.
    Handles embeddings, MLP, and training logic in one class.
    """

    def __init__(
        self,
        cards: Dict[str, int],
        feature_order: Sequence[str],
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        if not cards or not isinstance(cards, dict):
            raise ValueError("RangeNetLit: `cards` must be a non-empty dict")
        if not feature_order:
            raise ValueError("RangeNetLit: `feature_order` must be provided")

        self.save_hyperparameters(ignore=["cards", "feature_order"])
        self.cards = cards
        self.feature_order = list(feature_order)

        # ---- embeddings ----
        self.emb_layers = nn.ModuleDict()
        for name in self.feature_order:
            c = int(cards[name])
            if c <= 0:
                raise ValueError(f"Feature {name} has invalid cardinality {c}")
            d = min(64, max(4, int(round(min(16.0, (c ** 0.5) * 4)))))  # same heuristic
            self.emb_layers[name] = nn.Embedding(c, d, padding_idx=c - 1)

        emb_out_dim = sum(e.embedding_dim for e in self.emb_layers.values())

        # ---- MLP head ----
        layers = []
        in_dim = emb_out_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, HAND_COUNT)

        # ---- training params ----
        self.lr = lr
        self.wd = weight_decay
        self.ls = label_smoothing

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = []
        for name in self.feature_order:
            x = x_dict[name].long()
            embs.append(self.emb_layers[name](x))
        z = torch.cat(embs, dim=-1)
        logits = self.head(self.mlp(z))  # [B,169]
        return logits

    # -------------------------------------------------------
    # Training / Validation
    # -------------------------------------------------------
    def _kl_loss(self, logits, y, w):
        if self.ls > 0:
            y = (1 - self.ls) * y + self.ls / y.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        kl = torch.sum(y * (torch.log(y + 1e-8) - log_p), dim=-1)
        return torch.sum(w * kl) / (w.sum() + 1e-8)

    def training_step(self, batch, _):
        x, y, w = batch
        logits = self.forward(x)
        loss = self._kl_loss(logits, y, w)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y, w = batch
        logits = self.forward(x)
        val_loss = self._kl_loss(logits, y, w)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_kl", val_loss)
        return val_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2, min_lr=1e-5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "strict": False},
        }

    # For inference / eval
    def predict_step(self, batch, *_):
        x, *_ = batch
        return self.forward(x)