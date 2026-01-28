from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


@dataclass(frozen=True)
class PostflopBatch:
    """
    Contract expected from your dataset collate_fn.
    Keep this tiny + stable.
    """
    x_cat: torch.Tensor         # (B, C_cat) int64
    x_cont: torch.Tensor        # (B, C_cont) float32
    y: torch.Tensor             # (B, A) float32  (target probs)
    weight: torch.Tensor        # (B,) float32
    valid: torch.Tensor         # (B,) bool or 0/1


class PostflopPolicyLit(pl.LightningModule):
    """
    Thin training wrapper around an nn.Module that outputs logits over action_vocab.
    Uses cross-entropy to a *distribution* (soft targets).
    """

    def __init__(
        self,
        *,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
        action_vocab: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.label_smoothing = float(label_smoothing)
        self.action_vocab = list(action_vocab) if action_vocab else None

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        return self.net(x_cat=x_cat, x_cont=x_cont)  # (B, A) logits

    def _loss(self, logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Soft-target cross entropy:
          CE(p, q) = -sum_i p_i log softmax(logits)_i
        """
        logp = F.log_softmax(logits, dim=-1)
        if self.label_smoothing and self.label_smoothing > 0:
            a = y.shape[-1]
            y = (1.0 - self.label_smoothing) * y + self.label_smoothing * (1.0 / float(a))

        per_row = -(y * logp).sum(dim=-1)  # (B,)
        per_row = per_row * weight
        return per_row.mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        b = PostflopBatch(**batch) if isinstance(batch, dict) else batch
        mask = b.valid.bool()
        if mask.sum() == 0:
            loss = torch.zeros([], device=self.device, requires_grad=True)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        logits = self(b.x_cat[mask], b.x_cont[mask])
        loss = self._loss(logits, b.y[mask], b.weight[mask])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        b = PostflopBatch(**batch) if isinstance(batch, dict) else batch
        mask = b.valid.bool()
        if mask.sum() == 0:
            loss = torch.zeros([], device=self.device)
            self.log("val_loss", loss, prog_bar=True)
            return

        logits = self(b.x_cat[mask], b.x_cont[mask])
        loss = self._loss(logits, b.y[mask], b.weight[mask])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt