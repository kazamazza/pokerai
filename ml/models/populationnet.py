from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# actions: 0=FOLD, 1=CALL, 2=RAISE
N_ACTIONS = 3

def default_emb_dim(card: int) -> int:
    # small, safe rule: grows sublinearly with cardinality
    return int(min(32, max(4, round(1.6 * math.sqrt(card)))))

class PopulationNetLit(pl.LightningModule):
    """
    PopulationNet: categorical embeddings -> MLP -> action logits (3 classes).
    Uses per-sample weights if provided: loss is averaged with weights.
    """
    def __init__(
        self,
        # cardinalities of each categorical feature
        cards: Dict[str, int],                     # e.g. {"stakes_id": 4, "street_id": 4, "ctx_id": 16, "hero_pos_id": 6, "villain_pos_id": 6}
        emb_dims: Optional[Dict[str, int]] = None, # override embedding dims per feature (optional)
        hidden_dims: List[int] = [64, 64],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        feature_order: Optional[List[str]] = None  # order in which we read X columns
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cards = dict(cards)
        self.emb_dims = emb_dims or {k: default_emb_dim(v) for k, v in self.cards.items()}
        self.feature_order = feature_order or list(self.cards.keys())
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        # Build one embedding table per categorical feature
        self.emb_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=self.cards[name], embedding_dim=self.emb_dims[name])
            for name in self.feature_order
        })

        # MLP head
        in_dim = sum(self.emb_dims[name] for name in self.feature_order)
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(self.dropout)]
            last = h
        layers += [nn.Linear(last, N_ACTIONS)]
        self.mlp = nn.Sequential(*layers)

        # simple accuracy buffer
        self.train_acc_n = 0
        self.train_acc_d = 0
        self.val_acc_n = 0
        self.val_acc_d = 0

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x is a dict of {feature_name: tensor[batch]} of integer IDs.
        Returns logits [batch, 3].
        """
        embs = []
        for name in self.feature_order:
            # ensure long dtype for embedding lookup
            embs.append(self.emb_layers[name](x[name].long()))
        h = torch.cat(embs, dim=-1)  # [B, sum(emb_dims)]
        logits = self.mlp(h)         # [B, 3]
        return logits

    def _step(self, batch, stage: str):
        """
        batch is a tuple: (x_dict, y, w)
          - x_dict: {feature_name: tensor[batch]}
          - y: tensor[batch] with values in {0,1,2}
          - w: tensor[batch] float weights (or None / ones)
        """
        x_dict, y, w = batch
        logits = self.forward(x_dict)
        # per-sample CE
        ce = F.cross_entropy(logits, y.long(), reduction='none')
        if w is None:
            loss = ce.mean()
            w = torch.ones_like(ce)
        else:
            # normalize weights so they sum to batch size (stable scaling)
            w = w.float()
            w = w * (w.numel() / (w.sum().clamp_min(1e-8)))
            loss = (ce * w).mean()

        # accuracy
        pred = logits.argmax(dim=1)
        acc_n = (pred == y).sum().item()
        acc_d = y.numel()

        if stage == "train":
            self.train_acc_n += acc_n
            self.train_acc_d += acc_d
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            self.val_acc_n += acc_n
            self.val_acc_d += acc_d
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_train_epoch_end(self):
        if self.train_acc_d > 0:
            acc = self.train_acc_n / self.train_acc_d
            self.log("train_acc_epoch", acc, prog_bar=True)
        self.train_acc_n = 0; self.train_acc_d = 0

    def on_validation_epoch_end(self):
        if self.val_acc_d > 0:
            acc = self.val_acc_n / self.val_acc_d
            self.log("val_acc_epoch", acc, prog_bar=True)
        self.val_acc_n = 0; self.val_acc_d = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)