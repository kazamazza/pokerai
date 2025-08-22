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
    def __init__(
        self,
        cards: Dict[str, int],
        emb_dims: Optional[Dict[str, int]] = None,
        hidden_dims: List[int] = [64, 64],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        feature_order: Optional[List[str]] = None,
        use_soft_labels: bool = True,   # <-- lock it in
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cards = dict(cards)
        self.emb_dims = emb_dims or {k: default_emb_dim(v) for k, v in self.cards.items()}
        self.feature_order = feature_order or list(self.cards.keys())
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_soft_labels = bool(use_soft_labels)

        self.emb_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=self.cards[name], embedding_dim=self.emb_dims[name])
            for name in self.feature_order
        })

        in_dim = sum(self.emb_dims[name] for name in self.feature_order)
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(self.dropout)]
            last = h
        layers += [nn.Linear(last, 3)]
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
        x_dict, y, w = batch
        logits = self.forward(x_dict)  # [B, 3]

        if self.use_soft_labels:
            # y: Float[B,3], probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            ce_vec = -(y * log_probs).sum(dim=1)   # soft cross-entropy
            # accuracy via argmax target for logging only
            y_hard = y.argmax(dim=1)
        else:
            # y: Long[B], class indices
            ce_vec = F.cross_entropy(logits, y.long(), reduction='none')
            y_hard = y.long()

        if w is None:
            w = torch.ones_like(ce_vec)
        else:
            w = w.float()
            w = w * (w.numel() / w.sum().clamp_min(1e-8))

        loss = (ce_vec * w).mean()

        pred = logits.argmax(dim=1)
        acc_n = (pred == y_hard).sum().item()
        acc_d = y_hard.numel()

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