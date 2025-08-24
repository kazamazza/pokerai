from typing import Dict, List, Optional, Sequence, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

def default_emb_dim(card: int) -> int:
    # sublinear, small-safe rule
    return int(min(32, max(4, round(1.6 * math.sqrt(card)))))

class EquityNetLit(pl.LightningModule):
    """
    EquityNet: (categorical embeddings + numeric features) -> MLP -> equity prediction.

    Output modes:
      - scalar: predicts a single equity in [0,1] (uses BCEWithLogitsLoss)
      - triplet: predicts logits for [win, tie, lose] (uses KLDiv with soft labels or CE with hard)

    Expected batch (from DataLoader):
      (x_cat: Dict[str, LongTensor[B]],
       x_num: FloatTensor[B, D],
       y:     FloatTensor[B] or FloatTensor[B,3] or LongTensor[B],  # depends on output_mode & label type
       w:     FloatTensor[B])  # optional sample weights
    """

    def __init__(
        self,
        # categorical feature setup
        cards: Dict[str, int],                     # e.g. {"hero_pos_id":6, "opener_action_id":4, "hand_id":169}
        cat_order: Optional[List[str]] = None,     # order to read from x_cat
        emb_dims: Optional[Dict[str, int]] = None, # per-feature embedding dims (optional)

        # numeric feature setup
        num_features: Optional[Sequence[str]] = None,  # purely informational; collate provides x_num
        num_in_dim: int = 0,                            # dimension of x_num

        # head / training
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        output_mode: str = "scalar",  # "scalar" or "triplet"
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Inputs config ----
        self.cards = dict(cards)
        self.cat_order = cat_order or list(self.cards.keys())
        self.emb_dims = emb_dims or {k: default_emb_dim(v) for k, v in self.cards.items()}
        self.num_in_dim = int(num_in_dim)
        self.output_mode = output_mode

        # ---- Embeddings for categorical features ----
        self.emb_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=self.cards[name], embedding_dim=self.emb_dims[name])
            for name in self.cat_order
        })

        # ---- Optional numeric pre-projection (kept simple for now) ----
        # You can replace with LayerNorm or a small MLP if you want.
        self.num_proj = nn.Identity()
        proj_num_dim = self.num_in_dim

        # ---- MLP head ----
        emb_total = sum(self.emb_dims[name] for name in self.cat_order)
        in_dim = emb_total + proj_num_dim

        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h

        if self.output_mode == "scalar":
            out_dim = 1
        elif self.output_mode == "triplet":
            out_dim = 3
        else:
            raise ValueError("output_mode must be 'scalar' or 'triplet'")

        layers += [nn.Linear(last, out_dim)]
        self.mlp = nn.Sequential(*layers)

        # simple running metrics
        self.train_n = 0.0
        self.train_loss_sum = 0.0
        self.val_n = 0.0
        self.val_loss_sum = 0.0

    # ---- Forward ----
    def forward(self, x_cat: Dict[str, torch.Tensor], x_num: torch.Tensor) -> torch.Tensor:
        """
        x_cat: dict of {feature_name: LongTensor[B]}
        x_num: FloatTensor[B, D] (D may be 0; then pass an empty tensor)
        returns:
          - scalar: logits FloatTensor[B, 1]
          - triplet: logits FloatTensor[B, 3]
        """
        embs = [self.emb_layers[name](x_cat[name].long()) for name in self.cat_order]  # [B, e_i]
        h_cat = torch.cat(embs, dim=-1) if len(embs) > 0 else None

        if self.num_in_dim > 0:
            h_num = self.num_proj(x_num.float())
            h = torch.cat([h_cat, h_num], dim=-1) if h_cat is not None else h_num
        else:
            h = h_cat

        logits = self.mlp(h)
        return logits

    # ---- Training step (handles scalar or triplet targets; soft/hard) ----
    def _step(self, batch, stage: str):
        x_cat, x_num, y, w = batch  # collate must provide these shapes
        logits = self.forward(x_cat, x_num)

        # Build loss depending on mode/labels
        if self.output_mode == "scalar":
            # Expect y in [0,1] floats; use BCEWithLogitsLoss
            y = y.float().view(-1, 1)
            loss_vec = F.binary_cross_entropy_with_logits(logits, y, reduction='none').squeeze(-1)
        else:
            # triplet
            if y.dtype in (torch.float32, torch.float64):  # soft labels
                # y: [B,3], probabilities; use KLDiv between log_softmax and probs
                logp = F.log_softmax(logits, dim=-1)
                loss_vec = F.kl_div(logp, y, reduction='none').sum(dim=-1)
            else:
                # hard labels in {0,1,2}
                loss_vec = F.cross_entropy(logits, y.long(), reduction='none')

        # sample weights (optional)
        if w is None:
            loss = loss_vec.mean()
        else:
            w = w.float()
            w = w * (w.numel() / (w.sum().clamp_min(1e-8)))  # normalize weights
            loss = (loss_vec * w).mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # track epoch avg
        if stage == "train":
            self.train_loss_sum += loss.item() * y.shape[0]
            self.train_n += y.shape[0]
        else:
            self.val_loss_sum += loss.item() * y.shape[0]
            self.val_n += y.shape[0]

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_train_epoch_end(self):
        if self.train_n > 0:
            self.log("train_loss_epoch_mean", self.train_loss_sum / self.train_n, prog_bar=True)
        self.train_loss_sum = 0.0; self.train_n = 0.0

    def on_validation_epoch_end(self):
        if self.val_n > 0:
            self.log("val_loss_epoch_mean", self.val_loss_sum / self.val_n, prog_bar=True)
        self.val_loss_sum = 0.0; self.val_n = 0.0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)