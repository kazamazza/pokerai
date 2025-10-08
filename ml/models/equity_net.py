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
    def __init__(
        self,
        cards: Dict[str, int],
        cat_order: Optional[List[str]] = None,
        emb_dims: Optional[Dict[str, int]] = None,
        num_features: Optional[Sequence[str]] = None,
        num_in_dim: int = 0,
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        output_mode: str = "triplet",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cards = dict(cards)
        self.cat_order = cat_order or list(self.cards.keys())
        self.emb_dims = emb_dims or {k: default_emb_dim(v) for k, v in self.cards.items()}
        self.num_in_dim = int(num_in_dim)
        self.output_mode = output_mode

        # embeddings
        self.emb_layers = nn.ModuleDict({
            n: nn.Embedding(self.cards[n], self.emb_dims[n]) for n in self.cat_order
        })

        # numeric projection (for board_mask_52 etc.)
        proj_dim = 0
        if self.num_in_dim > 0:
            self.num_proj = nn.Sequential(
                nn.Linear(self.num_in_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            proj_dim = 64
        else:
            self.num_proj = nn.Identity()

        # MLP head
        in_dim = sum(self.emb_dims.values()) + proj_dim
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        out_dim = 1 if self.output_mode == "scalar" else 3
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    # ---------- Forward ----------
    def forward(self, x_cat: Dict[str, torch.Tensor], x_num: Optional[torch.Tensor] = None) -> torch.Tensor:
        embs = [self.emb_layers[n](x_cat[n].long()) for n in self.cat_order]
        h_cat = torch.cat(embs, dim=-1) if embs else None

        if self.num_in_dim > 0 and x_num is not None:
            h_num = self.num_proj(x_num)
            h = torch.cat([h_cat, h_num], dim=-1) if h_cat is not None else h_num
        else:
            h = h_cat

        return self.mlp(h)

    # ---------- Training ----------
    def _step(self, batch, stage: str):
        if len(batch) == 3:
            x_cat, y, w = batch
            x_num = None
        else:
            x_cat, x_num, y, w = batch

        logits = self.forward(x_cat, x_num)
        if self.output_mode == "triplet":
            logp = F.log_softmax(logits, dim=-1)
            loss_vec = F.kl_div(logp, y, reduction="none").sum(dim=-1)
        else:
            y = y.float().view(-1, 1)
            loss_vec = F.binary_cross_entropy_with_logits(logits, y, reduction="none").squeeze(-1)

        w = w.float()
        w = w * (w.numel() / w.sum().clamp_min(1e-8))
        loss = (loss_vec * w).mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, b, i): return self._step(b, "train")
    def validation_step(self, b, i): return self._step(b, "val")

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad: continue
            (no_decay if n.endswith("bias") or "norm" in n else decay).append(p)
        opt = torch.optim.AdamW([
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def on_save_checkpoint(self, ckpt: dict):
        ckpt["equity_meta"] = {
            "model_name": "EquityNet",
            "cat_feats": self.cat_order,
            "num_in_dim": self.num_in_dim,
            "emb_dims": self.emb_dims,
            "hidden_dims": self.hparams.hidden_dims,
            "dropout": self.hparams.dropout,
        }