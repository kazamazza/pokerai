# ml/models/rangenet.py
from __future__ import annotations
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

HAND_COUNT = 169

class CatEmbedBlock(nn.Module):
    def __init__(self, cards: Dict[str, int], feature_order: Sequence[str], emb_dim: int = 16):
        super().__init__()
        self.feature_order = list(feature_order)
        self.embs = nn.ModuleDict()
        for name in self.feature_order:
            c = int(cards[name])
            # simple embedding size rule-of-thumb
            d = min(64, max(4, int(round(min(16.0, (c**0.5)*4)))))
            self.embs[name] = nn.Embedding(c, d)
        self.out_dim = sum(e.embedding_dim for e in self.embs.values())

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        outs = []
        for name in self.feature_order:
            outs.append(self.embs[name](x_dict[name]))
        return torch.cat(outs, dim=-1)

class RangeNet(nn.Module):
    def __init__(self, cards: Dict[str, int], feature_order: Sequence[str],
                 hidden_dims: Sequence[int] = (128,128), dropout: float = 0.1):
        super().__init__()
        self.embed = CatEmbedBlock(cards, feature_order)
        layers: List[nn.Module] = []
        in_dim = self.embed.out_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, HAND_COUNT)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.embed(x_dict)           # [B, E]
        logits = self.mlp(z)             # [B, 169]
        probs = F.softmax(logits, dim=-1)
        return probs

class RangeNetLit(pl.LightningModule):
    def __init__(self, cards: Dict[str,int], feature_order: Sequence[str],
                 hidden_dims: Sequence[int]=(128,128), dropout: float=0.1,
                 lr: float=1e-3, weight_decay: float=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["cards","feature_order"])
        self.model = RangeNet(cards, feature_order, hidden_dims, dropout)
        self.lr = lr
        self.wd = weight_decay

    def training_step(self, batch, _):
        x, y, w = batch
        p = self.model(x)
        # KL divergence with soft targets y (sum=1)
        # Use log on predictions: KL(y || p) = sum y * (log y - log p)
        loss = torch.sum(w * torch.sum(y * (torch.log(y+1e-8) - torch.log(p+1e-8)), dim=-1)) / (w.sum()+1e-8)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y, w = batch
        p = self.model(x)
        loss = torch.sum(w * torch.sum(y * (torch.log(y+1e-8) - torch.log(p+1e-8)), dim=-1)) / (w.sum()+1e-8)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)