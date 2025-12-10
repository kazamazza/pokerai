# file: ml/models/evnet.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl  # PL <=2 still imports as this alias in many envs
except Exception:
    pl = None  # if you want pure nn.Module only, keep pl=None


# -----------------------------
# Small utilities
# -----------------------------
def _emb_dim_rule(card: int, max_dim: int = 64, min_dim: int = 4) -> int:
    if card <= 1:
        return 0
    d = int(math.ceil(max(1.0, math.log2(card)) * 4))
    return max(min_dim, min(max_dim, d))

def _mlp(d_in: int, hidden: Sequence[int], d_out: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = d_in
    for h in hidden:
        layers += [nn.Linear(last, int(h)), nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = int(h)
    layers.append(nn.Linear(last, d_out))
    return nn.Sequential(*layers)


# -----------------------------
# Core model
# -----------------------------

@dataclass
class EVNetConfig:
    cat_cardinalities: List[int]
    cont_dim: int
    action_vocab: Sequence[str]
    hidden_dims: Sequence[int] = (256, 256)
    dropout: float = 0.10
    max_emb_dim: int = 64
    min_emb_dim: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4

class EVNet(nn.Module):
    """
    Simple, fast EV regressor:
      - Per-categorical-feature embeddings → concat
      - Concat with continuous vector
      - MLP → |action_vocab| EV outputs (units: bb)
    """
    def __init__(self, cfg: EVNetConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab = list(cfg.action_vocab)
        self.vocab_size = len(self.vocab)

        # embeddings
        self.cardinalities: List[int] = list(cfg.cat_cardinalities)
        emb_layers: List[nn.Module] = []
        self.emb_dims: List[int] = []
        for card in self.cardinalities:
            d = _emb_dim_rule(card, cfg.max_emb_dim, cfg.min_emb_dim) if card > 1 else 0
            self.emb_dims.append(d)
            emb_layers.append(nn.Embedding(card, d) if d > 0 else nn.Identity())
        self.embeds = nn.ModuleList(emb_layers)

        emb_total = sum(self.emb_dims)
        d_in = emb_total + int(cfg.cont_dim)
        self.backbone = _mlp(d_in, cfg.hidden_dims, self.vocab_size, cfg.dropout)

    @property
    def config(self) -> EVNetConfig:
        """WHY: expose for Lightning snapshot/back-compat when EVLit saves hparams."""
        return self.cfg

    def forward(self, x_cat: torch.LongTensor, x_cont: torch.FloatTensor) -> torch.FloatTensor:
        """
        x_cat: [B, C] (C == len(cat_cardinalities))
        x_cont: [B, D]
        returns: [B, V]
        """
        B = x_cont.size(0)
        embs: List[torch.Tensor] = []
        if x_cat.numel() > 0:
            for i, emb in enumerate(self.embeds):
                if isinstance(emb, nn.Embedding):
                    embs.append(emb(x_cat[:, i]))
                else:
                    # keep dim alignment for cat with cardinality 1
                    embs.append(x_cat[:, i].new_zeros((B, 0)))
        x = torch.cat([*embs, x_cont], dim=-1) if embs else x_cont
        return self.backbone(x)

# -----------------------------
# Lightning wrapper (optional)
# -----------------------------
# --- loss with per-batch weight normalization ---
class WeightedMSE(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,          # [B, V]
        target: torch.Tensor,        # [B, V]
        weight: Optional[torch.Tensor] = None,   # [B]
        y_mask: Optional[torch.Tensor] = None,   # [B, V], 1=learn, 0=ignore
    ) -> torch.Tensor:
        err2 = (pred - target) ** 2  # [B, V]
        if y_mask is not None:
            # why: zero loss for illegal/impossible actions
            err2 = err2 * y_mask
            denom = y_mask.sum(dim=-1).clamp_min(1.0)   # avoid /0 when all masked
            loss_vec = err2.sum(dim=-1) / denom         # [B]
        else:
            loss_vec = err2.mean(dim=-1)                 # [B]
        if weight is not None:
            w = weight.view(-1).float()
            w = w / w.mean().clamp_min(1e-8)
            loss_vec = loss_vec * w
        return loss_vec.mean()

# ml/models/evnet.py

class EVLit(pl.LightningModule if pl is not None else nn.Module):  # type: ignore[misc]
    """
    Lightning wrapper for EVNet.
    Now supports clean load_from_checkpoint() by saving "config" in hparams:
      - If 'net' is provided, we snapshot its config into hparams.
      - If 'net' is None, we rebuild EVNet from hparams['config'].
    """

    def __init__(
        self,
        *,
        # when training new models, pass 'config' instead of 'net'
        config: Optional[dict] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # backward-compat: if you still pass a prebuilt net, we snapshot its config
        net: Optional["EVNet"] = None,
    ):
        super().__init__()

        # Save hparams so Lightning can recreate the module at load time
        # NOTE: don't store big objects; just the config and scalars.
        if net is None and config is None:
            raise ValueError("EVLit requires either a 'config' dict or a prebuilt 'net'.")

        if net is not None:
            # Extract config from the provided net (assumes net.config exists)
            try:
                cfg = {
                    "cat_cardinalities": list(net.config.cat_cardinalities),
                    "cont_dim": int(net.config.cont_dim),
                    "action_vocab": list(net.config.action_vocab),
                    "hidden_dims": list(net.config.hidden_dims),
                    "dropout": float(net.config.dropout),
                    # include any extra fields you support in EVNetConfig
                    "max_emb_dim": int(getattr(net.config, "max_emb_dim", 32)),
                }
            except Exception:
                # If EVNet does not expose 'config', require config explicitly
                if config is None:
                    raise
                cfg = dict(config)
        else:
            cfg = dict(config)

        # Lightning will pass these back into __init__ during load_from_checkpoint
        # so ensure the signature matches ('config', 'lr', 'weight_decay').
        self.save_hyperparameters({"config": cfg, "lr": float(lr), "weight_decay": float(weight_decay)})

        # Build or use the provided network
        if net is None:
            net = EVNet(EVNetConfig(
                cat_cardinalities=cfg["cat_cardinalities"],
                cont_dim=cfg["cont_dim"],
                action_vocab=cfg["action_vocab"],
                hidden_dims=cfg.get("hidden_dims", [256, 256]),
                dropout=cfg.get("dropout", 0.10),
                # include if your EVNetConfig supports it; otherwise drop it
                max_emb_dim=cfg.get("max_emb_dim", 32),
            ))

        self.net = net
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.criterion = WeightedMSE()

    def forward(self, x_cat: torch.LongTensor, x_cont: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x_cat, x_cont)

    @staticmethod
    def _metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mae = (pred - target).abs().mean()
        rmse = torch.sqrt(F.mse_loss(pred, target))
        return mae, rmse

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        y_hat = self.net(batch["x_cat"], batch["x_cont"])
        y     = batch["y"]
        w     = batch.get("w", None)
        m     = batch.get("y_mask", None)
        loss  = self.criterion(y_hat, y, w, m)
        mae   = (y_hat - y).abs().mean()
        rmse  = torch.sqrt(((y_hat - y) ** 2).mean())
        self.log_dict({"train_loss": loss, "train_mae_bb": mae, "train_rmse_bb": rmse},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        y_hat = self.net(batch["x_cat"], batch["x_cont"])
        y     = batch["y"]
        w     = batch.get("w", None)
        m     = batch.get("y_mask", None)
        loss  = self.criterion(y_hat, y, w, m)
        mae   = (y_hat - y).abs().mean()
        rmse  = torch.sqrt(((y_hat - y) ** 2).mean())
        self.log_dict({"val_loss": loss, "val_mae_bb": mae, "val_rmse_bb": rmse},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# -----------------------------
# Factory from dataset sidecar
# -----------------------------
def build_ev_model_from_dataset_sidecar(
    *,
    cat_cardinalities: Sequence[int],
    cont_dim: int,
    action_vocab: Sequence[str],
    hidden_dims: Sequence[int] = (256, 256),
    dropout: float = 0.10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[EVNet, Optional[EVLit]]:
    cfg = EVNetConfig(
        cat_cardinalities=list(cat_cardinalities),
        cont_dim=int(cont_dim),
        action_vocab=list(action_vocab),
        hidden_dims=list(hidden_dims),
        dropout=float(dropout),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    net = EVNet(cfg)
    lit = EVLit(net=net, lr=cfg.lr, weight_decay=cfg.weight_decay) if pl is not None else None  # <-- fixed
    return net, lit


# -----------------------------
# Example convenience hook
# -----------------------------
def build_from_dataset(ds, *, hidden_dims=(256, 256), dropout=0.10, lr=1e-3, weight_decay=1e-4):
    """
    ds: EVParquetDataset
    Returns: (EVNet, EVLit or None)
    """
    return build_ev_model_from_dataset_sidecar(
        cat_cardinalities=ds.cat_cardinalities,
        cont_dim=ds.cont_dim,
        action_vocab=ds.sidecar.action_vocab,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )