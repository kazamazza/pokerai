import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ml.models.policy_consts import default_emb_dim

N_ACTIONS = 3


class PopulationNetLit(pl.LightningModule):
    def __init__(
        self,
        *,
        cards: Dict[str, int],
        feature_order: Optional[List[str]] = None,
        emb_dims: Optional[Dict[str, int]] = None,
        hidden_dims: List[int] = (64, 64),
        dropout: float = 0.10,

        # optim
        lr: float = 1e-3,
        weight_decay: float = 1e-4,

        # scheduler (optional; if warmup_steps > 0 we enable warmup+cosine)
        warmup_steps: int = 0,
        max_steps: int = 0,        # if 0, Lightning will infer; cosine will still work once set
        min_lr_scale: float = 0.05,  # final LR = min_lr_scale * lr

        use_soft_labels: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Inputs ----
        self.cards = dict(cards)
        self.feature_order = list(feature_order) if feature_order else list(self.cards.keys())

        # embedding dims (per feature)
        self.emb_dims = dict(emb_dims) if emb_dims else {
            k: default_emb_dim(v) for k, v in self.cards.items()
        }

        # sanity: ensure we have emb tables for the order we’ll use
        missing = [k for k in self.feature_order if k not in self.cards]
        if missing:
            raise ValueError(f"Unknown features in feature_order: {missing}")

        # ---- Embeddings ----
        self.emb_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=int(self.cards[name]),
                               embedding_dim=int(self.emb_dims[name]))
            for name in self.feature_order
        })

        # ---- Head ----
        in_dim = sum(int(self.emb_dims[name]) for name in self.feature_order)
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, int(h)), nn.ReLU(), nn.Dropout(float(dropout))]
            last = int(h)
        layers += [nn.Linear(last, N_ACTIONS)]
        self.mlp = nn.Sequential(*layers)

        # training mode
        self.use_soft_labels = bool(use_soft_labels)

        # running acc (epoch-level)
        self.train_acc_n = 0
        self.train_acc_d = 0
        self.val_acc_n = 0
        self.val_acc_d = 0

    # -------- forward --------
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = [self.emb_layers[name](x[name].long()) for name in self.feature_order]
        h = torch.cat(embs, dim=-1) if len(embs) > 0 else torch.empty(
            (next(iter(x.values())).shape[0], 0), device=self.device
        )
        return self.mlp(h)

    # -------- step --------
    def _step(self, batch, stage: str):
        x_dict, y, w = batch
        logits = self.forward(x_dict)

        if self.use_soft_labels:
            logp = F.log_softmax(logits, dim=-1)
            loss_vec = -(y * logp).sum(dim=1)
            y_hard = y.argmax(dim=1)
        else:
            loss_vec = F.cross_entropy(logits, y.long(), reduction="none")
            y_hard = y.long()

        if w is None:
            w = torch.ones_like(loss_vec)
        else:
            w = w.float()
            # normalize weights so avg(~1)
            w = w * (w.numel() / w.sum().clamp_min(1e-8))

        loss = (loss_vec * w).mean()

        # accuracy (for logging)
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
            self.log("train_acc_epoch", self.train_acc_n / self.train_acc_d, prog_bar=True)
        self.train_acc_n = 0; self.train_acc_d = 0

    def on_validation_epoch_end(self):
        if self.val_acc_d > 0:
            self.log("val_acc_epoch", self.val_acc_n / self.val_acc_d, prog_bar=True)
        self.val_acc_n = 0; self.val_acc_d = 0

    # -------- optimizers / sched --------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=float(self.hparams.lr),
                                weight_decay=float(self.hparams.weight_decay))

        warmup_steps = int(self.hparams.warmup_steps)
        max_steps = int(self.hparams.max_steps)
        min_scale = float(self.hparams.min_lr_scale)

        # If no warmup requested, keep it simple
        if warmup_steps <= 0:
            return opt

        # Lightning will set trainer.estimated_stepping_batches later.
        # If user passed max_steps=0, try to infer at setup time; else fall back to cosine over 1x epoch count
        if max_steps <= 0 and hasattr(self.trainer, "estimated_stepping_batches"):
            max_steps = int(self.trainer.estimated_stepping_batches) or warmup_steps * 10
        elif max_steps <= 0:
            max_steps = warmup_steps * 10  # safe fallback

        base_lr = float(self.hparams.lr)
        final_lr = max(base_lr * min_scale, 1e-6)

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            # cosine anneal from 1.0 → min_scale over remaining steps
            t = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
            # scale from 1.0 → min_scale
            return (min_scale + (1.0 - min_scale) * cosine)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        # log LR each step
        self.lr_schedulers()  # Lightning will attach schedulers; safe to call in hooks
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "name": "warmup_cosine",
            },
        }