from typing import Dict, Sequence, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ml.models.policy_consts import VOCAB_SIZE, CatEmbedBlock, BoardBlock


class PostflopPolicySideLit(pl.LightningModule):
    """
    Single-side training module.
      side ∈ {"ip","oop"} decides which head is trained & saved.
      Dataset provides y, m for that side only.
    """
    def __init__(self, *, side: str, card_sizes: Dict[str,int], cat_feature_order: Sequence[str],
                 board_hidden: int = 64, mlp_hidden: Sequence[int] = (128,128), dropout: float = 0.10,
                 lr: float = 1e-3, weight_decay: float = 1e-4, label_smoothing: float = 0.0,
                 class_weights_path: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        side = side.lower()
        assert side in ("ip","oop"), "side must be 'ip' or 'oop'"
        self.side = side

        # --- embeddings & board ---
        self.cat_order = list(cat_feature_order)
        self.cards = {k: int(v) for k, v in card_sizes.items()}

        self.embed = CatEmbedBlock(self.cards, self.cat_order)
        use_cluster = int(self.cards.get("board_cluster", 0)) > 0
        self.board = BoardBlock(hidden=board_hidden,
                                n_clusters=(int(self.cards["board_cluster"]) if use_cluster else None),
                                cluster_dim=8)

        in_dim = self.embed.out_dim + self.board.out_dim
        layers, last = [], in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        # heads (we keep both for code reuse, but will use only one)
        self.head_ip  = nn.Linear(last, VOCAB_SIZE)
        self.head_oop = nn.Linear(last, VOCAB_SIZE)

        self.register_buffer("class_weights", None, persistent=False)
        if class_weights_path:
            try:
                w = torch.load(class_weights_path, map_location="cpu")
                if w.numel() == VOCAB_SIZE:
                    self.class_weights = w.float()
            except Exception:
                pass

    def forward(self, x_cat, x_cont):
        z_cat = self.embed(x_cat)
        cluster_id = x_cat.get("board_cluster", None) if hasattr(x_cat, "get") else None
        z_brd = self.board(
            x_cont["board_mask_52"], x_cont["pot_bb"], x_cont["eff_stack_bb"],
            cluster_id=cluster_id
        )
        h = self.trunk(torch.cat([z_cat, z_brd], dim=-1))
        logits_ip  = self.head_ip(h)
        logits_oop = self.head_oop(h)
        return logits_ip, logits_oop

    @staticmethod
    def _ensure_nonempty_mask(mask: torch.Tensor) -> torch.Tensor:
        row_sum = mask.sum(dim=-1, keepdim=True)
        needs_fix = (row_sum <= 0)
        if needs_fix.any():
            fixed = mask.clone()
            fixed[needs_fix.expand_as(mask)] = 1.0
            return fixed
        return mask

    def _side_loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = self._ensure_nonempty_mask(mask)
        t = (target * mask)
        t = t / (t.sum(dim=-1, keepdim=True) + 1e-8)   # renorm within legal set
        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits, big_neg)
        logp = F.log_softmax(masked, dim=-1)
        return -(t * logp).sum(dim=-1)   # [B]

    def training_step(self, batch, _):
        x_cat, x_cont, y, m, w = batch
        li, lo = self(x_cat, x_cont)
        logits = li if self.side == "ip" else lo
        loss_vec = self._side_loss(logits, y, m)            # [B]
        loss = (loss_vec * w).sum() / (w.sum().clamp_min(1e-8))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x_cat, x_cont, y, m, w = batch
        li, lo = self(x_cat, x_cont)
        logits = li if self.side == "ip" else lo
        loss_vec = self._side_loss(logits, y, m)
        loss = (loss_vec * w).sum() / (w.sum().clamp_min(1e-8))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "strict": False}}