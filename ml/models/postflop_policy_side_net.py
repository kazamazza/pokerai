from typing import Dict, Sequence, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ml.models.policy_consts import CatEmbedBlock, BoardBlock, ACTION_VOCAB


class PostflopPolicySideLit(pl.LightningModule):
    """
    Single-side training module for postflop policy.
    - side ∈ {"ip","oop"} only controls logging/metadata; dataset already provides y & mask for this side.
    - Head width == len(action_vocab) (ROOT or FACING vocab).
    - Loss: masked cross-entropy on soft targets (renormalized within mask).
    """
    def __init__(self, *,
                 side: str,
                 card_sizes: Dict[str,int],
                 cat_feature_order: Sequence[str],
                 board_hidden: int = 64,
                 mlp_hidden: Sequence[int] = (128,128),
                 dropout: float = 0.10,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 label_smoothing: float = 0.0,
                 class_weights_path: Optional[str] = None,
                 action_vocab: Optional[Sequence[str]] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["action_vocab"])
        side = side.lower(); assert side in ("ip","oop")
        self.side = side

        # ----- vocab for this side -----
        self.vocab = list(action_vocab) if action_vocab is not None else list(ACTION_VOCAB)
        self.V = len(self.vocab)

        # ----- embeddings & board encoder -----
        self.cat_order = list(cat_feature_order)
        self.cards = {k: int(v) for k, v in card_sizes.items()}

        self.embed = CatEmbedBlock(self.cards, self.cat_order)

        use_cluster = int(self.cards.get("board_cluster", 0)) > 0
        self.board = BoardBlock(
            hidden=board_hidden,
            n_clusters=(int(self.cards["board_cluster"]) if use_cluster else None),
            cluster_dim=8,
        )

        in_dim = self.embed.out_dim + self.board.out_dim
        layers, last = [], in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        # ----- single head with width V (side-specific) -----
        self.head = nn.Linear(last, self.V)

        # optional class weights (must match V)
        self.register_buffer("class_weights", None, persistent=False)
        if class_weights_path:
            try:
                w = torch.load(class_weights_path, map_location="cpu").float()
                if w.numel() == self.V:
                    self.class_weights = w
                else:
                    print(f"[warn] class_weights size {w.numel()} != V {self.V}; ignoring.")
            except Exception as e:
                print(f"[warn] failed loading class weights: {e}")

        # training hparams
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = float(label_smoothing)

    # ----- forward -----
    def forward(self, x_cat: Dict[str, torch.Tensor], x_cont: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_cat = self.embed(x_cat)
        cluster_id = x_cat.get("board_cluster") if "board_cluster" in self.cat_order else None
        z_brd = self.board(
            x_cont["board_mask_52"],
            x_cont["pot_bb"],
            x_cont["eff_stack_bb"],
            cluster_id=cluster_id,
        )
        h = self.trunk(torch.cat([z_cat, z_brd], dim=-1))
        return self.head(h)  # [B, V]

    @staticmethod
    def _ensure_nonempty_mask(mask: torch.Tensor) -> torch.Tensor:
        row_sum = mask.sum(dim=-1, keepdim=True)
        needs = (row_sum <= 0)
        if needs.any():
            fixed = mask.clone()
            fixed[needs.expand_as(mask)] = 1.0
            return fixed
        return mask

    def _side_loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits/target/mask are [B, V]
        B, V = logits.shape
        assert target.shape == (B, V) and mask.shape == (B, V)
        mask = self._ensure_nonempty_mask(mask)

        # label smoothing on masked targets
        t = target * mask
        t = t / (t.sum(dim=-1, keepdim=True) + 1e-8)
        if self.label_smoothing > 0.0:
            smooth = self.label_smoothing
            t = (1.0 - smooth) * t + smooth * (mask / (mask.sum(dim=-1, keepdim=True) + 1e-8))

        # masked log-softmax
        big_neg = torch.finfo(logits.dtype).min / 4
        masked_logits = torch.where(mask > 0.5, logits, big_neg)
        logp = F.log_softmax(masked_logits, dim=-1)
        ce = -(t * logp).sum(dim=-1)  # [B]
        return ce

    # ----- steps -----
    def training_step(self, batch, _):
        x_cat, x_cont, y, m, w = batch
        logits = self(x_cat, x_cont)       # [B, V]
        loss_vec = self._side_loss(logits, y, m)  # [B]
        loss = (loss_vec * w).sum() / (w.sum().clamp_min(1e-8))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x_cat, x_cont, y, m, w = batch
        logits = self(x_cat, x_cont)
        loss_vec = self._side_loss(logits, y, m)
        loss = (loss_vec * w).sum() / (w.sum().clamp_min(1e-8))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2, min_lr=1e-5
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "strict": False}}