from typing import Dict, Sequence, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ml.models.policy_consts import CatEmbedBlock, BoardBlock, VOCAB_SIZE, soft_kl


class PostflopPolicyLit(pl.LightningModule):
    """
    Single-class model: Lightning **is** the model.
    Inputs (from collate):
      x_cat  : dict[str, LongTensor[B]]       (hero_pos, ip_pos, oop_pos, ctx, street, ...)
      x_cont : dict[str, FloatTensor]         (board_mask_52[B,52], pot_bb[B,1], eff_stack_bb[B,1], ...)
      y_ip/y_oop : Float[B,V]                 (soft or one-hot targets)
      m_ip/m_oop : Float[B,V] in {0,1}        (role masks or legality masks)
      w      : Float[B]                       (sample weights)
    Outputs: masked policy over ACTION_VOCAB for IP and OOP.
    """
    def __init__(
        self,
        *,
        card_sizes: Dict[str, int],
        cat_feature_order: Sequence[str],
        board_hidden: int = 64,
        mlp_hidden: Sequence[int] = (128, 128),
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,  # reserved if you later smooth hard targets
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- blocks ----
        self.cat_order = list(cat_feature_order)
        self.cards     = {k: int(v) for k, v in card_sizes.items()}

        self.embed = CatEmbedBlock(self.cards, self.cat_order)
        self.board = BoardBlock(hidden=board_hidden)

        in_dim = self.embed.out_dim + self.board.out_dim
        layers: List[nn.Module] = []
        last = in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.head_ip  = nn.Linear(last, VOCAB_SIZE)
        self.head_oop = nn.Linear(last, VOCAB_SIZE)

    # ---- forward / heads ----
    def forward(self, x_cat: Dict[str, torch.Tensor], x_cont: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_cat = self.embed(x_cat)
        z_brd = self.board(x_cont["board_mask_52"], x_cont["pot_bb"], x_cont["eff_stack_bb"])
        h = self.trunk(torch.cat([z_cat, z_brd], dim=-1))
        return self.head_ip(h), self.head_oop(h)

    # ---- masked log-softmax (for illegal actions or role masking) ----
    @staticmethod
    def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits, big_neg)
        return F.log_softmax(masked, dim=-1)

    def _side_loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Normalize target *within the mask* so KL is well-posed even if some actions are illegal.
        t = target * mask
        t = t / (t.sum(dim=-1, keepdim=True) + 1e-8)
        logp = self._masked_log_softmax(logits, mask)
        return soft_kl(t, logp)  # [B]

    # ---- training / validation ----
    def training_step(self, batch, _):
        x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w = batch
        li, lo = self(x_cat, x_cont)
        kl_ip  = self._side_loss(li, y_ip,  m_ip)  # [B]
        kl_oop = self._side_loss(lo, y_oop, m_oop) # [B]
        loss_vec = kl_ip + kl_oop                   # [B]
        loss = (loss_vec * w).sum() / (w.sum() + 1e-8)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ip",  (kl_ip  * w).sum() / (w.sum() + 1e-8))
        self.log("train_oop", (kl_oop * w).sum() / (w.sum() + 1e-8))
        return loss

    def validation_step(self, batch, _):
        x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w = batch
        li, lo = self(x_cat, x_cont)
        kl_ip  = self._side_loss(li, y_ip,  m_ip)  # [B]
        kl_oop = self._side_loss(lo, y_oop, m_oop) # [B]
        loss_vec = kl_ip + kl_oop
        loss = (loss_vec * w).sum() / (w.sum() + 1e-8)  # mirror training weighting
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ip",  (kl_ip  * w).sum() / (w.sum() + 1e-8))
        self.log("val_oop", (kl_oop * w).sum() / (w.sum() + 1e-8))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    # ---- convenience inference (optional) ----
    @torch.no_grad()
    def predict_proba(self, x_cat: Dict[str, torch.Tensor], x_cont: Dict[str, torch.Tensor],
                      side: str = "ip", mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return probabilities over ACTION_VOCAB for the chosen side ('ip'|'oop')."""
        li, lo = self(x_cat, x_cont)
        logits = li if side.lower() == "ip" else lo
        if mask is None:
            mask = torch.ones_like(logits)
        logp = self._masked_log_softmax(logits, mask)
        return torch.exp(logp)