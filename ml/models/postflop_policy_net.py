from typing import Dict, Sequence, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ml.models.policy_consts import VOCAB_SIZE, CatEmbedBlock, BoardBlock


def _safe_soft_kl(p: torch.Tensor, logq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(p||q) where p is a proper distribution over last dim and logq are log-probs.
    Returns [B]; numerically stable; no batch reduction.
    """
    p = p.clamp_min(eps)
    return (p * (torch.log(p) - logq)).sum(dim=-1)


def _label_smooth_onehot(target: torch.Tensor, smoothing: float) -> torch.Tensor:
    """
    Apply label smoothing to one-hot-like distributions.
    Expects last-dim to sum to 1 already.
    """
    if smoothing <= 0.0:
        return target
    V = target.shape[-1]
    return (1.0 - smoothing) * target + smoothing * (1.0 / V)


def _ensure_nonempty_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    If any row has zero valid actions in mask, set that row to all-ones.
    Avoids -inf rows in masked softmax and NaNs in KL.
    """
    row_sum = mask.sum(dim=-1, keepdim=True)
    needs_fix = (row_sum <= 0)
    if needs_fix.any():
        fixed = mask.clone()
        fixed[needs_fix.expand_as(mask)] = 1.0
        return fixed
    return mask


class PostflopPolicyLit(pl.LightningModule):
    """
    Masked policy over ACTION_VOCAB for IP and OOP.
    Inputs from collate:
      x_cat:  dict[str, LongTensor[B]]
      x_cont: dict[str, FloatTensor], expects keys: board_mask_52[B,52], pot_bb[B,1], eff_stack_bb[B,1]
      y_ip/y_oop: Float[B,V]
      m_ip/m_oop: Float[B,V] in {0,1}
      w: Float[B]
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
        label_smoothing: float = 0.0,
        class_weights_path: Optional[str] = None,  # optional external weights tensor (Float[V])
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- embeddings & board encoder ----
        self.cat_order = list(cat_feature_order)
        self.cards = {k: int(v) for k, v in card_sizes.items()}

        self.embed = CatEmbedBlock(self.cards, self.cat_order)
        self.board = BoardBlock(hidden=board_hidden)

        in_dim = self.embed.out_dim + self.board.out_dim
        layers: List[nn.Module] = []
        last = in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        self.head_ip = nn.Linear(last, VOCAB_SIZE)
        self.head_oop = nn.Linear(last, VOCAB_SIZE)

        # optional class weights
        self.register_buffer("class_weights", None, persistent=False)
        if class_weights_path:
            try:
                w = torch.load(class_weights_path, map_location="cpu")
                if w.numel() == VOCAB_SIZE:
                    self.class_weights = w.float()
                else:
                    print(f"[warn] class_weights size {w.numel()} != VOCAB_SIZE {VOCAB_SIZE}; ignoring weights.")
            except Exception as e:
                print(f"[warn] failed loading class weights: {e}")

    # ---- forward ----
    def forward(self, x_cat: Dict[str, torch.Tensor], x_cont: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_cat = self.embed(x_cat)
        z_brd = self.board(x_cont["board_mask_52"], x_cont["pot_bb"], x_cont["eff_stack_bb"])
        h = self.trunk(torch.cat([z_cat, z_brd], dim=-1))
        return self.head_ip(h), self.head_oop(h)

    # ---- masked log-softmax ----
    @staticmethod
    def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = _ensure_nonempty_mask(mask)
        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits, big_neg)
        return F.log_softmax(masked, dim=-1)

    def _side_loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # validate
        B, V = logits.shape
        assert target.shape == (B, V) and mask.shape == (B, V)

        # normalize targets within mask
        t = (target * mask)
        t = t / (t.sum(dim=-1, keepdim=True) + 1e-8)

        # masked log-softmax then CE = -∑ t log p  (non-negative)
        logp = self._masked_log_softmax(logits, mask)
        ce = -(t * logp).sum(dim=-1)  # [B] >= 0
        return ce

    # ---- training / validation ----
    def training_step(self, batch, _):
        x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w = batch

        # defensive shape checks on inputs
        B = w.shape[0]
        for name, t in [("y_ip", y_ip), ("y_oop", y_oop), ("m_ip", m_ip), ("m_oop", m_oop)]:
            assert t.dim() == 2 and t.shape[0] == B and t.shape[1] == VOCAB_SIZE, \
                f"{name} shape {t.shape} incompatible with batch {B} and VOCAB_SIZE {VOCAB_SIZE}"

        li, lo = self(x_cat, x_cont)                   # [B,V], [B,V]
        kl_ip  = self._side_loss(li, y_ip,  m_ip)      # [B]
        kl_oop = self._side_loss(lo, y_oop, m_oop)     # [B]
        loss_vec = kl_ip + kl_oop                      # [B]
        loss = (loss_vec * w).sum() / (w.sum().clamp_min(1e-8))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ip",  (kl_ip  * w).sum() / (w.sum().clamp_min(1e-8)))
        self.log("train_oop", (kl_oop * w).sum() / (w.sum().clamp_min(1e-8)))
        return loss

    def validation_step(self, batch, _):
        x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w = batch
        li, lo = self(x_cat, x_cont)
        kl_ip  = self._side_loss(li, y_ip,  m_ip)
        kl_oop = self._side_loss(lo, y_oop, m_oop)
        loss_vec = kl_ip + kl_oop
        loss = (loss_vec * w).sum() / (w.sum().clamp_min(1e-8))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ip",  (kl_ip  * w).sum() / (w.sum().clamp_min(1e-8)))
        self.log("val_oop", (kl_oop * w).sum() / (w.sum().clamp_min(1e-8)))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    # ---- convenience inference ----
    @torch.no_grad()
    def predict_proba(self, x_cat: Dict[str, torch.Tensor], x_cont: Dict[str, torch.Tensor],
                      side: str = "ip", mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        li, lo = self(x_cat, x_cont)
        logits = li if side.lower() == "ip" else lo
        if mask is None:
            mask = torch.ones_like(logits)
        mask = _ensure_nonempty_mask(mask)
        logp = self._masked_log_softmax(logits, mask)
        return torch.exp(logp)