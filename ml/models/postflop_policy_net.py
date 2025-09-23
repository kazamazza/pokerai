from typing import Dict, Sequence, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300",
    "ALLIN",
]
VOCAB_INDEX = {a: i for i, a in enumerate(ACTION_VOCAB)}
VOCAB_SIZE = len(ACTION_VOCAB)

def soft_kl(y_true: torch.Tensor, log_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(y || p) = sum y * (log y - log p); expects y_true already normalized & masked.
    """
    y = torch.clamp(y_true, min=eps)
    return torch.sum(y * (torch.log(y) - log_probs), dim=-1)

# ---------------- Embedding blocks ----------------

class CatEmbedBlock(nn.Module):
    """
    Simple categorical embedder for:
      - positions (string ID -> index)
      - bet_sizing_id (string ID -> index)
      - topology/context (string ID -> index)  [optional]
    """
    def __init__(self, cards: Dict[str, int], feature_order: Sequence[str]):
        super().__init__()
        self.feature_order = list(feature_order)
        self.embs = nn.ModuleDict()
        for name in self.feature_order:
            c = int(cards[name])
            d = min(64, max(8, int(round(min(32.0, (c ** 0.5) * 4)))))
            self.embs[name] = nn.Embedding(c, d)
        self.out_dim = sum(e.embedding_dim for e in self.embs.values())

    def forward(self, x_cat: Dict[str, torch.Tensor]) -> torch.Tensor:
        outs = []
        for name in self.feature_order:
            outs.append(self.embs[name](x_cat[name]))  # [B, d_name]
        return torch.cat(outs, dim=-1)  # [B, sum_d]

class BoardBlock(nn.Module):
    """
    Encode board as a 52-bit mask + rank histogram.
    Inputs:
      - board_mask_52: FloatTensor [B, 52] in {0,1}
      - pot_bb, stack_bb: FloatTensor [B, 1]
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(52 + 2 + 13, hidden)  # 52 + (pot, stack) + 13-rank hist
        self.fc2 = nn.Linear(hidden, hidden)
        self.out_dim = hidden

    @staticmethod
    def rank_histogram(board_mask_52: torch.Tensor) -> torch.Tensor:
        B = board_mask_52.size(0)
        return board_mask_52.view(B, 13, 4).sum(dim=2)  # [B,13]

    def forward(self, board_mask_52: torch.Tensor, pot_bb: torch.Tensor, stack_bb: torch.Tensor) -> torch.Tensor:
        rank_hist = self.rank_histogram(board_mask_52)
        h = torch.cat([board_mask_52, pot_bb, stack_bb, rank_hist], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return h

# ---------------- Model ----------------

class PostflopPolicyNet(nn.Module):
    """
    Predict masked root policy over ACTION_VOCAB for IP and OOP.
    Inputs:
      x_cat: dict of categorical tensors (positions_id, bet_menu_id, ctx_id)
      x_cont: dict with board_mask_52 [B,52], pot_bb [B,1], eff_stack_bb [B,1]
      mask_ip, mask_oop: Float [B, V] in {0,1} (legal actions per side)
    Outputs:
      logits_ip, logits_oop: [B, V] (masked softmax occurs in loss)
    """
    def __init__(self, card_sizes: Dict[str, int], cat_feature_order: Sequence[str],
                 board_hidden: int = 64, mlp_hidden: Sequence[int] = (128,128), dropout: float = 0.1):
        super().__init__()
        self.embed = CatEmbedBlock(card_sizes, cat_feature_order)
        self.board = BoardBlock(hidden=board_hidden)

        in_dim = self.embed.out_dim + self.board.out_dim
        layers: List[nn.Module] = []
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head_ip  = nn.Linear(in_dim, VOCAB_SIZE)
        self.head_oop = nn.Linear(in_dim, VOCAB_SIZE)

    def forward(self, x_cat: Dict[str, torch.Tensor], x_cont: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_cat = self.embed(x_cat)  # [B, Dcat]
        z_brd = self.board(x_cont["board_mask_52"], x_cont["pot_bb"], x_cont["eff_stack_bb"])  # [B, Dbrd]
        z = torch.cat([z_cat, z_brd], dim=-1)
        h = self.trunk(z)
        return self.head_ip(h), self.head_oop(h)

# ---------------- Lightning wrapper ----------------

class PostflopPolicyNetLit(pl.LightningModule):
    def __init__(
        self,
        card_sizes: Dict[str, int],
        cat_feature_order: Sequence[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
        board_hidden: int = 64,
        mlp_hidden: Sequence[int] = (128,128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = PostflopPolicyNet(
            card_sizes, cat_feature_order,
            board_hidden=board_hidden, mlp_hidden=mlp_hidden, dropout=dropout
        )

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask in {0,1}; set very negative where illegal
        big_neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0.5, logits, big_neg)
        return F.log_softmax(masked, dim=-1)

    def _loss_side(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t = (target * mask)
        t = t / (t.sum(dim=-1, keepdim=True) + 1e-8)
        log_p = self._masked_softmax(logits, mask)
        return soft_kl(t, log_p)  # [B]

    def training_step(self, batch, _):
        x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w = batch
        li, lo = self.model(x_cat, x_cont)
        kl_ip = self._loss_side(li, y_ip, m_ip)  # [B]
        kl_oop = self._loss_side(lo, y_oop, m_oop)  # [B]
        kl = kl_ip + kl_oop  # [B]
        # weighted mean
        loss = (kl * w).sum() / (w.sum() + 1e-8)
        self.log_dict({"train_loss": loss, "train_ip": (kl_ip * w).sum() / (w.sum() + 1e-8),
                       "train_oop": (kl_oop * w).sum() / (w.sum() + 1e-8)}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w = batch
        li, lo = self.model(x_cat, x_cont)
        loss_ip  = self._loss_side(li, y_ip, m_ip)
        loss_oop = self._loss_side(lo, y_oop, m_oop)
        loss = (loss_ip + loss_oop) * w.mean()
        self.log_dict({"val_loss": loss, "val_ip": loss_ip, "val_oop": loss_oop}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)