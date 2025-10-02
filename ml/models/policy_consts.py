from typing import Dict, Sequence
import torch.nn.functional as F
import torch
from torch import nn

ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500",
    "ALLIN",
]
VOCAB_INDEX = {a: i for i, a in enumerate(ACTION_VOCAB)}
VOCAB_SIZE  = len(ACTION_VOCAB)

def soft_kl(y_true: torch.Tensor, log_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(y || p) = sum y * (log y - log p). y_true must be row-normalized."""
    y = torch.clamp(y_true, min=eps)
    return torch.sum(y * (torch.log(y + eps) - log_probs), dim=-1)

# ===================== small blocks =====================
class CatEmbedBlock(nn.Module):
    def __init__(self, cards: Dict[str, int], feature_order: Sequence[str]):
        super().__init__()
        self.feature_order = list(feature_order)
        self.embs = nn.ModuleDict()
        for name in self.feature_order:
            c = int(cards[name])
            d = min(64, max(8, int(round(min(32.0, (c ** 0.5) * 4)))))  # simple heuristic
            self.embs[name] = nn.Embedding(c, d)
        self.out_dim = sum(e.embedding_dim for e in self.embs.values())
    def forward(self, x_cat: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.embs[n](x_cat[n]) for n in self.feature_order], dim=-1)

class BoardBlock(nn.Module):
    """Encodes board via 52-bit mask + rank histogram; also uses pot_bb & eff_stack_bb."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(52 + 2 + 13, hidden)  # 52 + (pot, stack) + 13 ranks
        self.fc2 = nn.Linear(hidden, hidden)
        self.out_dim = hidden
    @staticmethod
    def rank_histogram(board_mask_52: torch.Tensor) -> torch.Tensor:
        B = board_mask_52.size(0)
        return board_mask_52.view(B, 13, 4).sum(dim=2)  # [B,13]
    def forward(self, board_mask_52: torch.Tensor, pot_bb: torch.Tensor, eff_stack_bb: torch.Tensor) -> torch.Tensor:
        rh = self.rank_histogram(board_mask_52)
        h = torch.cat([board_mask_52, pot_bb, eff_stack_bb, rh], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return h