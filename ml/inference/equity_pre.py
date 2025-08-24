# ml/inference/equity_pre.py
import torch
from pathlib import Path

from ml.datasets.equitynet import EquityDatasetParquet
from ml.models.equity_net import EquityNetLit
from ml.utils.config import load_model_config

import torch.nn.functional as F

class EquityPreInfer:
    def __init__(self, cfg_path: str, ckpt_path: str, device: str = "cpu"):
        cfg = load_model_config(cfg_path)
        parquet = cfg["dataset"]["parquet_pre"]
        x_cols  = cfg["dataset"]["x_cols_pre"]      # ["stack_bb","hero_pos","opener_action","hand_id"]
        y_cols  = cfg["dataset"]["y_cols"]          # ["y_win","y_tie","y_lose"]
        w_col   = cfg["dataset"]["weight_col"]

        self.ds = EquityDatasetParquet(parquet, x_cols, y_cols, w_col, device=torch.device("cpu"))
        cards   = self.ds.cards()                   # {"stack_bb":C0,"hero_pos":C1,...}
        order   = self.ds.feature_order             # matches x_cols
        self.enc = self.ds.id_maps()                # raw_value -> id per column

        self.device = torch.device(device)
        self.model = EquityNetLit(cards=cards, cat_order=order)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.to(self.device).eval()

    def _enc(self, col, val):
        m = self.enc[col]
        # unseen → unknown bucket (the dataset added it as last id if needed)
        return m.get(val, len(m))

    @torch.no_grad()
    def predict_one(self, stack_bb: int, hero_pos: str, opener_action: str, hand_id: int):
        x = {
            "stack_bb":      torch.tensor([ self._enc("stack_bb", stack_bb) ], dtype=torch.long, device=self.device),
            "hero_pos":      torch.tensor([ self._enc("hero_pos", hero_pos) ], dtype=torch.long, device=self.device),
            "opener_action": torch.tensor([ self._enc("opener_action", opener_action) ], dtype=torch.long, device=self.device),
            "hand_id":       torch.tensor([ self._enc("hand_id", hand_id) ], dtype=torch.long, device=self.device),
        }
        logits = self.model(x)
        probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return {"win": float(probs[0]), "tie": float(probs[1]), "lose": float(probs[2])}