# ml/inference/equity_post.py
import torch
import torch.nn.functional as F

from ml.datasets.equitynet import EquityDatasetParquet
from ml.features.boards import load_board_clusterer
from ml.models.equity_net import EquityNetLit
from ml.utils.config import load_model_config


class EquityPostInfer:
    def __init__(self, cfg_path: str, ckpt_path: str, device: str = "cpu"):
        cfg = load_model_config(cfg_path)
        parquet = cfg["dataset"]["parquet_post"]
        x_cols  = cfg["dataset"]["x_cols_post"]     # ["stack_bb","hero_pos","opener_action","hand_id","board_cluster_id"]
        y_cols  = cfg["dataset"]["y_cols"]
        w_col   = cfg["dataset"]["weight_col"]

        self.ds = EquityDatasetParquet(parquet, x_cols, y_cols, w_col, device=torch.device("cpu"))
        cards   = self.ds.cards()
        order   = self.ds.feature_order
        self.enc = self.ds.id_maps()

        self.clusterer = load_board_clusterer(cfg)  # rule/kmeans

        self.device = torch.device(device)
        self.model = EquityNetLit(cards=cards, cat_order=order)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.to(self.device).eval()

    def _enc(self, col, val):
        m = self.enc[col]
        return m.get(val, len(m))

    def _cluster_id_from_board(self, board_cards):
        # expects ["Ah","Kd","7c"], clusterer.predict needs "AhKd7c"
        return int(self.clusterer.predict(["".join(board_cards)])[0])

    @torch.no_grad()
    def predict_one(self, stack_bb: int, hero_pos: str, opener_action: str,
                    hand_id: int, board_cards=None, board_cluster_id=None):
        if board_cluster_id is None:
            if not board_cards:
                raise ValueError("Provide board_cards or board_cluster_id")
            board_cluster_id = self._cluster_id_from_board(board_cards)

        x = {
            "stack_bb":        torch.tensor([ self._enc("stack_bb", stack_bb) ], dtype=torch.long, device=self.device),
            "hero_pos":        torch.tensor([ self._enc("hero_pos", hero_pos) ], dtype=torch.long, device=self.device),
            "opener_action":   torch.tensor([ self._enc("opener_action", opener_action) ], dtype=torch.long, device=self.device),
            "hand_id":         torch.tensor([ self._enc("hand_id", hand_id) ], dtype=torch.long, device=self.device),
            "board_cluster_id":torch.tensor([ self._enc("board_cluster_id", board_cluster_id) ], dtype=torch.long, device=self.device),
        }
        logits = self.model(x)
        probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return {"win": float(probs[0]), "tie": float(probs[1]), "lose": float(probs[2])}