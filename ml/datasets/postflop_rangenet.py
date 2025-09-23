import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

from ml.datasets.rangenet import canon_pos, canon_ctx, canon_street
from ml.models.postflop_policy_net import VOCAB_INDEX, VOCAB_SIZE, ACTION_VOCAB


class PostflopPolicyDatasetParquet(Dataset):
    """
    Dataset for PostflopPolicyNet.

    Expected columns (from your policy-parts builder):
      X:
        stack_bb (int), pot_bb (float),
        hero_pos, ip_pos, oop_pos (str in {UTG,HJ,CO,BTN,SB,BB}),
        street (FLOP/TURN/RIVER or 1/2/3),
        ctx (str/int),
        board_cluster (int)         [present if you bucketed]
        bet_sizing_id (str)         [optional]
        board_mask_52 (list[52])    [optional]
        actor (ip|oop)

      Y:
        Either soft policy columns per action (CHECK, BET_33, ...), or a single 'action' string.

      W:
        weight (float, optional; defaults to 1.0)
    """

    CAT_FEATURES = ["hero_pos", "ip_pos", "oop_pos", "ctx", "street"]
    SOFT_Y_COLS = ACTION_VOCAB  # e.g. ["CHECK","BET_33","BET_66","BET_100","DONK_33","RAISE_66","RAISE_100"]

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        weight_col: str = "weight",
        device: Optional[torch.device] = None,
        strict_canon: bool = True,
    ):
        self.parquet_path = Path(parquet_path)
        self.df = pd.read_parquet(str(self.parquet_path)).copy()

        # --- Canonicalize ---
        if "hero_pos" in self.df.columns:
            self.df["hero_pos"] = self.df["hero_pos"].map(canon_pos)
        if "ip_pos" in self.df.columns:
            self.df["ip_pos"] = self.df["ip_pos"].map(canon_pos)
        if "oop_pos" in self.df.columns:
            self.df["oop_pos"] = self.df["oop_pos"].map(canon_pos)
        if "ctx" in self.df.columns:
            self.df["ctx"] = self.df["ctx"].map(canon_ctx)
        if "street" in self.df.columns:
            self.df["street"] = self.df["street"].map(canon_street)

        if strict_canon:
            for col in ["hero_pos", "ip_pos", "oop_pos", "ctx", "street"]:
                if col in self.df.columns:
                    bad = self.df[pd.isna(self.df[col])]
                    if not bad.empty:
                        raise ValueError(f"Invalid values in {col}: {len(bad)} bad row(s).")

        # --- Weights ---
        self.weights = (
            torch.tensor(self.df[weight_col].values, dtype=torch.float32)
            if weight_col in self.df.columns
            else torch.ones(len(self.df), dtype=torch.float32)
        )

        # --- Targets: prefer soft policy columns, else fall back to hard 'action' ---
        has_soft = all(c in self.df.columns for c in self.SOFT_Y_COLS)
        self.has_soft = has_soft

        if not has_soft and "action" not in self.df.columns:
            raise ValueError("Policy parquet must contain either per-action columns or an 'action' column.")

        if not has_soft:
            # map action string -> index
            act_idx = self.df["action"].map(lambda a: VOCAB_INDEX.get(str(a).upper(), None))
            if act_idx.isna().any():
                bad = self.df[act_idx.isna()]
                raise ValueError(f"Unmapped actions found: {sorted(bad['action'].astype(str).unique().tolist())}")
            self.action_idx = torch.tensor(act_idx.values, dtype=torch.long)
        else:
            # store the soft policy matrix (N, V)
            self.y_soft = torch.tensor(self.df[self.SOFT_Y_COLS].values, dtype=torch.float32)

        # --- Features we will emit ---
        self.cat_features = [c for c in self.CAT_FEATURES if c in self.df.columns]
        self.cont_features = ["stack_bb", "pot_bb"] + (["board_cluster"] if "board_cluster" in self.df.columns else [])

        self.device = device

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # categorical dict
        x_cat: Dict[str, torch.Tensor] = {}
        for f in self.cat_features:
            x_cat[f] = torch.tensor(int(row[f]), dtype=torch.long)

        # continuous dict
        x_cont: Dict[str, torch.Tensor] = {}
        for f in self.cont_features:
            x_cont[f] = torch.tensor(float(row[f]), dtype=torch.float32).view(1)

        # board mask (optional)
        if "board_mask_52" in self.df.columns:
            bm = torch.tensor(row["board_mask_52"], dtype=torch.float32)
            x_cont["board_mask_52"] = bm
        else:
            x_cont["board_mask_52"] = torch.zeros(52, dtype=torch.float32)

        # targets by actor
        actor = str(row.get("actor", "ip")).lower()
        if actor not in ("ip", "oop"):
            actor = "ip"

        if self.has_soft:
            y_vec = self.y_soft[idx]
        else:
            y_vec = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
            y_vec[self.action_idx[idx]] = 1.0

        y_ip  = y_vec if actor == "ip"  else torch.zeros_like(y_vec)
        y_oop = y_vec if actor == "oop" else torch.zeros_like(y_vec)

        # masks (all ones unless you later add legality columns)
        m_ip = torch.ones(VOCAB_SIZE, dtype=torch.float32)
        m_oop = torch.ones(VOCAB_SIZE, dtype=torch.float32)

        w = self.weights[idx]

        return x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w


# ---------------- Collate ----------------

def postflop_policy_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cat_list, x_cont_list, y_ip_list, y_oop_list, m_ip_list, m_oop_list, w_list = zip(*batch)

    # categorical features
    x_cat: Dict[str, torch.Tensor] = {}
    for k in x_cat_list[0].keys():
        x_cat[k] = torch.stack([x[k] for x in x_cat_list], dim=0)

    # continuous features
    x_cont: Dict[str, torch.Tensor] = {}
    for k in x_cont_list[0].keys():
        x_cont[k] = torch.stack([x[k] for x in x_cont_list], dim=0)

    y_ip = torch.stack(y_ip_list, dim=0)
    y_oop = torch.stack(y_oop_list, dim=0)
    m_ip = torch.stack(m_ip_list, dim=0)
    m_oop = torch.stack(m_oop_list, dim=0)
    w = torch.as_tensor(w_list, dtype=torch.float32)

    return x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w