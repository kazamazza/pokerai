# tools/exploitnet/node_dataset.py
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

NUMERIC_FEATS = ["spr", "vpip", "pfr", "three_bet", "pot_bb", "hands_obs"]
CAT_FEATS = ["street", "villain_pos"]

class ExploitNodeDataset(Dataset):
    """
    Loads node-level parquet (flatten_nodes output).
    Produces per-sample (x_num_tensor, x_cat_idx_dict, y_long, weight)
    For simplicity the trainer will collate numeric+categorical into tensors.
    """
    def __init__(self, parquet_path: str | Path, min_weight: float = 0.0, fit_scaler: bool = True):
        df = pd.read_parquet(str(parquet_path))
        # Basic sanity
        if df.empty:
            raise RuntimeError("ExploitNodeDataset: empty parquet")

        # Keep only rows with valid y_action 0/1/2
        df = df[df["y_action"].isin([0,1,2])].copy().reset_index(drop=True)

        # Fill missing numeric with zeros
        for c in NUMERIC_FEATS:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

        # Ensure cat columns exist
        for c in CAT_FEATS:
            if c not in df.columns:
                df[c] = "UNK"

        # weight defaults
        if "w" not in df.columns:
            df["w"] = 1.0
        df["w"] = pd.to_numeric(df["w"], errors="coerce").fillna(0.0).astype(float)
        if min_weight and min_weight > 0:
            df = df[df["w"] >= float(min_weight)].reset_index(drop=True)

        self.df = df

        # build categorical maps
        self.cat_maps = {}
        for c in CAT_FEATS:
            vals = pd.Series(self.df[c].astype(str).fillna("UNK").unique()).tolist()
            # reserve 0 for PAD/UNK
            mapping = {v: i+1 for i, v in enumerate(sorted(vals))}
            mapping["UNK"] = 0
            self.cat_maps[c] = mapping
            # map to indices
            self.df[c + "_idx"] = self.df[c].astype(str).map(lambda x: mapping.get(x, 0)).astype(int)

        # numeric scaler (simple mean/std)
        self.num_mean = self.df[NUMERIC_FEATS].mean().values.astype(np.float32)
        self.num_std = self.df[NUMERIC_FEATS].std().replace(0.0, 1.0).values.astype(np.float32)

        # store arrays
        self.x_num = ((self.df[NUMERIC_FEATS].values.astype("float32") - self.num_mean) / self.num_std)
        self.x_cat = self.df[[c + "_idx" for c in CAT_FEATS]].values.astype("int64")
        self.y = self.df["y_action"].astype("int64").values
        self.w = self.df["w"].astype("float32").values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        num = torch.from_numpy(self.x_num[idx])
        cat = torch.from_numpy(self.x_cat[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        w = torch.tensor(float(self.w[idx]), dtype=torch.float32)
        return {"num": num, "cat": cat, "y": y, "w": w}

    # utilities for trainer sidecar
    def cardinals(self):
        """Return cardinals for each categorical feature (including UNK=0 slot)."""
        return {c: max(v)+1 for c,v in ((k, list(self.cat_maps[k].values())) for k in self.cat_maps)}

    def numeric_stats(self):
        return {"mean": self.num_mean.tolist(), "std": self.num_std.tolist()}