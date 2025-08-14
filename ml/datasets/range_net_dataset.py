import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from utils.encoders import encode_position, encode_category


class RangeNetDataset(Dataset):
    """
    PyTorch dataset for postflop GTO strategy templates.
    Converts JSON files to contextual feature vectors and action label vectors.
    """

    def __init__(self, strategy_dir: Path):
        self.strategy_paths = list(strategy_dir.rglob("*.json"))
        self.villain_profiles = ["GTO", "FISH", "NIT", "MANIAC", "REGULAR", "RECREATIONAL"]
        self.exploit_settings = ["GTO", "EXPLOIT_LIGHT", "EXPLOIT_HARD"]
        self.multiway_contexts = ["HU", "3WAY", "4WAY"]
        self.population_types = ["REGULAR", "RECREATIONAL"]
        self.action_contexts = ["OPEN", "VS_LIMP", "VS_OPEN", "VS_ISO", "VS_3BET", "VS_4BET"]

    def __len__(self) -> int:
        return len(self.strategy_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.strategy_paths[idx]
        with open(path, "r") as f:
            data = json.load(f)

        meta = data["meta"]

        # === Feature vector ===
        features = [
            encode_position(meta["ip_position"]),
            encode_position(meta["oop_position"]),
            meta["stack_bb"] / 300.0,  # Normalize stack
            encode_category(meta["villain_profile"], self.villain_profiles),
            encode_category(meta["exploit_setting"], self.exploit_settings),
            encode_category(meta["multiway_context"], self.multiway_contexts),
            encode_category(meta["population_type"], self.population_types),
            encode_category(meta["action_context"], self.action_contexts),
            data["cluster_id"] / 128.0  # Normalize flop cluster ID
        ]

        # === Label vector: IP action distribution ===
        ip_actions = data["ip_strategy"]["actions"]
        label_vector = [0.0, 0.0, 0.0]  # [CHECK, BET_SMALL, BET_LARGE]

        for action in ip_actions:
            act = action["action"]
            size = action.get("size")
            freq = action["frequency"]

            if act == "CHECK":
                label_vector[0] += freq
            elif size is not None and size <= 0.5:
                label_vector[1] += freq
            elif size is not None and size > 0.5:
                label_vector[2] += freq

        # Normalize label vector
        total = sum(label_vector)
        label_vector = [v / total if total > 0 else 0.0 for v in label_vector]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label_vector, dtype=torch.float32)