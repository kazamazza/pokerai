# ml/datasets/rangenet/postflop_rangenet.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
import torch

from ml.datasets.rangenet import RangeNetDatasetParquet, canon_pos, canon_ctx

# Minimal street normalizer (swap for your own if you have one)
_STREET_MAP = {
    1: 1, 2: 2, 3: 3,
    "FLOP": 1, "TURN": 2, "RIVER": 3,
    "flop": 1, "turn": 2, "river": 3,
}
def canon_street(x) -> Optional[int]:
    return _STREET_MAP.get(x, None)

class PostflopRangeDatasetParquet(RangeNetDatasetParquet):
    """
    Locked-schema postflop dataset.

    Expected Parquet columns:
      X:
        - stack_bb        (int)
        - pot_bb          (int/float)  -- pot size entering the street
        - hero_pos        (str in {UTG,HJ,CO,BTN,SB,BB})
        - ip_pos          (str in {UTG,HJ,CO,BTN,SB,BB})
        - oop_pos         (str in {UTG,HJ,CO,BTN,SB,BB})
        - street          (FLOP/TURN/RIVER or 1/2/3)
        - ctx             (int or str from your Ctx)
        - board_cluster   (int; your board bucket id)
      Y: y_0..y_168       (float distribution over 169 hand classes)
      W: weight           (float, optional semantics ~ sample weight)

    Notes:
      - This class normalizes categorical fields (pos, ctx, street) before
        delegating to the generic RangeNetDatasetParquet.
      - It enforces presence and validity if strict_canon=True.
    """
    DEFAULT_X = [
        "stack_bb", "pot_bb",
        "hero_pos", "ip_pos", "oop_pos",
        "street", "ctx", "board_cluster",
    ]

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        x_cols: Optional[Sequence[str]] = None,
        weight_col: str = "weight",
        device: Optional[torch.device] = None,
        min_weight: Optional[float] = None,
        strict_canon: bool = True,
    ):
        df = pd.read_parquet(str(parquet_path)).copy()

        # --- Canonicalize positions / context / street ---
        if "hero_pos" in df.columns:
            df["hero_pos_c"] = df["hero_pos"].map(canon_pos)
        if "ip_pos" in df.columns:
            df["ip_pos_c"] = df["ip_pos"].map(canon_pos)
        if "oop_pos" in df.columns:
            df["oop_pos_c"] = df["oop_pos"].map(canon_pos)
        if "ctx" in df.columns:
            df["ctx_c"] = df["ctx"].map(canon_ctx)
        if "street" in df.columns:
            df["street_c"] = df["street"].map(canon_street)

        # Optional strict validation
        if strict_canon:
            for col, canon_col in [
                ("hero_pos","hero_pos_c"),
                ("ip_pos","ip_pos_c"),
                ("oop_pos","oop_pos_c"),
                ("ctx","ctx_c"),
                ("street","street_c"),
            ]:
                if col in df.columns and canon_col in df.columns:
                    bad = df[pd.isna(df[canon_col])]
                    if not bad.empty:
                        raise ValueError(f"Non-canonical or missing values in {col}: {len(bad)} bad row(s).")

        # Swap canonical columns in-place
        for src, dst in [
            ("hero_pos_c","hero_pos"),
            ("ip_pos_c","ip_pos"),
            ("oop_pos_c","oop_pos"),
            ("ctx_c","ctx"),
            ("street_c","street"),
        ]:
            if dst in df.columns and src in df.columns:
                df[dst] = df[src]; df.drop(columns=[src], inplace=True)

        # Ensure numeric fields present and sane
        for num_col in ["stack_bb", "pot_bb", "board_cluster"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        # Persist a normalized snapshot for the base class to read once
        tmp_path = Path(parquet_path).with_suffix(".postflop.norm.parquet")
        df.to_parquet(tmp_path, index=False)

        super().__init__(
            parquet_path=tmp_path,
            x_cols=list(x_cols) if x_cols else list(self.DEFAULT_X),
            weight_col=weight_col,
            device=device,
            min_weight=min_weight,
        )