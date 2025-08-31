from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Sequence
import pandas as pd
import torch
from ml.datasets.rangenet import RangeNetDatasetParquet, canon_pos, canon_action, canon_ctx


class PreflopRangeDatasetParquet(RangeNetDatasetParquet):
    """
    Locked-schema preflop dataset.

    Expected Parquet columns:
      X:
        - stack_bb          (int)
        - hero_pos          (str in {UTG,HJ,CO,BTN,SB,BB})
        - opener_pos        (str in {UTG,HJ,CO,BTN,SB,BB})
        - ctx               (int or str from your Ctx)
        - opener_action     (str like RAISE/ALL_IN/LIMP/3BET/...)
      Y: y_0..y_168         (float)
      W: weight             (float, optional semantics ~ sample weight)
    """
    DEFAULT_X = ["stack_bb","hero_pos","opener_pos","ctx","opener_action"]

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
        # Pre-clean to ensure consistent categorical space
        df = pd.read_parquet(str(parquet_path)).copy()

        if "hero_pos" in df.columns:
            df["hero_pos_c"] = df["hero_pos"].map(canon_pos)
        if "opener_pos" in df.columns:
            df["opener_pos_c"] = df["opener_pos"].map(canon_pos)
        if "opener_action" in df.columns:
            df["opener_action_c"] = df["opener_action"].map(canon_action)
        if "ctx" in df.columns:
            df["ctx_c"] = df["ctx"].map(canon_ctx)

        # sanity / optional strictness
        if strict_canon:
            for col, canon_col in [
                ("hero_pos","hero_pos_c"),
                ("opener_pos","opener_pos_c"),
                ("opener_action","opener_action_c"),
                ("ctx","ctx_c"),
            ]:
                if col in df.columns and canon_col in df.columns:
                    bad = df[pd.isna(df[canon_col])]
                    if not bad.empty:
                        raise ValueError(f"Non-canonical or missing values in {col}: found {len(bad)} bad row(s).")

        # swap-in canonical columns (if present)
        for src, dst in [
            ("hero_pos_c","hero_pos"),
            ("opener_pos_c","opener_pos"),
            ("opener_action_c","opener_action"),
            ("ctx_c","ctx"),
        ]:
            if dst in df.columns and src in df.columns:
                df[dst] = df[src]; df.drop(columns=[src], inplace=True)

        # Persist a temp normalized parquet (so parent ctor can read once)
        tmp_path = Path(parquet_path).with_suffix(".preflop.norm.parquet")
        df.to_parquet(tmp_path, index=False)

        super().__init__(
            parquet_path=tmp_path,
            x_cols=list(x_cols) if x_cols else list(self.DEFAULT_X),
            weight_col=weight_col,
            device=device,
            min_weight=min_weight,
        )