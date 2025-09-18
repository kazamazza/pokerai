import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ml.datasets.rangenet import RangeNetDatasetParquet, canon_pos, canon_action, canon_ctx


class PreflopRangeDatasetParquet(RangeNetDatasetParquet):
    """
    Locked-schema preflop dataset.

    Expected Parquet columns:
      X:
        - stack_bb          (int/float)
        - hero_pos          (str in {UTG,HJ,CO,BTN,SB,BB})
        - opener_pos        (str in {UTG,HJ,CO,BTN,SB,BB})
        - ctx               (int or str)
        - opener_action     (str like RAISE/ALL_IN/LIMP/3BET/...)
      Y:
        - y_0 .. y_168      (float, frequencies or probs)
      W:
        - weight            (float, optional)
    """
    DEFAULT_X = ["stack_bb", "hero_pos", "opener_pos", "ctx", "opener_action"]

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        x_cols: Optional[Sequence[str]] = None,
        weight_col: str = "weight",
        device: Optional[torch.device] = None,
        min_weight: Optional[float] = None,
        strict_canon: bool = True,
        normalize_labels: bool = True,
        clip_labels: bool = True,
        eps: float = 1e-8,
    ):
        df = pd.read_parquet(str(parquet_path)).copy()

        # --- 1) Canonicalize categoricals (create *_c, validate, then swap in)
        if "hero_pos" in df.columns:
            df["hero_pos_c"] = df["hero_pos"].map(canon_pos)
        if "opener_pos" in df.columns:
            df["opener_pos_c"] = df["opener_pos"].map(canon_pos)
        if "opener_action" in df.columns:
            df["opener_action_c"] = df["opener_action"].map(canon_action)
        if "ctx" in df.columns:
            df["ctx_c"] = df["ctx"].map(canon_ctx)

        if strict_canon:
            for col, ccol in [
                ("hero_pos", "hero_pos_c"),
                ("opener_pos", "opener_pos_c"),
                ("opener_action", "opener_action_c"),
                ("ctx", "ctx_c"),
            ]:
                if col in df.columns and ccol in df.columns:
                    bad = df[pd.isna(df[ccol])]
                    if not bad.empty:
                        # show a couple offenders to help debugging
                        samples = bad[[col]].head(3).to_dict(orient="records")
                        raise ValueError(f"Non-canonical or missing values in {col}: {len(bad)} row(s), e.g. {samples}")

        for src, dst in [
            ("hero_pos_c", "hero_pos"),
            ("opener_pos_c", "opener_pos"),
            ("opener_action_c", "opener_action"),
            ("ctx_c", "ctx"),
        ]:
            if dst in df.columns and src in df.columns:
                df[dst] = df[src]
                df.drop(columns=[src], inplace=True)

        # --- 2) Ensure weight exists and filter if requested
        if weight_col not in df.columns:
            df[weight_col] = 1.0
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
        if min_weight is not None:
            df = df[df[weight_col] >= float(min_weight)]

        # --- 3) Lock Y schema: y_0..y_168 as float32, clipped and (optionally) normalized
        y_cols = [c for c in df.columns if re.fullmatch(r"y_\d{1,3}", c)]
        # If not present, try older schema like 'range_169' list -> explode
        if not y_cols and "range_169" in df.columns:
            r = df["range_169"].apply(lambda v: list(v) if isinstance(v, (list, tuple)) else [0.0]*169)
            y_arr = np.vstack(r.values).astype("float32")
            for i in range(169):
                df[f"y_{i}"] = y_arr[:, i]
            df.drop(columns=["range_169"], inplace=True)
            y_cols = [f"y_{i}" for i in range(169)]

        # assert exactly 169
        expect = [f"y_{i}" for i in range(169)]
        missing = [c for c in expect if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing[:5]}{'...' if len(missing)>5 else ''}")

        df[expect] = df[expect].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

        if clip_labels:
            df[expect] = df[expect].clip(lower=0.0)

        if normalize_labels:
            sums = df[expect].sum(axis=1).values
            nz = sums > eps
            df.loc[nz, expect] = (df.loc[nz, expect].values / sums[nz, None]).astype("float32")
            # if rows sum to 0 (degenerate), leave zeros; the KL will ignore via smoothing/upstream checks

        # --- 4) Ensure X types are consistent
        if "stack_bb" in df.columns:
            df["stack_bb"] = pd.to_numeric(df["stack_bb"], errors="coerce").fillna(0).astype("float32")

        # Persist a normalized temp parquet so the parent can mmap it fast
        tmp_path = Path(parquet_path).with_suffix(".preflop.norm.parquet")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path, index=False)

        super().__init__(
            parquet_path=tmp_path,
            x_cols=list(x_cols) if x_cols else list(self.DEFAULT_X),
            weight_col=weight_col,
            device=device,
            min_weight=min_weight,
        )