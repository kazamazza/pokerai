import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Sequence, List, Tuple, Dict, Any
from ml.datasets.rangenet import RangeNetDatasetParquet, canon_pos, canon_action, canon_ctx


class PreflopRangeDatasetParquet(RangeNetDatasetParquet):
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
        debug: bool = True,
    ):
        df = pd.read_parquet(str(parquet_path)).copy()
        n_in = len(df)

        # --- 1) Canonicalize categoricals (create *_c, validate, then swap in)
        if "hero_pos" in df.columns:        df["hero_pos_c"]        = df["hero_pos"].map(canon_pos)
        if "opener_pos" in df.columns:      df["opener_pos_c"]      = df["opener_pos"].map(canon_pos)
        if "opener_action" in df.columns:   df["opener_action_c"]   = df["opener_action"].map(canon_action)
        if "ctx" in df.columns:             df["ctx_c"]             = df["ctx"].map(canon_ctx)

        # STRICT: error out on any non-canonical values.
        # If strict_canon causes everything to be rejected downstream, we’ll relax below.
        def _strict_check(_df):
            offenders = {}
            for col, ccol in [("hero_pos","hero_pos_c"), ("opener_pos","opener_pos_c"),
                              ("opener_action","opener_action_c"), ("ctx","ctx_c")]:
                if col in _df.columns and ccol in _df.columns:
                    bad = _df[pd.isna(_df[ccol])]
                    if not bad.empty:
                        offenders[col] = bad[col].head(5).astype(str).tolist()
            return offenders

        offenders = _strict_check(df)
        if strict_canon and offenders:
            # Fail fast with a helpful message (before silent emptying).
            raise ValueError(f"Non-canonical values found: {offenders}. "
                             f"Either fix the input or set strict_canon=False temporarily.")

        # Swap in canonical columns if present
        for src, dst in [("hero_pos_c","hero_pos"), ("opener_pos_c","opener_pos"),
                         ("opener_action_c","opener_action"), ("ctx_c","ctx")]:
            if dst in df.columns and src in df.columns:
                df[dst] = df[src]
                df.drop(columns=[src], inplace=True)

        # --- 2) Ensure weight exists and filter if requested
        if weight_col not in df.columns:
            df[weight_col] = 1.0
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)  # default to 1.0, not 0.0
        n_before_w = len(df)
        if min_weight is not None:
            df = df[df[weight_col] >= float(min_weight)]
        n_after_w = len(df)

        # --- 3) Lock Y schema: y_0..y_168
        y_cols = [c for c in df.columns if re.fullmatch(r"y_\d{1,3}", c)]
        if not y_cols and "range_169" in df.columns:
            r = df["range_169"].apply(lambda v: list(v) if isinstance(v, (list, tuple)) else [0.0]*169)
            y_arr = np.vstack(r.values).astype("float32")
            for i in range(169):
                df[f"y_{i}"] = y_arr[:, i]
            df.drop(columns=["range_169"], inplace=True)
            y_cols = [f"y_{i}" for i in range(169)]

        expect = [f"y_{i}" for i in range(169)]
        missing = [c for c in expect if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns (expect y_0..y_168). Missing examples: {missing[:6]}")

        df[expect] = df[expect].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
        if clip_labels:
            df[expect] = df[expect].clip(lower=0.0)
        if normalize_labels:
            sums = df[expect].sum(axis=1).values
            nz = sums > eps
            if nz.any():
                df.loc[nz, expect] = (df.loc[nz, expect].values / sums[nz, None]).astype("float32")

        # --- 4) X types
        if "stack_bb" in df.columns:
            df["stack_bb"] = pd.to_numeric(df["stack_bb"], errors="coerce").fillna(0).astype("float32")

        # Optional: sanity print
        if debug:
            def vc(s):
                try: return s.value_counts().head(6).to_dict()
                except Exception: return {}
            print(
                "[preflop-ds] rows_in=", n_in,
                " rows_after_weight=", n_after_w,
                " weight_dropped=", (n_before_w - n_after_w),
            )
            if {"hero_pos","opener_pos","opener_action","ctx"}.issubset(df.columns):
                print("[preflop-ds] hero_pos=", vc(df["hero_pos"]),
                      " opener_pos=", vc(df["opener_pos"]),
                      " action=", vc(df["opener_action"]),
                      " ctx=", vc(df["ctx"]))

        # If we somehow ended with 0 rows (e.g., strict_canon in a different code path): relax once.
        if len(df) == 0 and strict_canon:
            print("[preflop-ds] strict_canon produced 0 rows; retrying with strict_canon=False fallback…")
            return PreflopRangeDatasetParquet(
                parquet_path=parquet_path,
                x_cols=x_cols,
                weight_col=weight_col,
                device=device,
                min_weight=min_weight,
                strict_canon=False,
                normalize_labels=normalize_labels,
                clip_labels=clip_labels,
                eps=eps,
                debug=debug,
            )

        # Persist normalized tmp parquet (what the base class will memory-map)
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


def rangenet_preflop_collate_fn(
    batch: List[Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Custom collate function for RangeNetPreflop datasets.
    Combines a batch of tuples (x_dict, y, w) into tensors suitable for Lightning.

    Each x_dict maps feature_name → 1D LongTensor[B].
    The collate function stacks each feature across the batch to produce
    x_out[feature_name] = LongTensor[B], where B = batch size.

    Args:
        batch: list of (x_dict, y, w)
               x_dict: dict[str, Tensor]  (categorical inputs)
               y: FloatTensor[169]        (target range probabilities)
               w: FloatTensor[]           (per-sample weight)

    Returns:
        (x_out, y_out, w_out)
        - x_out: dict[str, LongTensor[B]]
        - y_out: FloatTensor[B, 169]
        - w_out: FloatTensor[B]
    """
    x_list, y_list, w_list = zip(*batch)

    # Collect all feature names
    all_keys = set()
    for x in x_list:
        all_keys.update(x.keys())

    # Stack features
    x_out: Dict[str, torch.Tensor] = {}
    for k in all_keys:
        vals = []
        for x in x_list:
            v = x.get(k)
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=torch.long)
            vals.append(v.view(-1))
        x_out[k] = torch.cat(vals, dim=0) if vals[0].dim() == 1 else torch.stack(vals, dim=0)

    # Stack targets (y) and weights (w)
    y_out = torch.stack([torch.as_tensor(y, dtype=torch.float32) for y in y_list], dim=0)
    w_out = torch.as_tensor([float(w) for w in w_list], dtype=torch.float32).view(-1)

    return x_out, y_out, w_out