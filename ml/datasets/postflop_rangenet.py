import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

CAT_FEATURES_DEFAULT = ["hero_pos", "ip_pos", "oop_pos", "ctx", "street"]


class PostflopPolicyDatasetParquet(Dataset):
    SOFT_Y_COLS: List[str] = []
    CAT_FEATURES: List[str] = CAT_FEATURES_DEFAULT

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
        self.device = device

        # Determine which soft columns we expect from ACTION_VOCAB
        if not self.SOFT_Y_COLS:
            # Import here to avoid hard import at module import time.
            from ml.models.postflop_policy_net import ACTION_VOCAB as _VOC
            self.SOFT_Y_COLS = list(_VOC)

        def canon_pos(v: str) -> str:
            """Canonicalize position strings (UTG+, CO, BTN, SB, BB)."""
            if v is None:
                return None
            v = str(v).strip().upper()
            aliases = {
                "UTG+1": "HJ", "UTG+2": "CO",
                "UTG1": "HJ", "UTG2": "CO",
                "DEALER": "BTN", "BUTTON": "BTN",
                "SMALL BLIND": "SB", "BIG BLIND": "BB",
            }
            return aliases.get(v, v)

        def canon_ctx(v: str) -> str:
            """Canonicalize context strings (optional)."""
            if v is None:
                return None
            return str(v).strip().upper()

        def canon_street(v: str | int) -> str:
            """Canonicalize street to FLOP/TURN/RIVER (accept int 1/2/3)."""
            if v is None:
                return None
            if isinstance(v, int):
                return {1: "FLOP", 2: "TURN", 3: "RIVER"}.get(v, str(v))
            v = str(v).strip().upper()
            aliases = {"1": "FLOP", "2": "TURN", "3": "RIVER"}
            return aliases.get(v, v)

        _canon_mod = {
            "canon_pos": canon_pos,
            "canon_ctx": canon_ctx,
            "canon_street": canon_street,
        }

        # -------- Canonicalize categorical fields (string -> canonical token) --------
        # We try existing canon_* helpers; if not present, we pass-through and factorize later.
        def _maybe_apply(series: pd.Series, fn_name: str) -> pd.Series:
            try:
                fn = _canon_mod.get(fn_name)
            except Exception:
                fn = None
            if callable(fn):
                return series.map(fn)
            return series

        for col, fn in [
            ("hero_pos", "canon_pos"),
            ("ip_pos", "canon_pos"),
            ("oop_pos", "canon_pos"),
            ("ctx", "canon_ctx"),
            ("street", "canon_street"),
        ]:
            if col in self.df.columns:
                self.df[col] = _maybe_apply(self.df[col], fn)

        # -------- Factorize any non-integer categorical columns to stable ids --------
        self._id_maps: Dict[str, Dict[str, int]] = {}

        def _ensure_id(col: str):
            if col not in self.df.columns:
                return
            # already integer-like?
            if pd.api.types.is_integer_dtype(self.df[col]):
                return
            vals = self.df[col].astype(str).fillna("__NA__")
            uniq = sorted(vals.unique().tolist())
            id_map = {v: i for i, v in enumerate(uniq)}
            self._id_maps[col] = id_map
            self.df[col] = vals.map(id_map)

        for c in self.CAT_FEATURES:
            _ensure_id(c)

        if strict_canon:
            for c in self.CAT_FEATURES:
                if c in self.df.columns:
                    bad = self.df[self.df[c].isna()]
                    if not bad.empty:
                        raise ValueError(f"Invalid/uncanonical values in '{c}': {len(bad)} row(s).")

        # -------- Weights --------
        if weight_col in self.df.columns:
            w = self.df[weight_col].astype(float).values
            w[np.isnan(w)] = 1.0
            self.weights = torch.tensor(w, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(self.df), dtype=torch.float32)

        # -------- Targets (prefer soft per-action columns) --------
        has_soft = all(c in self.df.columns for c in self.SOFT_Y_COLS)
        self.has_soft = has_soft

        if not has_soft and "action" not in self.df.columns:
            raise ValueError("Policy parquet must have either per-action columns (soft) or an 'action' column (hard).")

        if has_soft:
            y = torch.tensor(self.df[self.SOFT_Y_COLS].values, dtype=torch.float32)
            self.y_soft = y / y.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            # map hard action -> index -> one-hot at __getitem__
            from ml.models.postflop_policy_net import VOCAB_INDEX
            act_idx = self.df["action"].astype(str).str.upper().map(VOCAB_INDEX.get)
            if act_idx.isna().any():
                bad = sorted(self.df.loc[act_idx.isna(), "action"].astype(str).str.upper().unique().tolist())
                raise ValueError(f"Unmapped actions found: {bad}")
            self.action_idx = torch.tensor(act_idx.values, dtype=torch.long)

        # -------- Feature lists to emit --------
        self.cat_features = [c for c in self.CAT_FEATURES if c in self.df.columns]
        self.cont_features = ["stack_bb", "pot_bb"]
        if "board_cluster" in self.df.columns:
            self.cont_features.append("board_cluster")

        # Pre-calc a boolean for board mask presence
        self._has_board_mask = "board_mask_52" in self.df.columns

    def cards(self) -> Dict[str, int]:
        """
        Return vocab sizes for categorical features (max id + 1). Works for both
        factorized (self._id_maps) and already-integer-coded columns.
        """
        sizes = {}
        for c in self.cat_features:
            if c in self._id_maps:
                sizes[c] = len(self._id_maps[c])
            else:
                sizes[c] = int(self.df[c].max()) + 1
        return sizes

    # ---- public helpers ----
    def id_maps(self) -> Dict[str, Dict[str, int]]:
        """Return categorical id maps (useful for sidecars)."""
        return dict(self._id_maps)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ---- x_cat: categorical ids (already factorized to ints) ----
        x_cat: Dict[str, torch.Tensor] = {}
        for f in self.cat_features:
            x_cat[f] = torch.tensor(int(row[f]), dtype=torch.long)

        # ---- x_cont: numeric features ----
        x_cont: Dict[str, torch.Tensor] = {}
        for f in self.cont_features:
            x_cont[f] = torch.tensor(float(row[f]), dtype=torch.float32).view(1)

        # Optional board mask
        if self._has_board_mask:
            bm = row["board_mask_52"]
            # tolerate list/np.ndarray/torch.Tensor
            if isinstance(bm, torch.Tensor):
                bm_t = bm.float().view(-1)
            else:
                bm_t = torch.tensor(np.asarray(bm, dtype=np.float32)).view(-1)
            if bm_t.numel() != 52:
                # keep shape consistent (failsafe)
                pad = torch.zeros(52, dtype=torch.float32)
                pad[: min(52, bm_t.numel())] = bm_t[: min(52, bm_t.numel())]
                bm_t = pad
            x_cont["board_mask_52"] = bm_t
        else:
            x_cont["board_mask_52"] = torch.zeros(52, dtype=torch.float32)

        # ---- targets and masks, by actor ----
        actor = str(row.get("actor", "ip")).lower()
        if actor not in ("ip", "oop"):
            actor = "ip"

        from ml.models.postflop_policy_net import VOCAB_SIZE

        if self.has_soft:
            y_vec = self.y_soft[idx]
        else:
            y_vec = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
            y_vec[self.action_idx[idx]] = 1.0

        if actor == "ip":
            y_ip, y_oop = y_vec, torch.zeros_like(y_vec)
            m_ip = torch.ones(VOCAB_SIZE, dtype=torch.float32)
            m_oop = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        else:
            y_ip, y_oop = torch.zeros_like(y_vec), y_vec
            m_ip = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
            m_oop = torch.ones(VOCAB_SIZE, dtype=torch.float32)

        w = self.weights[idx]

        return x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w

def postflop_policy_collate_fn(
    batch: List[
        Tuple[
            Dict[str, torch.Tensor],  # x_cat
            Dict[str, torch.Tensor],  # x_cont
            torch.Tensor,             # y_ip
            torch.Tensor,             # y_oop
            torch.Tensor,             # m_ip
            torch.Tensor,             # m_oop
            torch.Tensor,             # w
        ]
    ]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cat_list, x_cont_list, y_ip_list, y_oop_list, m_ip_list, m_oop_list, w_list = zip(*batch)

    # categorical features
    x_cat: Dict[str, torch.Tensor] = {k: torch.stack([x[k] for x in x_cat_list], dim=0)
                                      for k in x_cat_list[0].keys()}

    # continuous features
    x_cont: Dict[str, torch.Tensor] = {k: torch.stack([x[k] for x in x_cont_list], dim=0)
                                       for k in x_cont_list[0].keys()}
    x_cont["eff_stack_bb"] = x_cont.pop("stack_bb")

    y_ip  = torch.stack(y_ip_list,  dim=0)
    y_oop = torch.stack(y_oop_list, dim=0)
    m_ip  = torch.stack(m_ip_list,  dim=0)
    m_oop = torch.stack(m_oop_list, dim=0)
    w     = torch.stack(w_list,     dim=0).float()

    return x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w