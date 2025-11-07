from typing import Dict, List, Optional, Sequence, Tuple
import warnings, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.models.vocab_actions import FACING_ACTION_VOCAB  # ["FOLD","CALL","RAISE_150",...,"ALLIN"]

FACING_VOCAB = list(FACING_ACTION_VOCAB)
FACING_INDEX = {a: i for i, a in enumerate(FACING_VOCAB)}
V_FACING = len(FACING_VOCAB)

ROOT_PREFIXES = ("CHECK", "BET_", "DONK_")  # columns to drop if they sneak in


class PostflopPolicyDatasetFacing(Dataset):
    """
    Facing-only postflop policy dataset (OOP responding to a bet on flop).
    Emits (x_cat, x_cont, y, m, w) with y/m width == len(FACING_ACTION_VOCAB).
    Automatically ensures 'size_frac' exists (as faced bet fraction) and keeps it in cont_features.
    """
    def __init__(
        self,
        parquet_path: str | pd.Series,
        *,
        cat_features: List[str],
        cont_features: List[str],
        weight_col: str = "weight",
        strict_canon: bool = True,
        allowed_raises: Optional[List[int]] = None,   # e.g. [150,200,300]
        allow_allin: bool = True,
    ):
        self.parquet_path = parquet_path
        df = pd.read_parquet(str(parquet_path)).copy()

        # --- canonicalize some common categoricals ---
        def _canon_pos(v):
            if v is None: return None
            v = str(v).strip().upper()
            alias = {
                "UTG+1":"HJ","UTG+2":"CO","UTG1":"HJ","UTG2":"CO",
                "DEALER":"BTN","BUTTON":"BTN","SMALL BLIND":"SB","BIG BLIND":"BB",
            }
            return alias.get(v, v)
        def _canon_ctx(v): return None if v is None else str(v).strip().upper()

        for col, fn in {"hero_pos": _canon_pos, "ip_pos": _canon_pos,
                        "oop_pos": _canon_pos, "ctx": _canon_ctx}.items():
            if col in df.columns:
                df[col] = df[col].map(fn)

        if "board_cluster" in df.columns and "board_cluster_id" not in df.columns:
            df = df.rename(columns={"board_cluster": "board_cluster_id"})

        self.cat_features  = list(cat_features or [])
        self.cont_features = list(cont_features or [])
        self.weight_col    = weight_col

        # --- faced bet as size_frac ---
        if "size_frac" not in df.columns:
            if "size_pct" in df.columns:
                df["size_frac"] = pd.to_numeric(df["size_pct"], errors="coerce") / 100.0
            elif "faced_size_pct" in df.columns:
                df["size_frac"] = pd.to_numeric(df["faced_size_pct"], errors="coerce") / 100.0
            elif "faced_size_frac" in df.columns:
                df["size_frac"] = pd.to_numeric(df["faced_size_frac"], errors="coerce")
            else:
                df["size_frac"] = np.nan

        df["size_frac"] = (
            df["size_frac"]
            .astype("float32")
            .fillna(0.0)
            .clip(lower=0.0, upper=1.5)
        )
        if "size_frac" not in self.cont_features:
            self.cont_features.append("size_frac")

        # --- drop root-ish columns; enforce facing label space ---
        drop_cols = [c for c in df.columns if any(c.startswith(p) for p in ROOT_PREFIXES)]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        self.target_cols = FACING_VOCAB[:]  # exact facing label space
        for tok in self.target_cols:
            if tok not in df.columns:
                df[tok] = 0.0

        # --- validate required metas/features exist ---
        needed_meta = set(self.cat_features + self.cont_features + [self.weight_col])
        missing_meta = [c for c in needed_meta if c not in df.columns]
        if missing_meta:
            raise ValueError(f"Facing parquet missing columns: {missing_meta}")

        # --- encode categoricals ---
        self._id_maps: dict[str, dict[str, int]] = {}
        self._cards: dict[str, int] = {}

        def _encode_cat(col: str):
            nonlocal df
            toks = df[col].fillna("__UNK__").astype(str).str.upper()
            uniq = sorted(toks.unique().tolist())
            tok2id = {t: i for i, t in enumerate(uniq)}
            self._id_maps[col] = tok2id
            self._cards[col]   = max(1, len(tok2id))
            df[col] = toks.map(tok2id)

        for c in self.cat_features:
            _encode_cat(c)

        if strict_canon:
            for c in self.cat_features:
                if df[c].isna().any():
                    raise ValueError(f"Invalid values in '{c}' (NaNs after encoding)")

        # --- weights ---
        w = df[self.weight_col].astype(float).to_numpy()
        w = np.where(np.isnan(w), 1.0, w)
        self.weights = torch.tensor(w, dtype=torch.float32)

        # --- legality config used in __getitem__ ---
        self.allowed_raises = list(allowed_raises or [150, 200, 300])
        self.allow_allin = bool(allow_allin)

        self._df = df.reset_index(drop=True)

    @staticmethod
    def _to_mask_52(v) -> torch.Tensor:
        # robustly convert Avro/pyarrow or JSON-serialized arrays to a 52-d float tensor
        def _to_list(x):
            if hasattr(x, "as_py"):
                try: x = x.as_py()
                except Exception: pass
            if isinstance(x, str):
                s = x.strip()
                if s.startswith("[") and s.endswith("]"):
                    try: return json.loads(s)
                    except Exception: return None
                return None
            if isinstance(x, (list, tuple)):
                out = []
                for it in x:
                    if hasattr(it, "as_py"):
                        try: it = it.as_py()
                        except Exception: pass
                    if isinstance(it, dict) and "element" in it:
                        out.append(it["element"])
                    else:
                        out.append(it)
                return out
            return x

        arr = _to_list(v)
        try:
            a = np.asarray(arr, dtype=np.float32).reshape(-1) if arr is not None else np.zeros(52, np.float32)
        except Exception:
            a = np.zeros(52, np.float32)
        out = np.zeros(52, dtype=np.float32)
        n = min(52, a.size)
        if n > 0: out[:n] = a[:n]
        return torch.from_numpy(out)

    def _legal_tokens(self) -> List[str]:
        toks = ["FOLD", "CALL"]
        for r in self.allowed_raises:
            tok = f"RAISE_{int(r)}"
            if tok in FACING_INDEX:
                toks.append(tok)
        if self.allow_allin and "ALLIN" in FACING_INDEX:
            toks.append("ALLIN")
        return toks

    @property
    def id_maps(self) -> dict[str, dict[str, int]]:
        return self._id_maps

    @property
    def cards(self) -> dict[str, int]:
        return dict(self._cards)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]

        # Categoricals
        x_cat: Dict[str, torch.Tensor] = {}
        for f in self.cat_features:
            v = int(row[f]) if f in row and pd.notna(row[f]) else 0
            x_cat[f] = torch.tensor(v, dtype=torch.long)

        # Continuous (incl. board_mask_52)
        x_cont: Dict[str, torch.Tensor] = {}
        for f in self.cont_features:
            if f == "board_mask_52":
                x_cont["board_mask_52"] = self._to_mask_52(row.get("board_mask_52", None))
            else:
                v = row[f] if f in row else 0.0
                try:
                    fv = float(v)
                    if np.isnan(fv): fv = 0.0
                except Exception:
                    fv = 0.0
                x_cont[f] = torch.tensor(fv, dtype=torch.float32).view(1)

        # Legal mask (facing)
        m = torch.zeros(V_FACING, dtype=torch.float32)
        for tok in self._legal_tokens():
            i = FACING_INDEX.get(tok)
            if i is not None:
                m[i] = 1.0

        # Targets: use FACING_VOCAB columns only
        y_raw = np.array([float(row.get(tok, 0.0) or 0.0) for tok in self.target_cols], dtype=np.float32)
        y = torch.from_numpy(y_raw) * m
        s = float(y.sum().item())
        if s <= 1e-12:
            y.zero_()
            ci = FACING_INDEX.get("CALL", None)
            if ci is not None:
                y[ci] = 1.0
        else:
            y = y / s

        w = self.weights[idx]
        return x_cat, x_cont, y, m, w


def postflop_policy_facing_collate_fn(
    batch: List[
        Tuple[
            Dict[str, torch.Tensor],  # x_cat
            Dict[str, torch.Tensor],  # x_cont
            torch.Tensor,             # y
            torch.Tensor,             # m
            torch.Tensor,             # w
        ]
    ]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cat_list, x_cont_list, y_list, m_list, w_list = zip(*batch)

    x_cat = {k: torch.stack([x[k] for x in x_cat_list], dim=0)
             for k in x_cat_list[0].keys()} if x_cat_list and x_cat_list[0] else {}

    x_cont: Dict[str, torch.Tensor] = {}
    if x_cont_list and x_cont_list[0]:
        for k in x_cont_list[0].keys():
            vals = [x[k] for x in x_cont_list]
            if k == "board_mask_52":
                x_cont[k] = torch.stack([v.to(torch.float32).view(52) for v in vals], dim=0)
            else:
                x_cont[k] = torch.stack([v.to(torch.float32).view(1) for v in vals], dim=0)

    y = torch.stack(y_list, dim=0)  # [B, V_facing]
    m = torch.stack(m_list, dim=0)  # [B, V_facing]
    w = torch.stack(w_list, dim=0).float()  # [B]

    if y.shape != m.shape or y.shape[1] != V_FACING:
        raise RuntimeError(f"Facing shapes mismatch: y={y.shape}, m={m.shape}, V_facing={V_FACING}")

    return x_cat, x_cont, y, m, w