# file: ml/data/ev/ev_parquet_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import importlib
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _import_symbol(path: str) -> Any:
    """
    Import "pkg.mod:SYMBOL" or "pkg.mod.SYMBOL" and return the symbol.
    """
    if not path:
        raise ValueError("empty import path")
    if ":" in path:
        mod, sym = path.split(":", 1)
    else:
        # last dot as separator
        i = path.rfind(".")
        if i <= 0:
            raise ValueError(f"cannot parse import path: {path}")
        mod, sym = path[:i], path[i + 1 :]
    return getattr(importlib.import_module(mod), sym)


def _ensure_list_str(x: Iterable[Any]) -> List[str]:
    return [str(a) for a in list(x)]


def _board_mask_from_row(row: Mapping[str, Any]) -> Optional[List[float]]:
    """
    Accept either:
      - columns bm0..bm51
      - a single 'board_mask_52' column containing a 52-length sequence
    Returns a python list[float] length 52, or None if not present.
    """
    # vector column
    if "board_mask_52" in row:
        v = row["board_mask_52"]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 52:
            return [float(x) for x in v]
    # expanded columns
    keys = [f"bm{i}" for i in range(52)]
    if all(k in row for k in keys):
        return [float(row[k]) for k in keys]
    return None


def _hashable_strat_key(df: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    """Build a hashable stratification key from multiple columns (as strings)."""
    if not keys:
        return np.zeros(len(df), dtype=np.int64)
    cols = []
    for k in keys:
        v = df[k] if k in df.columns else pd.Series([""], index=df.index)
        cols.append(v.astype(str))
    cat = pd.util.hash_pandas_object(pd.concat(cols, axis=1), index=False).values
    return cat.astype(np.int64)


@dataclass
class EVSidecar:
    action_vocab: List[str]
    x_cols: List[str]
    cont_cols: List[str]
    id_maps: Dict[str, Dict[str, int]]
    cont_expanded_cols: List[str]  # after expansion (e.g., board_mask_52 → bm0..bm51)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_vocab": list(self.action_vocab),
            "x_cols": list(self.x_cols),
            "cont_cols": list(self.cont_cols),
            "id_maps": {k: dict(v) for k, v in self.id_maps.items()},
            "cont_expanded_cols": list(self.cont_expanded_cols),
            "notes": self.notes or "",
        }


class EVParquetDataset(Dataset):
    """
    Generic EV dataset for all three EV models (preflop, postflop-root, postflop-facing).

    - Targets are EVs aligned to `action_vocab`, pulled from columns `ev_<TOKEN>`.
    - Categorical inputs (`x_cols`) are integer-encoded via `id_maps` (provided or auto-built).
    - Continuous inputs (`cont_cols`) are concatenated to a single float vector.
      Special handling:
        * "board_mask_52" is expanded from either a 52-d vector column or bm0..bm51 columns.

    __getitem__ returns a dict with:
      - "x_cat":  LongTensor [C]        (categorical features as ids)
      - "x_cont": FloatTensor [D]       (continuous features)
      - "y":      FloatTensor [V]       (EV vector; same order as action_vocab)
      - "w":      FloatTensor []        (scalar weight)
      - "meta":   dict                  (original values for optional debugging)
    """

    def __init__(
        self,
        *,
        parquet_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        action_vocab: Optional[Sequence[str]] = None,
        action_vocab_import: Optional[str] = None,  # e.g. "ml.models.vocab_actions:FACING_ACTION_VOCAB"
        x_cols: Sequence[str],
        cont_cols: Sequence[str],
        y_cols: Optional[Sequence[str]] = None,     # default → ["ev_<tok>" for tok in action_vocab]
        weight_col: Optional[str] = None,
        id_maps: Optional[Mapping[str, Mapping[str, int]]] = None,  # provide to fix encodings
        cache_arrays: bool = True,
    ):
        if (parquet_path is None) == (dataframe is None):
            raise ValueError("Provide exactly one of parquet_path or dataframe")
        self.df = pd.read_parquet(parquet_path) if dataframe is None else dataframe.copy()

        # Resolve action vocab
        if action_vocab is None and action_vocab_import:
            action_vocab = _import_symbol(action_vocab_import)
        if not action_vocab:
            raise ValueError("action_vocab is required (or action_vocab_import must resolve to it)")

        self.action_vocab: List[str] = _ensure_list_str(action_vocab)

        # Columns configuration
        self.x_cols: List[str] = _ensure_list_str(x_cols)
        self.cont_cols_raw: List[str] = _ensure_list_str(cont_cols)
        self.weight_col = str(weight_col) if weight_col else None

        # Target columns (default from vocab)
        if y_cols is None:
            self.y_cols = [f"ev_{tok}" for tok in self.action_vocab]
        else:
            self.y_cols = _ensure_list_str(y_cols)

        self._validate_presence()

        # Build/accept categorical id maps
        self.id_maps: Dict[str, Dict[str, int]] = {}
        if id_maps:
            for k, mp in id_maps.items():
                self.id_maps[str(k)] = {str(a): int(b) for a, b in mp.items()}
        for col in self.x_cols:
            if col not in self.id_maps:
                vals = sorted(self.df[col].astype(str).unique().tolist())
                self.id_maps[col] = {v: i for i, v in enumerate(vals)}

        # Expand continuous columns (handle board_mask_52)
        self.cont_expanded_cols: List[str] = []
        for c in self.cont_cols_raw:
            if c == "board_mask_52":
                keys = [f"bm{i}" for i in range(52)]
                if "board_mask_52" in self.df.columns:
                    # normalize to bm0..bm51 columns in-memory (explode once)
                    arr = self.df["board_mask_52"].apply(
                        lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [0.0] * 52
                    )
                    bm = np.stack(arr.values, axis=0).astype(np.float32) if len(arr) else np.zeros((0, 52), np.float32)
                    for i, k in enumerate(keys):
                        self.df[k] = bm[:, i]
                else:
                    # expect bm0..bm51 already present
                    missing = [k for k in keys if k not in self.df.columns]
                    if missing:
                        raise KeyError(f"missing board mask columns: {missing[:4]}... total missing={len(missing)}")
                self.cont_expanded_cols.extend(keys)
            else:
                self.cont_expanded_cols.append(c)

        # Cache to numpy arrays for speed (optional)
        self._cache_arrays = bool(cache_arrays)
        if self._cache_arrays:
            # categorical
            cat_arrays = []
            for col in self.x_cols:
                mp = self.id_maps[col]
                cat_arrays.append(self.df[col].astype(str).map(mp).astype(np.int64).values)
            self._X_cat = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(self.df), 0), dtype=np.int64)

            # continuous
            self._X_cont = self.df[self.cont_expanded_cols].astype(np.float32).values

            # targets
            self._Y = self.df[self.y_cols].astype(np.float32).values

            # weights
            if self.weight_col and self.weight_col in self.df.columns:
                self._W = self.df[self.weight_col].astype(np.float32).values
            else:
                self._W = np.ones((len(self.df),), dtype=np.float32)

        # store a light sidecar for consumers
        self._sidecar = EVSidecar(
            action_vocab=self.action_vocab,
            x_cols=self.x_cols,
            cont_cols=self.cont_cols_raw,
            id_maps=self.id_maps,
            cont_expanded_cols=self.cont_expanded_cols,
            notes="EVParquetDataset auto-generated schema",
        )

    # ----------------------- public API -----------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._cache_arrays:
            x_cat = torch.as_tensor(self._X_cat[idx], dtype=torch.long) if self._X_cat.shape[1] else torch.zeros(0, dtype=torch.long)
            x_cont = torch.as_tensor(self._X_cont[idx], dtype=torch.float32) if self._X_cont.shape[1] else torch.zeros(0, dtype=torch.float32)
            y = torch.as_tensor(self._Y[idx], dtype=torch.float32)
            w = torch.as_tensor(self._W[idx], dtype=torch.float32)
            meta = None
        else:
            row = self.df.iloc[idx]
            # categorical
            cat_ids: List[int] = []
            for col in self.x_cols:
                mp = self.id_maps[col]
                cat_ids.append(int(mp[str(row[col])]))
            x_cat = torch.tensor(cat_ids, dtype=torch.long) if cat_ids else torch.zeros(0, dtype=torch.long)

            # continuous
            cont_vals: List[float] = []
            # expand board mask on the fly if requested
            if "board_mask_52" in self.cont_cols_raw:
                bm = _board_mask_from_row(row)
                if bm is None:
                    raise KeyError("board_mask_52 requested but not available in row")
                cont_vals.extend([float(x) for x in bm])
                # add any other cont cols except the special mask
                for c in self.cont_cols_raw:
                    if c != "board_mask_52":
                        cont_vals.append(float(row[c]))
            else:
                for c in self.cont_cols_raw:
                    cont_vals.append(float(row[c]))
            x_cont = torch.tensor(cont_vals, dtype=torch.float32) if cont_vals else torch.zeros(0, dtype=torch.float32)

            # targets
            y_vals = [float(row[col]) for col in self.y_cols]
            y = torch.tensor(y_vals, dtype=torch.float32)

            # weight
            w_val = float(row[self.weight_col]) if (self.weight_col and self.weight_col in row) else 1.0
            w = torch.tensor(w_val, dtype=torch.float32)

            # meta (optional)
            meta = None

        return {"x_cat": x_cat, "x_cont": x_cont, "y": y, "w": w, "meta": meta}

    # ----------------------- helpers -----------------------

    @property
    def sidecar(self) -> EVSidecar:
        return self._sidecar

    @property
    def cat_cardinalities(self) -> List[int]:
        return [len(self.id_maps[c]) for c in self.x_cols]

    @property
    def cont_dim(self) -> int:
        # after expansion
        return len(self.cont_expanded_cols)

    @property
    def vocab_size(self) -> int:
        return len(self.action_vocab)

    def export_sidecar(self, path: Union[str, "os.PathLike"]) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self._sidecar.to_dict(), f, indent=2)

    def dataframe(self) -> pd.DataFrame:
        return self.df

    # ----------------------- class factories -----------------------

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        parquet_override: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> "EVParquetDataset":
        ds = cfg.get("dataset", {}) or {}
        paths = cfg.get("paths", {}) or {}
        model = cfg.get("model", {}) or {}

        parquet_path = parquet_override or paths.get("parquet_path") or (cfg.get("inputs", {}) or {}).get("parquet")
        if parquet_path is None and dataframe is None:
            raise ValueError("parquet path not provided and dataframe is None")

        action_vocab: Optional[Sequence[str]] = None
        action_vocab_import: Optional[str] = None

        # Allow either direct list via "model.action_vocab" or import string via "model.action_vocab_import"
        if "action_vocab" in model:
            action_vocab = model.get("action_vocab")
        elif "action_vocab_import" in model:
            action_vocab_import = model.get("action_vocab_import")

        x_cols = ds.get("x_cols") or []
        cont_cols = ds.get("cont_cols") or []
        y_cols = ds.get("y_cols") or None
        weight_col = ds.get("weight_col") or None

        return cls(
            parquet_path=parquet_path,
            dataframe=dataframe,
            action_vocab=action_vocab,
            action_vocab_import=action_vocab_import,
            x_cols=x_cols,
            cont_cols=cont_cols,
            y_cols=y_cols,
            weight_col=weight_col,
            id_maps=None,
            cache_arrays=True,
        )

    # ----------------------- splitting / collate -----------------------

    @staticmethod
    def split_train_val(
        df: pd.DataFrame,
        *,
        train_frac: float,
        seed: int = 42,
        stratify_keys: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Lightweight stratified split without sklearn."""
        n = len(df)
        if n == 0:
            return df.copy(), df.copy()
        rng = np.random.RandomState(int(seed))
        if stratify_keys:
            key = _hashable_strat_key(df, stratify_keys)
            # split within each key
            idx_train = []
            idx_val = []
            for k, grp in pd.Series(key).groupby(key):
                idx = grp.index.values
                m = len(idx)
                t = int(math.floor(m * float(train_frac)))
                perm = rng.permutation(idx)
                idx_train.append(perm[:t])
                idx_val.append(perm[t:])
            train_idx = np.concatenate(idx_train) if idx_train else np.array([], dtype=int)
            val_idx = np.concatenate(idx_val) if idx_val else np.array([], dtype=int)
        else:
            perm = rng.permutation(n)
            t = int(math.floor(n * float(train_frac)))
            train_idx = perm[:t]
            val_idx = perm[t:]
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {"x_cat": torch.empty(0, 0, dtype=torch.long),
                    "x_cont": torch.empty(0, 0, dtype=torch.float32),
                    "y": torch.empty(0, 0, dtype=torch.float32),
                    "w": torch.empty(0, dtype=torch.float32)}
        x_cat = torch.stack([b["x_cat"] for b in batch], dim=0) if batch[0]["x_cat"].numel() else torch.zeros(len(batch), 0, dtype=torch.long)
        x_cont = torch.stack([b["x_cont"] for b in batch], dim=0) if batch[0]["x_cont"].numel() else torch.zeros(len(batch), 0, dtype=torch.float32)
        y = torch.stack([b["y"] for b in batch], dim=0)
        w = torch.stack([b["w"] for b in batch], dim=0)
        return {"x_cat": x_cat, "x_cont": x_cont, "y": y, "w": w}

    # ----------------------- validators -----------------------

    def _validate_presence(self) -> None:
        missing_x = [c for c in self.x_cols if c not in self.df.columns]
        if missing_x:
            raise KeyError(f"missing x_cols in parquet: {missing_x}")
        missing_cont = [c for c in self.cont_cols_raw if (c != "board_mask_52" and c not in self.df.columns)]
        # board_mask_52 handled separately, so skip it here
        if missing_cont:
            raise KeyError(f"missing cont_cols in parquet: {missing_cont}")
        missing_y = [c for c in self.y_cols if c not in self.df.columns]
        if missing_y:
            raise KeyError(f"missing y_cols in parquet: {missing_y}")
        # optional weight col – don't error
        if self.weight_col and self.weight_col not in self.df.columns:
            # silently create a ones column to avoid crashing
            self.df[self.weight_col] = 1.0