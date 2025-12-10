from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import importlib
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.utils.ev import _board_mask_from_row


def _ensure_list_str(xs: Sequence[Any]) -> List[str]:
    return [str(x) for x in (xs or [])]


def _import_symbol(path: str):
    mod, sym = path.split(":")
    import importlib
    return getattr(importlib.import_module(mod), sym)


def _hashable_strat_key(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    # WHY: make stable, compact keys for group-wise splitting
    return df[keys].astype(str).agg("|".join, axis=1)
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _norm_id_maps(raw: Optional[Mapping[str, Mapping[str, int]]]) -> Dict[str, Dict[str, int]]:
    """why: freeze categorical encodings; ensure str→int."""
    out: Dict[str, Dict[str, int]] = {}
    for k, mp in (raw or {}).items():
        out[str(k)] = {str(a): int(b) for a, b in (mp or {}).items()}
    return out


@dataclass
class EVSidecar:
    # Required
    action_vocab: List[str]
    x_cols: List[str]
    cont_cols: List[str]
    id_maps: Dict[str, Dict[str, int]]

    # Optional/derived
    schema_version: str = "ev_sidecar_v1"
    model_name: str = "EVNet"
    units: str = "bb"  # fixed expectation for EV scale
    cont_expanded_cols: List[str] = field(default_factory=list)
    notes: str = ""
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    checkpoint_file: Optional[str] = None

    # ---- Derived helpers ----
    @property
    def vocab_index(self) -> Dict[str, int]:
        return {tok: i for i, tok in enumerate(self.action_vocab)}

    @property
    def cat_cardinalities(self) -> Dict[str, int]:
        return {c: len(self.id_maps.get(c, {})) for c in self.x_cols}

    # ---- Serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "action_vocab": list(self.action_vocab),
            "vocab_index": self.vocab_index,
            # canonical keys
            "x_cols": list(self.x_cols),
            "cont_cols": list(self.cont_cols),
            "id_maps": {k: dict(v) for k, v in self.id_maps.items()},
            "cat_cardinalities": self.cat_cardinalities,
            "cont_expanded_cols": list(self.cont_expanded_cols),
            "units": self.units,
            "notes": self.notes,
            "created_utc": self.created_utc,
            "checkpoint_file": self.checkpoint_file,
            # legacy aliases (for older readers)
            "cat_feature_order": list(self.x_cols),
            "cont_feature_order": list(self.cont_cols),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    def save(self, path: str | Path, duplicate_stem_copy: bool = False) -> str:
        """why: persist alongside checkpoints and optionally duplicate with _sidecar suffix."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        if p.suffix == "":
            # treat as directory; write standard filename
            p = p / "best_sidecar.json"
        p.write_text(payload)
        if duplicate_stem_copy and p.suffix:
            stem_copy = p.with_name(f"{p.stem}_sidecar.json")
            stem_copy.write_text(payload)
        return str(p)

    # ---- Deserialization ----
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EVSidecar":
        # accept legacy keys
        action_vocab = list(d.get("action_vocab") or [])
        x_cols = list(d.get("x_cols") or d.get("cat_feature_order") or [])
        cont_cols = list(d.get("cont_cols") or d.get("cont_feature_order") or [])
        id_maps = _norm_id_maps(d.get("id_maps"))

        if not action_vocab or not x_cols or not cont_cols:
            raise ValueError("EVSidecar requires action_vocab, x_cols/cont_cols")

        return cls(
            action_vocab=action_vocab,
            x_cols=x_cols,
            cont_cols=cont_cols,
            id_maps=id_maps,
            schema_version=str(d.get("schema_version") or "ev_sidecar_v1"),
            model_name=str(d.get("model_name") or "EVNet"),
            units=str(d.get("units") or "bb"),
            cont_expanded_cols=list(d.get("cont_expanded_cols") or []),
            notes=str(d.get("notes") or ""),
            created_utc=str(d.get("created_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            checkpoint_file=d.get("checkpoint_file"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EVSidecar":
        p = Path(path)
        data = json.loads(p.read_text())
        return cls.from_dict(data)

    # ---- Convenience from dataset ----
    @classmethod
    def from_dataset(
        cls,
        *,
        action_vocab: List[str],
        x_cols: List[str],
        cont_cols: List[str],
        id_maps: Mapping[str, Mapping[str, int]],
        cont_expanded_cols: Optional[List[str]] = None,
        model_name: str = "EVNet",
        checkpoint_file: Optional[str] = None,
        notes: str = "",
    ) -> "EVSidecar":
        return cls(
            action_vocab=list(action_vocab),
            x_cols=list(x_cols),
            cont_cols=list(cont_cols),
            id_maps=_norm_id_maps(id_maps),
            model_name=model_name,
            checkpoint_file=checkpoint_file,
            notes=notes,
            cont_expanded_cols=list(cont_expanded_cols or []),
        )

class EVParquetDataset(Dataset):
    """
    Minimal EV dataset:
      - Targets: ['ev_<tok>' for tok in action_vocab] (units: bb).
      - Optional 'board_mask_52' cont col expanded to bm0..bm51.
      - Optional illegal mask column: 'illegal_mask_root'/'illegal_mask_facing' → y_mask (1=learn,0=ignore).
    """
    def __init__(
        self,
        *,
        parquet_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        action_vocab: Optional[Sequence[str]] = None,
        action_vocab_import: Optional[str] = None,
        x_cols: Sequence[str],
        cont_cols: Sequence[str],
        y_cols: Optional[Sequence[str]] = None,
        weight_col: Optional[str] = None,
        id_maps: Optional[Mapping[str, Mapping[str, int]]] = None,  # <-- restored
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

        # Target columns (default from vocab order)
        self.y_cols = _ensure_list_str(y_cols) if y_cols is not None else [f"ev_{tok}" for tok in self.action_vocab]

        self._validate_presence()

        # ---- categorical encodings (freeze if provided) ----
        self.id_maps: Dict[str, Dict[str, int]] = {}
        if id_maps:
            # use provided maps exactly
            for k, mp in id_maps.items():
                self.id_maps[str(k)] = {str(a): int(b) for a, b in (mp or {}).items()}

        for col in self.x_cols:
            if col not in self.id_maps:
                vals = sorted(self.df[col].astype(str).unique().tolist())
                self.id_maps[col] = {v: i for i, v in enumerate(vals)}

        # ---- expand continuous columns (board_mask_52) ----
        self.cont_expanded_cols: List[str] = []
        for c in self.cont_cols_raw:
            if c == "board_mask_52":
                keys = [f"bm{i}" for i in range(52)]
                if "board_mask_52" in self.df.columns:
                    arr = self.df["board_mask_52"].apply(
                        lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [0.0] * 52
                    )
                    bm = np.stack(arr.values, axis=0).astype(np.float32) if len(arr) else np.zeros((0, 52), np.float32)
                    for i, k in enumerate(keys):
                        self.df[k] = bm[:, i]
                else:
                    missing = [k for k in keys if k not in self.df.columns]
                    if missing:
                        raise KeyError(f"missing board mask columns: {missing[:4]}... total missing={len(missing)}")
                self.cont_expanded_cols.extend(keys)
            else:
                self.cont_expanded_cols.append(c)

        # Optional y_mask (illegal tokens)
        self._has_illegal_mask = False
        if "illegal_mask_root" in self.df.columns or "illegal_mask_facing" in self.df.columns:
            self._has_illegal_mask = True
            self._illegal_col = "illegal_mask_root" if "illegal_mask_root" in self.df.columns else "illegal_mask_facing"

        # ---- cache arrays ----
        self._cache_arrays = bool(cache_arrays)
        if self._cache_arrays:
            # categorical → ids via frozen maps; unseen values map to 0
            cat_arrays = []
            for col in self.x_cols:
                mp = self.id_maps[col]
                cat_arrays.append(self.df[col].astype(str).map(mp).fillna(0).astype(np.int64).values)
            self._X_cat = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(self.df), 0), dtype=np.int64)

            # continuous
            self._X_cont = self.df[self.cont_expanded_cols].astype(np.float32).values

            # targets
            self._Y = self.df[self.y_cols].astype(np.float32).values

            # sample weights
            if self.weight_col and self.weight_col in self.df.columns:
                self._W = self.df[self.weight_col].astype(np.float32).values
            else:
                self._W = np.ones((len(self.df),), dtype=np.float32)

            # per-token mask (1=learn, 0=ignore)
            if self._has_illegal_mask:
                mask_list = self.df[self._illegal_col].tolist()
                self._M = np.array([[0 if m else 1 for m in row] for row in mask_list], dtype=np.float32)  # invert
            else:
                self._M = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache_arrays:
            x_cat = torch.as_tensor(self._X_cat[idx], dtype=torch.long) if self._X_cat.shape[1] else torch.zeros(0, dtype=torch.long)
            x_cont = torch.as_tensor(self._X_cont[idx], dtype=torch.float32) if self._X_cont.shape[1] else torch.zeros(0, dtype=torch.float32)
            y = torch.as_tensor(self._Y[idx], dtype=torch.float32)
            w = torch.as_tensor(self._W[idx], dtype=torch.float32)
            out = {"x_cat": x_cat, "x_cont": x_cont, "y": y, "w": w}
            if getattr(self, "_M", None) is not None:
                out["y_mask"] = torch.as_tensor(self._M[idx], dtype=torch.float32)
            return out

        # non-cached path (rare)
        row = self.df.iloc[idx]
        # cats
        ids = []
        for col in self.x_cols:
            mp = self.id_maps[col]
            ids.append(int(mp.get(str(row[col]), 0)))
        x_cat = torch.tensor(ids, dtype=torch.long) if ids else torch.zeros(0, dtype=torch.long)
        # cont
        cont_vals: List[float] = []
        if "board_mask_52" in self.cont_cols_raw:
            bm = _board_mask_from_row(row)
            if bm is None:
                raise KeyError("board_mask_52 requested but not available in row")
            cont_vals.extend([float(x) for x in bm])
            for c in self.cont_cols_raw:
                if c != "board_mask_52":
                    cont_vals.append(float(row[c]))
        else:
            for c in self.cont_cols_raw:
                cont_vals.append(float(row[c]))
        x_cont = torch.tensor(cont_vals, dtype=torch.float32) if cont_vals else torch.zeros(0, dtype=torch.float32)
        # targets/weights
        y = torch.tensor([float(row[c]) for c in self.y_cols], dtype=torch.float32)
        w = torch.tensor(float(row[self.weight_col]) if (self.weight_col and self.weight_col in row) else 1.0, dtype=torch.float32)
        out = {"x_cat": x_cat, "x_cont": x_cont, "y": y, "w": w}
        if self._has_illegal_mask:
            mask = row[self._illegal_col]
            m = torch.tensor([0.0 if v else 1.0 for v in mask], dtype=torch.float32)
            out["y_mask"] = m
        return out

    # expose maps for splits/sidecar
    @property
    def sidecar(self) -> EVSidecar:
        return EVSidecar(
            action_vocab=self.action_vocab,
            x_cols=self.x_cols,
            cont_cols=self.cont_cols_raw,
            id_maps=self.id_maps,
            cont_expanded_cols=self.cont_expanded_cols,
            notes="EVParquetDataset auto-generated schema",
        )

def ev_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {"x_cat": torch.empty(0,0,dtype=torch.long), "x_cont": torch.empty(0,0,dtype=torch.float32),
                "y": torch.empty(0,0,dtype=torch.float32), "w": torch.empty(0,dtype=torch.float32)}
    x_cat  = torch.stack([b["x_cat"]  for b in batch], dim=0) if batch[0]["x_cat"].numel() else torch.zeros(len(batch), 0, dtype=torch.long)
    x_cont = torch.stack([b["x_cont"] for b in batch], dim=0) if batch[0]["x_cont"].numel() else torch.zeros(len(batch), 0, dtype=torch.float32)
    y      = torch.stack([b["y"]      for b in batch], dim=0)
    w      = torch.stack([b["w"]      for b in batch], dim=0)
    out = {"x_cat": x_cat, "x_cont": x_cont, "y": y, "w": w}
    if "y_mask" in batch[0]:
        out["y_mask"] = torch.stack([b["y_mask"] for b in batch], dim=0)
    return out