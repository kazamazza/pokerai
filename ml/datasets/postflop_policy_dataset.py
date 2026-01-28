from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset

from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB


Split = Literal["train", "val", "test"]
Kind = Literal["root", "facing"]


# ============================================================
# Vocab loader (categorical -> id)
# ============================================================
class ColumnVocab:
    def __init__(self, mapping: Dict[str, int]) -> None:
        # require __UNK__=0 for safety
        if "__UNK__" not in mapping:
            mapping = {"__UNK__": 0, **mapping}
        self.stoi = dict(mapping)
        self.unk = int(self.stoi.get("__UNK__", 0))

    def encode(self, x: Any) -> int:
        if x is None:
            return self.unk
        s = str(x)
        return int(self.stoi.get(s, self.unk))


class CatVocabs:
    """
    Expected JSON shape (example):
      {
        "columns": {
          "ctx": {"__UNK__":0, "VS_OPEN":1, ...},
          "ip_pos": {"__UNK__":0, "BTN":1, ...},
          ...
        }
      }
    """
    def __init__(self, vocabs: Dict[str, ColumnVocab]) -> None:
        self.vocabs = vocabs

    @classmethod
    def load(cls, path: str | Path) -> "CatVocabs":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        cols = obj.get("columns") or {}
        if not isinstance(cols, dict) or not cols:
            raise ValueError(f"Bad cat vocab json: {path}")
        out: Dict[str, ColumnVocab] = {}
        for col, mapping in cols.items():
            if isinstance(mapping, dict):
                # ensure int values
                out[col] = ColumnVocab({str(k): int(v) for k, v in mapping.items()})
        if not out:
            raise ValueError(f"No usable vocabs in: {path}")
        return cls(out)

    def encode(self, col: str, x: Any) -> int:
        v = self.vocabs.get(col)
        if v is None:
            # Missing vocab column: treat as unknown
            return 0
        return v.encode(x)


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class PostflopDatasetSpec:
    kind: Kind
    parts_dir: str                       # directory with parquet parts
    cat_vocabs_json: str                 # artifact built earlier
    x_cols: List[str]                    # feature cols (mix of cat + cont)
    cont_cols: List[str]                 # columns that are list[float] (e.g. board_mask_52)
    y_cols: List[str]                    # label probs columns
    weight_col: str = "weight"
    valid_col: str = "valid"

    # streaming / split control
    seed: int = 42
    shard_index: Optional[int] = None
    shard_count: Optional[int] = None


# ============================================================
# Helpers
# ============================================================
def _is_listlike(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))

def _to_float_tensor(x: Any) -> torch.Tensor:
    try:
        return torch.tensor(float(x), dtype=torch.float32)
    except Exception:
        return torch.tensor(0.0, dtype=torch.float32)

def _to_long_tensor(x: Any) -> torch.Tensor:
    try:
        return torch.tensor(int(x), dtype=torch.long)
    except Exception:
        return torch.tensor(0, dtype=torch.long)

def _to_float_vec(x: Any, *, expected_len: Optional[int] = None) -> torch.Tensor:
    if x is None:
        if expected_len is None:
            return torch.zeros(0, dtype=torch.float32)
        return torch.zeros(int(expected_len), dtype=torch.float32)

    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False).ravel()
    elif _is_listlike(x):
        arr = np.asarray(list(x), dtype=np.float32).ravel()
    else:
        # sometimes parquet stores as stringified json
        try:
            obj = json.loads(str(x))
            arr = np.asarray(obj, dtype=np.float32).ravel()
        except Exception:
            arr = np.zeros(0, dtype=np.float32)

    if expected_len is not None:
        n = int(expected_len)
        if arr.size != n:
            out = np.zeros(n, dtype=np.float32)
            m = min(n, int(arr.size))
            if m > 0:
                out[:m] = arr[:m]
            arr = out
    return torch.from_numpy(arr)


def _stable_shard(s: str, m: int) -> int:
    # same style as earlier sharding — stable hash
    import hashlib
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % m


# ============================================================
# Dataset
# ============================================================
class PostflopPolicyIterableDataset(IterableDataset):
    """
    Streams rows from parquet parts. Yields:
      x_cat: LongTensor [n_cat]      (categorical feature ids)
      x_num: FloatTensor [n_num]     (numeric scalar features)
      x_cont: FloatTensor [sum(cont)] (concatenated cont vectors, e.g. board_mask_52)
      y: FloatTensor [n_actions]
      w: FloatTensor []
    """
    def __init__(
        self,
        spec: PostflopDatasetSpec,
        *,
        split: Split = "train",
        split_fracs: Tuple[float, float, float] = (0.96, 0.02, 0.02),
        shuffle_files: bool = True,
        drop_invalid: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.split = split
        self.split_fracs = split_fracs
        self.shuffle_files = shuffle_files
        self.drop_invalid = drop_invalid

        self.vocabs = CatVocabs.load(spec.cat_vocabs_json)

        # Decide action vocab by dataset kind
        if spec.kind == "root":
            self.action_vocab = list(ROOT_ACTION_VOCAB)
        else:
            self.action_vocab = list(FACING_ACTION_VOCAB)

        # Pre-compute which x_cols are categorical vs numeric scalars.
        # Rule: if col exists in vocabs JSON => categorical id; else numeric scalar.
        self.cat_cols = [c for c in spec.x_cols if c in self.vocabs.vocabs]
        self.num_cols = [c for c in spec.x_cols if c not in self.vocabs.vocabs and c not in spec.cont_cols]

        # cont cols are vectors
        self.cont_cols = list(spec.cont_cols)

    def _iter_files(self) -> List[Path]:
        root = Path(self.spec.parts_dir)
        if not root.exists():
            raise FileNotFoundError(f"parts_dir not found: {root}")
        files = sorted(root.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet parts in: {root}")

        # deterministic file order per split/seed
        files2 = list(files)
        if self.shuffle_files:
            rng = random.Random(self.spec.seed + {"train": 1, "val": 2, "test": 3}[self.split])
            rng.shuffle(files2)

        return files2

    def _row_in_split(self, row_key: str) -> bool:
        a, b, c = self.split_fracs
        s = a + b + c
        if s <= 0:
            return True
        a, b, c = a / s, b / s, c / s

        u = _stable_shard(row_key, 10_000) / 10_000.0
        if self.split == "train":
            return u < a
        if self.split == "val":
            return a <= u < (a + b)
        return (a + b) <= u

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        files = self._iter_files()

        # Optional extra sharding across workers/machines
        if self.spec.shard_count is not None and self.spec.shard_index is not None:
            sc = int(self.spec.shard_count)
            si = int(self.spec.shard_index)
            keep = []
            for p in files:
                if _stable_shard(str(p), sc) == si:
                    keep.append(p)
            files = keep

        for p in files:
            dataset = ds.dataset(str(p), format="parquet")

            # Stream in record batches
            for batch in dataset.to_batches():
                cols = batch.to_pydict()
                n = len(next(iter(cols.values()))) if cols else 0

                for i in range(n):
                    # Filter invalid rows early if requested
                    valid = cols.get(self.spec.valid_col)
                    if valid is not None:
                        try:
                            is_valid = bool(valid[i])
                        except Exception:
                            is_valid = True
                        if self.drop_invalid and (not is_valid):
                            continue

                    # Stable split key (sha1 + s3_key is ideal)
                    sha1 = cols.get("sha1", [None])[i]
                    s3_key = cols.get("s3_key", [None])[i]
                    row_key = f"{sha1}|{s3_key}"
                    if not self._row_in_split(row_key):
                        continue

                    # ---- X categorical ids ----
                    x_cat_ids: List[int] = []
                    for c in self.cat_cols:
                        x_cat_ids.append(self.vocabs.encode(c, cols.get(c, [None])[i]))
                    x_cat = torch.tensor(x_cat_ids, dtype=torch.long)

                    # ---- X numeric scalars ----
                    x_nums: List[float] = []
                    for c in self.num_cols:
                        v = cols.get(c, [0.0])[i]
                        try:
                            x_nums.append(float(v))
                        except Exception:
                            x_nums.append(0.0)
                    x_num = torch.tensor(x_nums, dtype=torch.float32)

                    # ---- X cont vectors ----
                    cont_vecs: List[torch.Tensor] = []
                    for c in self.cont_cols:
                        v = cols.get(c, [None])[i]
                        # board_mask_52 is expected len 52; if you add others, adjust here
                        expected_len = 52 if c == "board_mask_52" else None
                        cont_vecs.append(_to_float_vec(v, expected_len=expected_len))
                    x_cont = torch.cat(cont_vecs, dim=0) if cont_vecs else torch.zeros(0, dtype=torch.float32)

                    # ---- Y (action probs) ----
                    y_vals: List[float] = []
                    for a in self.spec.y_cols:
                        v = cols.get(a, [0.0])[i]
                        try:
                            y_vals.append(float(v))
                        except Exception:
                            y_vals.append(0.0)
                    y = torch.tensor(y_vals, dtype=torch.float32)

                    # ---- weight ----
                    wv = cols.get(self.spec.weight_col, [1.0])[i]
                    w = _to_float_tensor(wv)

                    yield x_cat, x_num, x_cont, y, w


# ============================================================
# Collate
# ============================================================
def collate_postflop_policy(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Pads cont vectors if needed (currently fixed 52 so no padding required).
    """
    x_cat = torch.stack([b[0] for b in batch], dim=0)
    x_num = torch.stack([b[1] for b in batch], dim=0)
    x_cont = torch.stack([b[2] for b in batch], dim=0)
    y = torch.stack([b[3] for b in batch], dim=0)
    w = torch.stack([b[4] for b in batch], dim=0).view(-1)
    return {"x_cat": x_cat, "x_num": x_num, "x_cont": x_cont, "y": y, "w": w}