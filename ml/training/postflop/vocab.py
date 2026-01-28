# ml/training/postflop/vocab.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# IMPORTANT: do NOT redeclare action vocabs anywhere else.
# We only import these for optional validation / metadata.
from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB


UNK_TOKEN = "__UNK__"
UNK_ID = 0


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    # pandas / numpy NaN
    try:
        return bool(isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def _try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _stable_sort(values: List[str]) -> List[str]:
    """
    Deterministic ordering:
      - if all values are numeric-like -> numeric sort
      - else -> lexicographic sort
    """
    nums: List[Tuple[float, str]] = []
    for v in values:
        f = _try_float(v)
        if f is None:
            nums = []
            break
        nums.append((f, v))
    if nums and len(nums) == len(values):
        nums.sort(key=lambda t: (t[0], t[1]))
        return [v for _, v in nums]
    return sorted(values)


def _collect_unique_as_str(series: pd.Series) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in series.tolist():
        if _is_missing(x):
            continue
        s = str(x)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


@dataclass(frozen=True)
class ColumnVocab:
    """
    Maps stringified category -> id.
    ID 0 is reserved for UNK.
    """
    col: str
    token_to_id: Dict[str, int]

    def encode(self, value: Any) -> int:
        if _is_missing(value):
            return UNK_ID
        return int(self.token_to_id.get(str(value), UNK_ID))

    @property
    def size(self) -> int:
        return len(self.token_to_id)


def build_column_vocab(series: pd.Series, *, col: str, min_count: int = 1) -> ColumnVocab:
    """
    Build a stable vocab from a column.
    We stringify values to avoid parquet dtype quirks.
    """
    # counts
    counts: Dict[str, int] = {}
    for x in series.tolist():
        if _is_missing(x):
            continue
        s = str(x)
        counts[s] = counts.get(s, 0) + 1

    tokens = [t for t, c in counts.items() if c >= int(min_count)]
    tokens = _stable_sort(tokens)

    token_to_id: Dict[str, int] = {UNK_TOKEN: UNK_ID}
    next_id = 1
    for t in tokens:
        if t == UNK_TOKEN:
            continue
        token_to_id[t] = next_id
        next_id += 1

    return ColumnVocab(col=col, token_to_id=token_to_id)


def build_categorical_vocabs_from_parquets(
    parquet_paths: List[str],
    *,
    cat_cols: List[str],
    min_count: int = 1,
) -> Dict[str, ColumnVocab]:
    """
    Build vocabs for each column in cat_cols by scanning parquet files.
    """
    if not parquet_paths:
        raise ValueError("No parquet_paths provided")

    # aggregate counts per column across files without loading everything at once
    col_counts: Dict[str, Dict[str, int]] = {c: {} for c in cat_cols}

    for p in parquet_paths:
        df = pd.read_parquet(p, columns=cat_cols)
        for c in cat_cols:
            if c not in df.columns:
                raise KeyError(f"Missing column '{c}' in parquet: {p}")
            for tok in _collect_unique_as_str(df[c]):
                # count presence per row, not just unique presence
                # -> we actually want frequency; do per-value counts
                pass

        # do real counts (not unique) for correctness
        for c in cat_cols:
            s = df[c]
            for x in s.tolist():
                if _is_missing(x):
                    continue
                t = str(x)
                d = col_counts[c]
                d[t] = d.get(t, 0) + 1

    vocabs: Dict[str, ColumnVocab] = {}
    for c in cat_cols:
        # create a Series-like view from counts to reuse sorting logic
        counts = col_counts[c]
        tokens = [t for t, cnt in counts.items() if cnt >= int(min_count)]
        tokens = _stable_sort(tokens)

        token_to_id: Dict[str, int] = {UNK_TOKEN: UNK_ID}
        next_id = 1
        for t in tokens:
            if t == UNK_TOKEN:
                continue
            token_to_id[t] = next_id
            next_id += 1

        vocabs[c] = ColumnVocab(col=c, token_to_id=token_to_id)

    return vocabs


def save_vocabs_json(
    vocabs: Dict[str, ColumnVocab],
    *,
    out_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    out = {
        "unk_token": UNK_TOKEN,
        "unk_id": UNK_ID,
        "columns": {c: v.token_to_id for c, v in vocabs.items()},
        "meta": meta or {},
        # optional: record action vocabs used by training (imported, not redeclared)
        "action_vocabs": {
            "ROOT_ACTION_VOCAB": list(ROOT_ACTION_VOCAB),
            "FACING_ACTION_VOCAB": list(FACING_ACTION_VOCAB),
        },
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


def load_vocabs_json(path: str) -> Dict[str, ColumnVocab]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    cols = obj.get("columns") or {}
    out: Dict[str, ColumnVocab] = {}
    for c, token_to_id in cols.items():
        if not isinstance(token_to_id, dict):
            raise ValueError(f"Bad vocab for column {c}: expected dict")
        # ensure ints
        norm = {str(k): int(v) for k, v in token_to_id.items()}
        if UNK_TOKEN not in norm:
            norm[UNK_TOKEN] = UNK_ID
        out[c] = ColumnVocab(col=str(c), token_to_id=norm)
    return out


def list_parquets_under(*dirs: str) -> List[str]:
    paths: List[str] = []
    for d in dirs:
        dd = Path(d)
        if dd.is_file() and dd.suffix == ".parquet":
            paths.append(str(dd))
        elif dd.is_dir():
            paths.extend([str(p) for p in dd.glob("*.parquet")])
        else:
            # allow globs passed as-is
            for p in Path(".").glob(d):
                if p.is_file() and p.suffix == ".parquet":
                    paths.append(str(p))
    paths = sorted(set(paths))
    return paths