import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ml.models.policy_consts import ACTION_VOCAB


VOCAB_INDEX = {a: i for i, a in enumerate(ACTION_VOCAB)}
VOCAB_SIZE = len(ACTION_VOCAB)

class PostflopPolicyDatasetFacing(Dataset):
    """
    Facing-only postflop policy dataset (OOP responding to an IP bet on flop).
    Returns (x_cat, x_cont, y, m, w):

      x_cat: Dict[str, LongTensor]      # categorical features (encoded)
      x_cont: Dict[str, Float/Long]     # continuous features (incl. board_mask_52 if supplied)
      y: FloatTensor[V]                 # soft targets (renormalized over legal)
      m: FloatTensor[V]                 # legal-action mask
      w: FloatTensor[]                  # sample weight

    Parquet requirements:
      - one soft column per action in ACTION_VOCAB
      - 'size_pct' = bet you are facing (25/33/50/66/75/100)
      - 'weight'
      - typical meta columns (ctx, positions, etc.) consumed via cat/cont configs
    """
    def __init__(
        self,
        parquet_path: str | Path,
        *,
        cat_features: List[str],
        cont_features: List[str],
        weight_col: str = "weight",
        strict_canon: bool = True,
        # Which raise buckets to allow as legal (tokens must exist in ACTION_VOCAB):
        allowed_raises: Optional[List[int]] = None,   # e.g. [150, 200, 300]
        allow_allin: bool = True,
    ):
        self.parquet_path = Path(parquet_path)
        df = pd.read_parquet(str(self.parquet_path)).copy()

        # --- canonicalize some categoricals (same style as root) ---
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

        # Facing parquet should be OOP rows; tolerate if mixed and filter:
        if "actor" in df.columns:
            df = df[df["actor"].astype(str).str.lower() == "oop"].reset_index(drop=True)

        self.cat_features  = list(cat_features or [])
        self.cont_features = list(cont_features or [])
        self.weight_col    = weight_col

        # Columns we must see
        needed = set(self.cat_features + self.cont_features + [self.weight_col, "size_pct"]) \
                 | set(ACTION_VOCAB)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Facing parquet missing columns: {missing}")

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

        # Weights
        w = df[self.weight_col].astype(float).to_numpy()
        w = np.where(np.isnan(w), 1.0, w)
        self.weights = torch.tensor(w, dtype=torch.float32)

        # Config for legality
        self.allowed_raises = list(allowed_raises or [150, 200, 300])
        self.allow_allin = bool(allow_allin)

        # Stash df
        self._df = df.reset_index(drop=True)

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _to_mask_52(v) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().float().view(-1).numpy()
        else:
            if isinstance(v, str):
                vv = v.strip(); parsed = None
                if vv.startswith("[") and vv.endswith("]"):
                    try:
                        parsed = json.loads(vv)
                    except Exception:
                        parsed = None
                if parsed is None:
                    try:
                        parsed = [float(x) for x in vv.split(",")]
                    except Exception:
                        parsed = None
                v = parsed if parsed is not None else None
            try:
                arr = np.asarray(v, dtype=np.float32).reshape(-1) if v is not None else np.zeros(52, np.float32)
            except Exception:
                arr = np.zeros(52, np.float32)
        out = np.zeros(52, dtype=np.float32)
        n = min(52, arr.size)
        if n > 0: out[:n] = arr[:n]
        return torch.from_numpy(out)

    def _legal_tokens(self) -> List[str]:
        toks = ["FOLD", "CALL"]
        for r in self.allowed_raises:
            tok = f"RAISE_{int(r)}"
            if tok in VOCAB_INDEX:
                toks.append(tok)
        if self.allow_allin and "ALLIN" in VOCAB_INDEX:
            toks.append("ALLIN")
        return toks

    # ---- public ---------------------------------------------------------

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

        # Continuous (+ board mask if present/requested)
        x_cont: Dict[str, torch.Tensor] = {}
        for f in self.cont_features:
            if f == "board_mask_52":
                bm_t = self._to_mask_52(row.get("board_mask_52", None))
                x_cont["board_mask_52"] = bm_t
            else:
                v = row[f] if f in row else 0.0
                try:
                    fv = float(v)
                    fv = 0.0 if np.isnan(fv) else fv
                except Exception:
                    fv = 0.0
                x_cont[f] = torch.tensor(fv, dtype=torch.float32).view(1)

        # Legal mask (facing)
        legal_list = self._legal_tokens()
        m = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        for tok in legal_list:
            i = VOCAB_INDEX.get(tok)
            if i is not None:
                m[i] = 1.0

        # Targets: soft columns → renorm over legal
        y_raw = np.array([float(row.get(a, 0.0) or 0.0) for a in ACTION_VOCAB], dtype=np.float32)
        y = torch.from_numpy(y_raw)
        y = y * m
        s = float(y.sum().item())
        if s <= 1e-12:
            # Degenerate: put mass on CALL (safest default when facing a bet)
            y.zero_()
            if "CALL" in VOCAB_INDEX:
                y[VOCAB_INDEX["CALL"]] = 1.0
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

    x_cat: Dict[str, torch.Tensor] = {k: torch.stack([x[k] for x in x_cat_list], dim=0)
                                      for k in x_cat_list[0].keys()} if x_cat_list and x_cat_list[0] else {}

    x_cont: Dict[str, torch.Tensor] = {}
    if x_cont_list and x_cont_list[0]:
        keys = list(x_cont_list[0].keys())
        for k in keys:
            vals = [x.get(k) for x in x_cont_list]
            if k == "board_mask_52":
                x_cont[k] = torch.stack([v.to(torch.float32).view(52) for v in vals], dim=0)
            else:
                x_cont[k] = torch.stack([v.to(torch.float32).view(1) for v in vals], dim=0)

    y = torch.stack(y_list, dim=0)  # [B,V]
    m = torch.stack(m_list, dim=0)  # [B,V]
    w = torch.stack(w_list, dim=0).float()  # [B]

    if y.shape[1] != len(ACTION_VOCAB):
        raise RuntimeError(f"Target width {y.shape[1]} != len(ACTION_VOCAB) {len(ACTION_VOCAB)}")

    return x_cat, x_cont, y, m, w