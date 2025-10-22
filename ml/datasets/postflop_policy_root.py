# ml/datasets/postflop_policy_root.py

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ml.models.policy_consts import ACTION_VOCAB

BET_BUCKETS = {"BET_25","BET_33","BET_50","BET_66","BET_75","BET_100"}
DONK_BUCKETS = {"DONK_25","DONK_33","DONK_50","DONK_66","DONK_75","DONK_100"}

VOCAB_INDEX = {a:i for i,a in enumerate(ACTION_VOCAB)}
VOCAB_SIZE = len(ACTION_VOCAB)

class PostflopPolicyDatasetRoot(Dataset):
    """
    Root-only postflop policy dataset (IP first action on flop).
    - Targets are soft labels over ACTION_VOCAB, but only {CHECK, BET_size} are legal (DONK_size if actor=OOP).
    - Returns (x_cat, x_cont, y, m, w) per item:
        x_cat: Dict[str, LongTensor]
        x_cont: Dict[str, Float/Long Tensor] (incl. board_mask_52 as [52]-tensor)
        y: FloatTensor[V]   (soft label, normalized over legal actions)
        m: FloatTensor[V]   (1 for legal, 0 for illegal)
        w: FloatTensor[]    (scalar sample weight)
    """
    def __init__(
        self,
        parquet_path: str | Path,
        *,
        cat_features: List[str],         # from YAML (e.g., ["ctx","ip_pos","oop_pos"])
        cont_features: List[str],        # from YAML (e.g., ["pot_bb","effective_stack_bb"])
        weight_col: str = "weight",
        strict_canon: bool = True,
    ):
        self.parquet_path = Path(parquet_path)
        df = pd.read_parquet(str(self.parquet_path)).copy()

        # --- light canon for a few common categoricals ---
        def _canon_pos(v):
            if v is None: return None
            v = str(v).strip().upper()
            alias = {"UTG+1":"HJ","UTG+2":"CO","UTG1":"HJ","UTG2":"CO",
                     "DEALER":"BTN","BUTTON":"BTN","SMALL BLIND":"SB","BIG BLIND":"BB"}
            return alias.get(v, v)
        def _canon_ctx(v): return None if v is None else str(v).strip().upper()

        for col, fn in {"hero_pos": _canon_pos, "ip_pos": _canon_pos,
                        "oop_pos": _canon_pos, "ctx": _canon_ctx}.items():
            if col in df.columns:
                df[col] = df[col].map(fn)

        if "board_cluster" in df.columns and "board_cluster_id" not in df.columns:
            df = df.rename(columns={"board_cluster": "board_cluster_id"})

        # Root parquet should be IP rows only, but be robust:
        if "actor" in df.columns:
            df = df[df["actor"].astype(str).str.lower() == "ip"].reset_index(drop=True)

        self.cat_features  = list(cat_features or [])
        self.cont_features = list(cont_features or [])
        self.weight_col    = weight_col

        # Validate columns we rely on
        needed = set(self.cat_features + self.cont_features + [self.weight_col, "size_pct"]) \
                 | set(ACTION_VOCAB)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Root parquet missing columns: {missing}")

        # --- category id maps ---
        self._id_maps: dict[str, dict[str,int]] = {}
        self._cards: dict[str, int] = {}
        def _encode_cat(col: str):
            nonlocal df
            toks = df[col].fillna("__UNK__").astype(str).str.upper()
            uniq = sorted(toks.unique().tolist())
            tok2id = {t:i for i,t in enumerate(uniq)}
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

        # Stash df
        self._df = df.reset_index(drop=True)

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _to_mask_52(v) -> torch.Tensor:
        # Accept list/ndarray/JSON string/"comma,separated"
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().float().view(-1).numpy()
        else:
            if isinstance(v, str):
                vv = v.strip(); parsed = None
                if vv.startswith("[") and vv.endswith("]"):
                    try: parsed = json.loads(vv)
                    except Exception: parsed = None
                if parsed is None:
                    try: parsed = [float(x) for x in vv.split(",")]
                    except Exception: parsed = None
                v = parsed if parsed is not None else None
            try:
                arr = np.asarray(v, dtype=np.float32).reshape(-1) if v is not None else np.zeros(52, np.float32)
            except Exception:
                arr = np.zeros(52, np.float32)
        out = np.zeros(52, dtype=np.float32)
        n = min(52, arr.size)
        if n > 0: out[:n] = arr[:n]
        return torch.from_numpy(out)

    @staticmethod
    def _bet_token_from_size(size_pct: int, actor: str) -> str:
        size_pct = int(round(float(size_pct)))
        bucket = f"{size_pct}"
        if actor == "oop":
            tok = f"DONK_{bucket}"
            return tok if tok in DONK_BUCKETS else f"DONK_{bucket}"
        else:
            tok = f"BET_{bucket}"
            return tok if tok in BET_BUCKETS else f"BET_{bucket}"

    # ---- public ---------------------------------------------------------

    @property
    def id_maps(self) -> dict[str, dict[str,int]]:
        return self._id_maps

    @property
    def cards(self) -> dict[str,int]:
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

        # Continuous (scalar feats), plus board_mask_52 if present/requested
        x_cont: Dict[str, torch.Tensor] = {}
        for f in self.cont_features:
            if f == "board_mask_52":
                bm_t = self._to_mask_52(row.get("board_mask_52", None))
                x_cont["board_mask_52"] = bm_t
            else:
                v = row[f] if f in row else 0.0
                try: fv = float(v); fv = 0.0 if np.isnan(fv) else fv
                except Exception: fv = 0.0
                x_cont[f] = torch.tensor(fv, dtype=torch.float32).view(1)

        # Build legal mask for ROOT
        actor = "ip"  # root parts are IP by construction; keep guard for safety
        size_pct = int(round(float(row.get("size_pct", 33))))
        bet_tok  = self._bet_token_from_size(size_pct, actor="ip")

        legal = {"CHECK", bet_tok}
        # If you later add OOP-root variants, flip to DONK bucket:
        # if str(row.get("actor","ip")).lower() == "oop": legal = {"CHECK", self._bet_token_from_size(size_pct,"oop")}

        m = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        for a in legal:
            idx_a = VOCAB_INDEX.get(a)
            if idx_a is not None: m[idx_a] = 1.0

        # Targets: read soft columns and renormalize over legal only
        y_raw = np.array([float(row.get(a, 0.0) or 0.0) for a in ACTION_VOCAB], dtype=np.float32)
        y = torch.from_numpy(y_raw)
        y = y * m  # zero illegal
        s = float(y.sum().item())
        if s <= 1e-12:
            # Degenerate row; put mass on CHECK (common, safe)
            y.zero_()
            y[VOCAB_INDEX["CHECK"]] = 1.0
        else:
            y = y / s

        w = self.weights[idx]
        return x_cat, x_cont, y, m, w


def postflop_policy_root_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cat_list, x_cont_list, y_list, m_list, w_list = zip(*batch)

    x_cat: Dict[str, torch.Tensor] = {k: torch.stack([x[k] for x in x_cat_list], dim=0)
                                      for k in x_cat_list[0].keys()} if x_cat_list and x_cat_list[0] else {}

    # Stack continuous; ensure board_mask_52 stays [B,52]
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