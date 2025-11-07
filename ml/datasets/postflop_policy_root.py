from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.models.vocab_actions import ROOT_ACTION_VOCAB  # ["CHECK","BET_25","BET_33",...,"DONK_..."] if you keep DONK in root vocab; otherwise only CHECK/BET_XX

# ---- Root-only vocab (what the model head will use) ----
ROOT_VOCAB  = list(ROOT_ACTION_VOCAB)
ROOT_INDEX  = {a: i for i, a in enumerate(ROOT_VOCAB)}
V_ROOT      = len(ROOT_VOCAB)

# Buckets you expect for IP root bets
BET_BUCKETS = {"BET_25","BET_33","BET_50","BET_66","BET_75","BET_100"}

# Any facing tokens that might have leaked in
FACING_PREFIXES = ("FOLD", "CALL", "RAISE_", "ALLIN")


class PostflopPolicyDatasetRoot(Dataset):
    """
    ROOT-only (IP first action). Returns (x_cat, x_cont, y, m, w).

    - y and m are sized to ROOT_VOCAB.
    - Legal mask m contains {CHECK, BET_{size_pct}} — always IP root (no actor filtering).
    - size_pct/size_frac are used only to decide the legal BET bucket (not added to features).
    """
    def __init__(
        self,
        parquet_path: str | Path,
        *,
        cat_features: List[str],
        cont_features: List[str],
        weight_col: str = "weight",
        strict_canon: bool = True,
    ):
        self.parquet_path = Path(parquet_path)
        df = pd.read_parquet(str(self.parquet_path)).copy()

        # --- light canonicalization ---
        def _canon_pos(v):
            if v is None: return None
            v = str(v).strip().upper()
            alias = {
                "UTG+1":"HJ","UTG+2":"CO",
                "UTG1":"HJ","UTG2":"CO",
                "DEALER":"BTN","BUTTON":"BTN",
                "SMALL BLIND":"SB","BIG BLIND":"BB",
            }
            return alias.get(v, v)

        def _canon_ctx(v):
            return None if v is None else str(v).strip().upper()

        for col, fn in {"hero_pos": _canon_pos, "ip_pos": _canon_pos,
                        "oop_pos": _canon_pos, "ctx": _canon_ctx}.items():
            if col in df.columns:
                df[col] = df[col].map(fn)

        if "board_cluster" in df.columns and "board_cluster_id" not in df.columns:
            df = df.rename(columns={"board_cluster": "board_cluster_id"})

        # --- derive size_frac / size_pct for legality only ---
        if "size_frac" not in df.columns:
            if "size_pct" in df.columns:
                df["size_frac"] = pd.to_numeric(df["size_pct"], errors="coerce") / 100.0
            elif "bet_size_frac" in df.columns:
                df["size_frac"] = pd.to_numeric(df["bet_size_frac"], errors="coerce")
            else:
                df["size_frac"] = np.nan

        df["size_frac"] = (
            pd.to_numeric(df["size_frac"], errors="coerce")
              .astype("float32")
              .fillna(0.0)
              .clip(lower=0.0, upper=1.5)
        )

        if "size_pct" not in df.columns:
            df["size_pct"] = (df["size_frac"] * 100.0).round().astype("Int64")

        # --- drop any facing labels that leaked in ---
        drop_cols = [c for c in df.columns if any(c.startswith(p) for p in FACING_PREFIXES)]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        # --- guarantee root target columns exist (exactly ROOT_VOCAB) ---
        for tok in ROOT_VOCAB:
            if tok not in df.columns:
                df[tok] = 0.0

        # --- store feature config ---
        self.cat_features  = list(cat_features or [])
        self.cont_features = list(cont_features or [])  # do NOT auto-append size_frac for ROOT
        self.weight_col    = weight_col

        # ---- column presence checks ----
        needed_meta = set(self.cat_features + self.cont_features + [self.weight_col, "size_pct"])
        missing_meta = [c for c in needed_meta if c not in df.columns]
        if missing_meta:
            raise ValueError(f"Root parquet missing columns: {missing_meta}")

        # --- encode categoricals ---
        self._id_maps: Dict[str, Dict[str, int]] = {}
        self._cards: Dict[str, int] = {}

        def _encode_cat(col: str):
            nonlocal df
            toks = df[col].fillna("__UNK__").astype(str).str.upper()
            uniq = sorted(toks.unique().tolist())
            tok2id = {t: i for i, t in enumerate(uniq)}
            self._id_maps[col] = tok2id
            self._cards[col]   = max(1, len(tok2id))
            df[col] = toks.map(tok2id)

        for c in self.cat_features:
            if c in df.columns:
                _encode_cat(c)

        if strict_canon:
            for c in self.cat_features:
                if df[c].isna().any():
                    raise ValueError(f"Invalid values in '{c}' (NaNs after encoding)")

        # --- weights ---
        w = pd.to_numeric(df[self.weight_col], errors="coerce").fillna(1.0).astype(float).to_numpy()
        self.weights = torch.tensor(w, dtype=torch.float32)

        self._df = df.reset_index(drop=True)

    # ---------- helpers ----------

    @staticmethod
    def _to_mask_52(v) -> torch.Tensor:
        """
        Robustly parse board_mask_52 from:
          - list/ndarray of floats,
          - Avro/pyarrow list of {element: float} or StructScalar,
          - JSON string "[...]" or "a,b,c,...".
        """
        import numpy as np, json
        # pyarrow scalar wrapper?
        if hasattr(v, "as_py"):
            try:
                v = v.as_py()
            except Exception:
                v = None

        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    v = json.loads(s)
                except Exception:
                    v = None
            else:
                # "0,1,0,..."
                try:
                    v = [float(x) for x in s.split(",")]
                except Exception:
                    v = None

        if isinstance(v, (list, tuple)):
            out = []
            for it in v:
                if hasattr(it, "as_py"):
                    try:
                        it = it.as_py()
                    except Exception:
                        pass
                if isinstance(it, dict) and "element" in it:
                    out.append(it["element"])
                else:
                    out.append(it)
            v = out

        try:
            arr = np.asarray(v, dtype=np.float32).reshape(-1) if v is not None else np.zeros(52, np.float32)
        except Exception:
            arr = np.zeros(52, np.float32)

        out = np.zeros(52, dtype=np.float32)
        n = min(52, arr.size)
        if n > 0:
            out[:n] = arr[:n]
        return torch.from_numpy(out)

    @staticmethod
    def _nearest_bet_token(size_pct: int) -> str:
        """
        Map arbitrary integer size to nearest available BET_XX token present in ROOT_VOCAB.
        """
        # collect available BET_XX from ROOT_VOCAB
        options = []
        for tok in ROOT_VOCAB:
            if tok.startswith("BET_"):
                try:
                    options.append(int(tok.split("_", 1)[1]))
                except Exception:
                    pass
        if not options:
            return "CHECK"  # degenerate (shouldn't happen)

        s = int(round(float(size_pct)))
        nearest = min(options, key=lambda x: abs(x - s))
        return f"BET_{nearest}"

    # ---------- API ----------

    @property
    def id_maps(self) -> Dict[str, Dict[str, int]]:
        return self._id_maps

    @property
    def cards(self) -> Dict[str, int]:
        return dict(self._cards)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]

        # categoricals
        x_cat: Dict[str, torch.Tensor] = {}
        for f in self.cat_features:
            v = int(row[f]) if f in row and pd.notna(row[f]) else 0
            x_cat[f] = torch.tensor(v, dtype=torch.long)

        # continuous (+board mask)
        x_cont: Dict[str, torch.Tensor] = {}
        for f in self.cont_features:
            if f == "board_mask_52":
                x_cont["board_mask_52"] = self._to_mask_52(row.get("board_mask_52", None))
            else:
                try:
                    fv = float(row[f]) if f in row else 0.0
                    if np.isnan(fv): fv = 0.0
                except Exception:
                    fv = 0.0
                x_cont[f] = torch.tensor(fv, dtype=torch.float32).view(1)

        # legal mask: always IP root → {CHECK, BET_bucket}
        # legal mask: always IP root → {CHECK, BET_bucket}
        val = row.get("size_pct", 33)
        try:
            size_pct = float(pd.to_numeric(val, errors="coerce"))
            if np.isnan(size_pct):
                size_pct = 33
        except Exception:
            size_pct = 33
        size_pct = int(round(size_pct))
        bet_tok = self._nearest_bet_token(size_pct)

        legal = {"CHECK", bet_tok}
        m = torch.zeros(V_ROOT, dtype=torch.float32)
        for a in legal:
            i = ROOT_INDEX.get(a)
            if i is not None:
                m[i] = 1.0

        # targets over ROOT_VOCAB, renorm within legal
        y_raw = np.array([float(row.get(a, 0.0) or 0.0) for a in ROOT_VOCAB], dtype=np.float32)
        y = torch.from_numpy(y_raw) * m
        s = float(y.sum().item())
        if s <= 1e-12:
            y.zero_()
            y[ROOT_INDEX["CHECK"]] = 1.0
        else:
            y = y / s

        w = self.weights[idx]
        return x_cat, x_cont, y, m, w


def postflop_policy_root_collate_fn(
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
                                      for k in (x_cat_list[0].keys() if x_cat_list and x_cat_list[0] else [])}

    x_cont: Dict[str, torch.Tensor] = {}
    if x_cont_list and x_cont_list[0]:
        for k in x_cont_list[0].keys():
            vals = [x.get(k) for x in x_cont_list]
            if k == "board_mask_52":
                x_cont[k] = torch.stack([v.to(torch.float32).view(52) for v in vals], dim=0)
            else:
                x_cont[k] = torch.stack([v.to(torch.float32).view(1) for v in vals], dim=0)

    y = torch.stack(y_list, dim=0)  # [B, V_ROOT]
    m = torch.stack(m_list, dim=0)  # [B, V_ROOT]
    w = torch.stack(w_list, dim=0).float()  # [B]

    if y.shape[1] != len(ROOT_VOCAB):
        raise RuntimeError(f"Target width {y.shape[1]} != len(ROOT_VOCAB) {len(ROOT_VOCAB)}")

    return x_cat, x_cont, y, m, w