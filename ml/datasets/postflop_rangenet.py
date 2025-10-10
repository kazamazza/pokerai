import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

from ml.models.policy_consts import VOCAB_SIZE, VOCAB_INDEX, ACTION_VOCAB

BET_BUCKETS   = {"BET_25","BET_33","BET_50","BET_66","BET_75","BET_100","DONK_33"}
RAISE_BUCKETS = {"RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"}

class PostflopPolicyDatasetParquet(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        *,
        cat_features: list[str],            # from YAML
        cont_features: list[str],           # from YAML
        soft_y_cols: list[str] | None,      # from YAML (None => use hard_y_col)
        hard_y_col: str | None = None,      # required if soft_y_cols is None
        weight_col: str = "weight",
        device: Optional[torch.device] = None,
        strict_canon: bool = True,
        force_hard: bool = False,
    ):
        self.parquet_path = Path(parquet_path)
        df = pd.read_parquet(str(self.parquet_path)).copy()
        self.device = device
        self.force_hard = bool(force_hard)

        # ---------- light canon for a few common categoricals ----------
        def canon_pos(v: str | None) -> Optional[str]:
            if v is None: return None
            v = str(v).strip().upper()
            aliases = {
                "UTG+1":"HJ","UTG+2":"CO","UTG1":"HJ","UTG2":"CO",
                "DEALER":"BTN","BUTTON":"BTN","SMALL BLIND":"SB","BIG BLIND":"BB",
            }
            return aliases.get(v, v)

        def canon_ctx(v: str | None) -> Optional[str]:
            return None if v is None else str(v).strip().upper()

        def canon_street(v) -> Optional[str]:
            if v is None: return None
            if isinstance(v, int):
                return {1:"FLOP", 2:"TURN", 3:"RIVER"}.get(v, str(v))
            m = {"1":"FLOP","2":"TURN","3":"RIVER"}
            s = str(v).strip().upper()
            return m.get(s, s)

        for col, fn in {
            "hero_pos": canon_pos,
            "ip_pos": canon_pos,
            "oop_pos": canon_pos,
            "ctx": canon_ctx,
            "street": canon_street,
        }.items():
            if col in df.columns:
                df[col] = df[col].map(fn)

        # Normalize possible board cluster column variants
        if "board_cluster" in df.columns and "board_cluster_id" not in df.columns:
            df = df.rename(columns={"board_cluster": "board_cluster_id"})

        # ---------- schema (from YAML) ----------
        self.cat_features  = list(cat_features or [])
        self.cont_features = list(cont_features or [])
        self.soft_y_cols   = list(soft_y_cols or [])
        self.hard_y_col    = hard_y_col
        self.weight_col    = weight_col

        # ---------- validate presence ----------
        needed = set(self.cat_features + self.cont_features + [self.weight_col])
        if (not self.force_hard) and self.soft_y_cols:
            needed |= set(self.soft_y_cols)
        else:
            if not self.hard_y_col:
                raise ValueError("hard_y_col is required when soft_y_cols is None or force_hard=True")
            needed.add(self.hard_y_col)

        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing columns: {missing}")

        # ---------- build id maps & cardinalities for categoricals ----------
        self._id_maps: dict[str, dict[str, int]] = {}
        self._cards: dict[str, int] = {}

        def encode_cat(col: str):
            nonlocal df
            if pd.api.types.is_integer_dtype(df[col]):
                # identity map
                uniq = sorted(pd.Series(df[col]).dropna().unique().tolist())
                self.id_maps[col] = {str(int(v)): int(v) for v in uniq}
                self.cards[col]   = (max(uniq) + 1) if uniq else 1
            else:
                toks = df[col].fillna("__UNK__").astype(str).str.upper()
                uniq = sorted(toks.unique().tolist())
                tok2id = {t:i for i,t in enumerate(uniq)}
                self.id_maps[col] = tok2id
                self.cards[col]   = max(1, len(tok2id))
                df[col] = toks.map(tok2id)

        for c in self.cat_features:
            encode_cat(c)

        if strict_canon:
            for c in self.cat_features:
                if c in df.columns and df[c].isna().any():
                    raise ValueError(f"Invalid/uncanonical values in '{c}' (NaNs after encoding)")

        # ---------- weights ----------
        if self.weight_col in df.columns:
            w = df[self.weight_col].astype(float).to_numpy()
            w = np.where(np.isnan(w), 1.0, w)
            self.weights = torch.tensor(w, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(df), dtype=torch.float32)

        # ---------- labels (soft vs hard) ----------
        self.has_soft = (not self.force_hard) and bool(self.soft_y_cols)
        if self.has_soft:
            Y = df[self.soft_y_cols].to_numpy(dtype="float32")
            row_sum = Y.sum(axis=1, keepdims=True).clip(min=1e-8)
            self.y_soft = torch.tensor((Y / row_sum), dtype=torch.float32)
            self.action_idx = None
        else:
            # Hard labels: support either integer class ids or action tokens.
            y_raw = df[self.hard_y_col]
            if pd.api.types.is_integer_dtype(y_raw):
                self.action_idx = torch.tensor(y_raw.to_numpy(dtype="int64"), dtype=torch.long)
            else:
                # Map via global VOCAB_INDEX if available
                try:
                    idx = y_raw.astype(str).str.upper().map(lambda a: VOCAB_INDEX[a])  # noqa: F821
                except Exception:
                    raise ValueError(
                        "Hard labels are non-integer and VOCAB_INDEX is not available in scope."
                    )
                if idx.isna().any():
                    bad = sorted(y_raw[idx.isna()].astype(str).str.upper().unique().tolist())
                    raise ValueError(f"Unmapped hard-label actions: {bad}")
                self.action_idx = torch.tensor(idx.to_numpy(dtype="int64"), dtype=torch.long)
            self.y_soft = None

        # ---------- stash tensors/arrays ----------
        self._df = df.reset_index(drop=True)
        self._X_cat  = df[self.cat_features].to_numpy(dtype="int64")   if self.cat_features else None
        self._X_cont = df[self.cont_features].to_numpy(dtype="float32") if self.cont_features else None

        # for sidecar/export convenience
        self.feature_order = list(self.cat_features)
        self._has_cluster_id = "board_cluster_id" in df.columns
        self._has_board_mask = "board_mask_52" in df.columns

    @property
    def id_maps(self) -> dict[str, dict[str, int]]:
        """Return mapping of categorical features to token→id maps."""
        return self._id_maps

    @property
    def cards(self) -> dict[str, int]:
        """Return categorical vocab sizes per feature."""
        sizes: dict[str, int] = {}
        for c in self.cat_features:
            if c in self._id_maps:
                sizes[c] = len(self._id_maps[c])
            else:
                # fallback if feature wasn't mapped explicitly
                try:
                    sizes[c] = int(self._df[c].max()) + 1
                except Exception:
                    sizes[c] = 1
        return sizes

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]

        # --- CATEGORICALS (incl. board_cluster_id if present) ---
        x_cat: Dict[str, torch.Tensor] = {}
        for f in self.cat_features:
            v = row[f] if f in row else 0
            # robust cast (NaN/None/str)
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    iv = 0
                else:
                    iv = int(v)
            except Exception:
                iv = 0
            x_cat[f] = torch.tensor(iv, dtype=torch.long)

        # --- CONTINUOUS SCALARS (exclude the 52-d mask here) ---
        scalar_cont_feats = [f for f in self.cont_features if f != "board_mask_52"]
        x_cont: Dict[str, torch.Tensor] = {}
        for f in scalar_cont_feats:
            v = row[f] if f in row else 0.0
            try:
                fv = float(v)
                if np.isnan(fv):
                    fv = 0.0
            except Exception:
                fv = 0.0
            x_cont[f] = torch.tensor(fv, dtype=torch.float32).view(1)

        # --- BOARD MASK (52-d vector) ---
        def _to_mask_52(v) -> torch.Tensor:
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().float().view(-1).numpy()
            else:
                # allow list/tuple/ndarray/JSON string/"comma,separated"
                if isinstance(v, str):
                    vv = v.strip()
                    parsed = None
                    # try json first
                    if vv.startswith("[") and vv.endswith("]"):
                        try:
                            import json
                            parsed = json.loads(vv)
                        except Exception:
                            parsed = None
                    if parsed is None:
                        # try comma-separated numbers
                        try:
                            parsed = [float(x) for x in vv.split(",")]
                        except Exception:
                            parsed = None
                    v = parsed if parsed is not None else None
                try:
                    arr = np.asarray(v, dtype=np.float32).reshape(-1) if v is not None else np.zeros(52, np.float32)
                except Exception:
                    arr = np.zeros(52, np.float32)

            # enforce length 52 (pad/truncate)
            out = np.zeros(52, dtype=np.float32)
            n = min(52, arr.size)
            if n > 0:
                out[:n] = arr[:n]
            return torch.from_numpy(out)

        if getattr(self, "_has_board_mask", False):
            bm_t = _to_mask_52(row.get("board_mask_52", None))
        else:
            bm_t = torch.zeros(52, dtype=torch.float32)

        x_cont["board_mask_52"] = bm_t

        # --- LABELS / MASKS ---
        actor = str(row.get("actor", "ip")).lower()
        if actor not in ("ip", "oop"):
            actor = "ip"

        if self.has_soft:
            y_vec = self.y_soft[idx]  # [V]
        else:
            y_vec = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
            y_vec[int(self.action_idx[idx])] = 1.0

        if y_vec.numel() != VOCAB_SIZE:
            raise RuntimeError(f"y width {y_vec.numel()} != VOCAB_SIZE {VOCAB_SIZE}")

        facing_bet = int(row.get("facing_bet", 0) or 0)

        # Optional bet menu gating
        bet_menu = None
        if "bet_sizes" in self._df.columns:
            try:
                v = row["bet_sizes"]
                if isinstance(v, str):
                    import json
                    bet_menu = json.loads(v)
                elif isinstance(v, (list, tuple)):
                    bet_menu = list(v)
            except Exception:
                bet_menu = None

        m_actor = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        names = list(ACTION_VOCAB)

        def _has_size(menu, target):
            return any(abs(float(s) - target) < 1e-3 for s in menu)

        if facing_bet:
            legal = {"FOLD", "CALL"} | RAISE_BUCKETS
        else:
            legal = {"CHECK"} | BET_BUCKETS
            if "DONK_33" in legal and actor != "oop":
                legal.remove("DONK_33")
            if bet_menu:
                want = set()
                if _has_size(bet_menu, 0.25): want.add("BET_25")
                if _has_size(bet_menu, 0.33): want.add("BET_33"); want.add("DONK_33")
                if _has_size(bet_menu, 0.50): want.add("BET_50")
                if _has_size(bet_menu, 0.66): want.add("BET_66")
                if _has_size(bet_menu, 0.75): want.add("BET_75")
                if _has_size(bet_menu, 1.00): want.add("BET_100")
                for b in {"BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100"}:
                    if b not in want and b in legal:
                        legal.remove(b)
                if "DONK_33" not in want and "DONK_33" in legal:
                    legal.remove("DONK_33")

        for i, a in enumerate(names):
            if a in legal:
                m_actor[i] = 1.0

        if actor == "ip":
            y_ip, y_oop = y_vec, torch.zeros_like(y_vec)
            m_ip, m_oop = m_actor, torch.zeros_like(m_actor)
        else:
            y_ip, y_oop = torch.zeros_like(y_vec), y_vec
            m_ip, m_oop = torch.zeros_like(m_actor), m_actor

        w = self.weights[idx]
        return x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w

    @id_maps.setter
    def id_maps(self, value):
        self._id_maps = value

    @cards.setter
    def cards(self, value):
        self._cards = value


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

    x_cat: Dict[str, torch.Tensor] = {k: torch.stack([x[k] for x in x_cat_list], dim=0)
                                      for k in x_cat_list[0].keys()}
    x_cont: Dict[str, torch.Tensor] = {}
    for k in x_cont_list[0].keys():
        vals = [x[k] for x in x_cont_list]
        # keep cluster id as long; others are float
        if k == "board_cluster_id":
            x_cont[k] = torch.stack([v.to(torch.long) for v in vals], dim=0)
        else:
            x_cont[k] = torch.stack([v.to(torch.float32) for v in vals], dim=0)

    x_cont["eff_stack_bb"] = x_cont.pop("stack_bb")

    y_ip  = torch.stack(y_ip_list,  dim=0)  # [B,V]
    y_oop = torch.stack(y_oop_list, dim=0)  # [B,V]
    m_ip  = torch.stack(m_ip_list,  dim=0)  # [B,V]
    m_oop = torch.stack(m_oop_list, dim=0)  # [B,V]
    w     = torch.stack(w_list,     dim=0).float()  # [B]

    V = y_ip.shape[1]
    if V != len(ACTION_VOCAB):
        raise RuntimeError(f"Batch target width {V} != len(ACTION_VOCAB) {len(ACTION_VOCAB)}")

    return x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w