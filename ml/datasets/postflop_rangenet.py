import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

from ml.models.policy_consts import VOCAB_SIZE, VOCAB_INDEX, ACTION_VOCAB

CAT_FEATURES_DEFAULT = ["hero_pos", "ip_pos", "oop_pos", "ctx", "street"]
BET_BUCKETS   = {"BET_25","BET_33","BET_50","BET_66","BET_75","BET_100","DONK_33"}
RAISE_BUCKETS = {"RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"}

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
        force_hard: bool = False,             # allow forcing hard labels if desired
    ):
        self.parquet_path = Path(parquet_path)
        self.df = pd.read_parquet(str(self.parquet_path)).copy()
        self.device = device
        self.force_hard = bool(force_hard)

        def canon_pos(v: str | None) -> Optional[str]:
            if v is None: return None
            v = str(v).strip().upper()
            aliases = {
                "UTG+1": "HJ", "UTG+2": "CO", "UTG1": "HJ", "UTG2": "CO",
                "DEALER": "BTN", "BUTTON": "BTN",
                "SMALL BLIND": "SB", "BIG BLIND": "BB",
            }
            return aliases.get(v, v)

        def canon_ctx(v: str | None) -> Optional[str]:
            if v is None: return None
            return str(v).strip().upper()

        def canon_street(v) -> Optional[str]:
            if v is None: return None
            if isinstance(v, int):
                return {1: "FLOP", 2: "TURN", 3: "RIVER"}.get(v, str(v))
            m = {"1": "FLOP", "2": "TURN", "3": "RIVER"}
            return m.get(str(v).strip().upper(), str(v).strip().upper())

        canon_map = {
            "hero_pos": canon_pos,
            "ip_pos": canon_pos,
            "oop_pos": canon_pos,
            "ctx": canon_ctx,
            "street": canon_street,
        }

        for col, fn in canon_map.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(fn)

        self._id_maps: Dict[str, Dict[str, int]] = {}

        def _ensure_id(col: str):
            if col not in self.df.columns:
                return
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
                        raise ValueError(f"Invalid/uncanonical values in '{c}': {len(bad)} row(s)")

        if weight_col in self.df.columns:
            w = self.df[weight_col].astype(float).values
            w = np.where(np.isnan(w), 1.0, w)
            self.weights = torch.tensor(w, dtype=torch.float32)
        else:
            self.weights = torch.ones(len(self.df), dtype=torch.float32)

        if not self.SOFT_Y_COLS:
            self.SOFT_Y_COLS = list(ACTION_VOCAB)

        have_all_soft_cols = (len(self.SOFT_Y_COLS) > 0) and all(c in self.df.columns for c in self.SOFT_Y_COLS)
        has_soft = (not self.force_hard) and have_all_soft_cols
        self.has_soft = bool(has_soft)

        if not self.has_soft and "action" not in self.df.columns:
            raise ValueError("Parquet must have either per-action columns (soft) or an 'action' column (hard).")

        if self.has_soft:
            y = torch.tensor(self.df[self.SOFT_Y_COLS].values, dtype=torch.float32)
            if y.shape[1] != VOCAB_SIZE:
                missing = [c for c in ACTION_VOCAB if c not in self.df.columns]
                raise ValueError(
                    f"Soft label width {y.shape[1]} != VOCAB_SIZE {VOCAB_SIZE}. "
                    f"Missing columns: {missing}"
                )
            y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-8)
            self.y_soft = y
        else:
            act_idx = (
                self.df["action"]
                .astype(str)
                .str.upper()
                .map(lambda a: VOCAB_INDEX.get(a, None))
            )
            if act_idx.isna().any():
                bad = sorted(self.df.loc[act_idx.isna(), "action"].astype(str).str.upper().unique().tolist())
                raise ValueError(f"Unmapped actions found: {bad}")
            self.action_idx = torch.tensor(act_idx.values, dtype=torch.long)

        self.cat_features = [c for c in self.CAT_FEATURES if c in self.df.columns]
        self.cont_features = ["stack_bb", "pot_bb"]
        # Normalize possible column variants to a single name in memory
        if "board_cluster_id" in self.df.columns:
            self._has_cluster_id = True
        elif "board_cluster" in self.df.columns:
            # if you had an older name, normalize it
            self.df = self.df.rename(columns={"board_cluster": "board_cluster_id"})
            self._has_cluster_id = True
        else:
            self._has_cluster_id = False

        self._has_board_mask = "board_mask_52" in self.df.columns

    def cards(self) -> Dict[str, int]:
        sizes: Dict[str, int] = {}
        for c in self.cat_features:
            if c in self._id_maps:
                sizes[c] = len(self._id_maps[c])
            else:
                sizes[c] = int(self.df[c].max()) + 1
        return sizes

    def id_maps(self) -> Dict[str, Dict[str, int]]:
        return dict(self._id_maps)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x_cat: Dict[str, torch.Tensor] = {
            f: torch.tensor(int(row[f]), dtype=torch.long)
            for f in self.cat_features
        }
        x_cont: Dict[str, torch.Tensor] = {
            f: torch.tensor(float(row[f]), dtype=torch.float32).view(1)
            for f in self.cont_features
        }

        if self._has_board_mask:
            bm = row["board_mask_52"]
            if isinstance(bm, torch.Tensor):
                bm_t = bm.float().view(-1)
            else:
                bm_t = torch.tensor(np.asarray(bm, dtype=np.float32)).view(-1)
            if bm_t.numel() != 52:
                pad = torch.zeros(52, dtype=torch.float32)
                pad[: min(52, bm_t.numel())] = bm_t[: min(52, bm_t.numel())]
                bm_t = pad
            x_cont["board_mask_52"] = bm_t
        else:
            x_cont["board_mask_52"] = torch.zeros(52, dtype=torch.float32)

        if hasattr(self, "_has_cluster_id") and self._has_cluster_id:
            try:
                cid = int(row["board_cluster_id"])
            except KeyError:
                # fallback if parquet used old name
                cid = int(row.get("board_cluster", 0))
            x_cont["board_cluster_id"] = torch.tensor(cid, dtype=torch.long)

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

        bet_menu = None
        if "bet_sizes" in self.df.columns:
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
            # Not facing → CHECK and BET_* legal (optionally restrict to menu sizes)
            legal = {"CHECK"} | BET_BUCKETS
            if "DONK_33" in legal:
                if actor != "oop":
                    legal.remove("DONK_33")
            want = set()
            if bet_menu:
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