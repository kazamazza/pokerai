from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import json, gzip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ml.utils.enums import Types, load_types

def _one_hot(idx: int, size: int) -> torch.Tensor:
    v = torch.zeros(size, dtype=torch.float32)
    if 0 <= idx < size: v[idx] = 1.0
    return v

def _safe_float(x, default=0.0) -> float:
    try: return float(x)
    except: return float(default)

class PolicyNetParquetDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        types: Types,
        yaml_path="ml/config/settings.yaml",
        target: str = "dist",                # "dist" or "argmax"
        ctx_vocab: Optional[List[str]] = None,
        pos_vocab: Optional[List[str]] = None,
        board_vocab: int = 256,              # cluster count (0 disables board one-hot)
        stack_scale: float = 100.0,          # scale stacks to ~[0,1]
        drop_invalid: bool = True,

    ):
        self.path = Path(path)
        self.types = load_types(yaml_path)
        self.target = target
        self.pos_list = pos_vocab or types.positions
        self.ctx_list = ctx_vocab or types.action_contexts
        self.pos2i = {p:i for i,p in enumerate(self.pos_list)}
        self.ctx2i = {c:i for i,c in enumerate(self.ctx_list)}
        self.board_vocab = int(board_vocab)
        self.stack_scale = float(stack_scale)

        df = self._read(self.path)
        req_x = ["hero_pos","street","ctx","stack_bb"]
        missing = [c for c in req_x if c not in df.columns]
        if missing:
            raise ValueError(f"{self.path} missing required feature columns: {missing}")

        # Ensure a target exists
        has_dist = "action_probs" in df.columns
        has_cls  = "y_class" in df.columns
        if self.target == "dist" and not has_dist:
            raise ValueError(f"{self.path} missing 'action_probs' for target=dist")
        if self.target == "argmax" and not (has_cls or has_dist):
            raise ValueError(f"{self.path} needs 'y_class' or 'action_probs'")

        # Coerce/clean
        df["hero_pos"] = df["hero_pos"].astype(str)
        df["street"]   = df["street"].astype(str).str.lower()
        df["ctx"]      = df["ctx"].astype(str)
        df["stack_bb"] = pd.to_numeric(df["stack_bb"], errors="coerce").astype(float)

        # Optional numerics
        for k in ["pot_bb","amount_to_call_bb","last_raise_to_bb","min_raise_to_bb","spr"]:
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors="coerce").astype(float)

        # Optional board index
        if "board_idx" in df.columns:
            df["board_idx"] = pd.to_numeric(df["board_idx"], errors="coerce").astype("Int64")

        # Targets
        if has_dist:
            df["action_probs"] = df["action_probs"].apply(self._coerce_probs)
            if drop_invalid:
                df = df[df["action_probs"].map(lambda v: isinstance(v, list) and len(v) >= 2)]
        if has_cls:
            df["y_class"] = pd.to_numeric(df["y_class"], errors="coerce").astype("Int64")

        # Determine output dim (K)
        if self.target == "dist":
            first = None
            for v in df["action_probs"]:
                if isinstance(v, list) and len(v) >= 2:
                    first = v; break
            if first is None:
                raise ValueError("No valid 'action_probs' rows found.")
            self.out_dim = len(first)
        else:  # argmax
            # If we also have probs, infer K from there; else require K in config later
            if has_dist:
                self.out_dim = len(df["action_probs"].iloc[0])
            else:
                self.out_dim = None  # will only be used by the model head constructor

        # Cache
        self.df = df.reset_index(drop=True)
        self.N = len(self.df)

        # Compute and expose input dimension
        self.input_len = self._infer_input_dim()

    # ---------- IO ----------
    def _read(self, p: Path) -> pd.DataFrame:
        s = str(p).lower()
        if s.endswith(".parquet"):
            return pd.read_parquet(p)
        # fallback: JSONL (optionally gz)
        op = gzip.open if s.endswith(".gz") else open
        rows: List[Dict[str,Any]] = []
        with op(p, "rt", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                rows.append(json.loads(ln))
        # Allow {x:..., y:...} or flat rows
        flat = []
        for r in rows:
            if "x" in r or "y" in r:
                x = r.get("x", {}); y = r.get("y", {})
                flat.append({**x, **y})
            else:
                flat.append(r)
        return pd.DataFrame(flat)

    @staticmethod
    def _coerce_probs(v) -> Optional[List[float]]:
        if v is None: return None
        if isinstance(v, str):
            try: v = json.loads(v)
            except: return None
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=np.float32)
            if arr.size == 0: return None
            if np.any(~np.isfinite(arr)): return None
            s = float(arr.sum())
            if s <= 0:  # degenerate → put mass on fold(0)
                arr = np.zeros_like(arr); arr[0] = 1.0
            else:
                arr = arr / s
            return arr.tolist()
        return None

    # ---------- vectorization ----------
    def _infer_input_dim(self) -> int:
        # hero_pos (6) + street (4) + ctx (len(ctx_list)) + stack(1)
        dim = len(self.pos_list) + len(self.types.streets) + len(self.ctx_list) + 1
        # optional scalars we include
        for k in ["pot_bb","amount_to_call_bb","last_raise_to_bb","min_raise_to_bb","spr"]:
            if k in self.df.columns: dim += 1
        # board one-hot
        if self.board_vocab and "board_idx" in self.df.columns:
            dim += self.board_vocab
        return dim

    def _row_to_vec(self, r: pd.Series) -> torch.Tensor:
        # hero_pos
        pos_i = self.pos2i.get(r.hero_pos, -1)
        v_pos = _one_hot(pos_i, len(self.pos_list))

        # street
        try:
            street_i = self.types.streets.index(str(r.street).lower())
        except ValueError:
            street_i = -1
        v_street = _one_hot(street_i, len(self.types.streets))

        # ctx
        ctx_i = self.ctx2i.get(r.ctx, -1)
        v_ctx = _one_hot(ctx_i, len(self.ctx_list))

        # core scalar: stack
        v_core = torch.tensor([_safe_float(r.stack_bb)/self.stack_scale], dtype=torch.float32)

        # optional scalars
        scalars = []
        for k in ["pot_bb","amount_to_call_bb","last_raise_to_bb","min_raise_to_bb","spr"]:
            if k in r.index:
                scalars.append(_safe_float(r[k]))
        v_scalars = torch.tensor(scalars, dtype=torch.float32) if scalars else torch.zeros(0)

        # board one-hot (cluster id 0..board_vocab-1); if missing → zeros
        if self.board_vocab and ("board_idx" in r.index) and pd.notna(r["board_idx"]):
            bi = int(r["board_idx"])
            v_board = _one_hot(bi, self.board_vocab)
        else:
            v_board = torch.zeros(self.board_vocab, dtype=torch.float32) if self.board_vocab else torch.zeros(0)

        return torch.cat([v_pos, v_street, v_ctx, v_core, v_scalars, v_board], dim=0)

    # ---------- torch Dataset API ----------
    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        x_vec = self._row_to_vec(r)

        if self.target == "dist":
            tgt = torch.tensor([float(v) for v in r.action_probs], dtype=torch.float32)
            s = float(tgt.sum().item())
            if s <= 0:
                tgt = torch.zeros_like(tgt); tgt[0] = 1.0
            else:
                tgt = tgt / s
            Y = {"y_dist": tgt}
        else:
            # prefer explicit y_class; else argmax of probs
            if "y_class" in r and pd.notna(r.y_class):
                cls = int(r.y_class)
            else:
                probs = np.asarray(r.action_probs, dtype=np.float32)
                cls = int(np.argmax(probs))
            Y = {"y_class": torch.tensor([cls], dtype=torch.long)}

        return {"x_vec": x_vec}, Y