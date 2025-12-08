# ml/etl/ev/common.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd

def sanitize_ev_col(tok: str) -> str:
    return "ev__" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in tok.upper())

def stakes_id_from_cfg(cfg: Dict[str, Any], default: str = "2") -> str:
    try:
        m = (cfg.get("encoders") or {}).get("id_maps", {}).get("stakes_id", {})
        return next(iter(m.keys()))
    except Exception:
        return default

def pairs_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs = (cfg.get("build") or {}).get("pairs") or []
    out: List[Tuple[str, str]] = []
    for p in pairs:
        try:
            a, b = p
            out.append((str(a).upper(), str(b).upper()))
        except Exception:
            continue
    return out

def faced_fracs_for_stack(cfg: Dict[str, Any], stack_bb: float) -> List[float]:
    b = cfg.get("build") or {}
    ff = b.get("faced_fracs")
    if isinstance(ff, (list, tuple)) and len(ff) > 0:
        return [float(x) for x in ff]
    faced_bb = b.get("faced_bb")
    if isinstance(faced_bb, (list, tuple)) and len(faced_bb) > 0 and float(stack_bb) > 0:
        return [float(x) / float(stack_bb) for x in faced_bb]
    return [0.0]

def fill_missing_ev_cols(df: pd.DataFrame) -> pd.DataFrame:
    ev_cols = [c for c in df.columns if c.startswith("ev__")]
    for c in ev_cols:
        df[c] = df[c].fillna(0.0)
    return df

def write_outputs(df: pd.DataFrame, cfg: Dict[str, Any], *, manifest_key: str, parquet_key: str) -> None:
    paths = cfg.get("paths") or {}
    manifest_path = paths.get(manifest_key, "data/artifacts/tmp_manifest.parquet")
    parquet_path  = paths.get(parquet_key,  "data/datasets/tmp_dataset.parquet")
    df.to_parquet(manifest_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"✅ wrote manifest: {manifest_path}  rows={len(df):,}")
    print(f"✅ wrote parquet:  {parquet_path}   rows={len(df):,}")