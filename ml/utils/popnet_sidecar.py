from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd

from ml.utils.sidecar import save_sidecar_json


def write_popnet_sidecar(
    *,
    best_ckpt: str | Path,
    ds,                   # PopulationDatasetParquet instance
    model,                # PopulationNetLit instance
    model_name: str = "PopulationNet",
) -> Optional[Path]:
    """
    Write a single sidecar JSON next to the checkpoint using save_sidecar_json.
    Ensures feature_order, cards, and id_maps are present.
    If dataset doesn't expose id_maps(), synthesize identity maps (or token maps) from ds.df.
    """
    # ---- feature order ----
    feature_order = list(getattr(ds, "x_cols", getattr(ds, "feature_order", []))) or \
                    list(getattr(getattr(model, "hparams", object()), "feature_order", []))

    # ---- cards ----
    cards: Dict[str, int] = {}
    if hasattr(ds, "cards") and callable(getattr(ds, "cards", None)):
        cards = dict(ds.cards() or {})
    elif hasattr(ds, "cards"):
        cards = dict(getattr(ds, "cards") or {})
    elif hasattr(model, "cards"):
        cards = dict(getattr(model, "cards") or {})

    # ---- id_maps (prefer dataset; else synthesize) ----
    id_maps = None
    if hasattr(ds, "id_maps") and callable(getattr(ds, "id_maps", None)):
        try:
            id_maps = ds.id_maps()
        except Exception:
            id_maps = None

    if id_maps is None:
        # Build maps from dataset df
        if not hasattr(ds, "df"):
            # can't synthesize without the dataframe
            return None
        df = ds.df
        if not feature_order:
            # need feature list to know which columns to map
            feature_order = list(df.columns)
        id_maps = {}
        for col in feature_order:
            if col not in df.columns:
                # skip unknown columns quietly
                continue
            uniq = sorted(pd.Series(df[col]).dropna().unique().tolist())
            # If integers, create identity map; else enumerate tokens
            try:
                is_int = pd.api.types.is_integer_dtype(df[col].dtype)
            except Exception:
                is_int = False
            if is_int and all(isinstance(v, (int, np.integer)) for v in uniq):
                id_maps[col] = {str(int(v)): int(v) for v in uniq}
                # Fill cards if missing: assume dense 0..max
                if col not in cards:
                    cards[col] = int(max(uniq) + 1) if len(uniq) > 0 else 1
            else:
                id_maps[col] = {str(v): i for i, v in enumerate(uniq)}
                if col not in cards:
                    cards[col] = max(len(uniq), 1)

    # Final sanity: must have feature_order and cards
    if not feature_order or not cards:
        return None

    extra = {
        "actions": ["FOLD", "CALL", "RAISE"],
        "soft_labels": True,
        "notes": "PopulationNet trained on soft labels (p_fold,p_call,p_raise).",
    }

    return save_sidecar_json(
        best_ckpt,
        model_name=model_name,
        feature_order=list(feature_order),
        cards=dict(cards),
        id_maps=id_maps,   # guaranteed non-None here
        extra=extra,
    )