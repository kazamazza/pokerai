from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Union
from ml.utils.sidecar import save_sidecar_json

def write_equity_sidecar(
    *,
    best_ckpt: Union[str, Path],
    ds,                   # EquityDatasetParquet instance (preferred source of schema)
    model,                # EquityNetLit instance (fallback for schema)
    model_name: str = "EquityNet",
) -> Optional[Path]:
    """
    Write a single sidecar JSON next to the checkpoint using save_sidecar_json.
    Produces: <checkpoint>.sidecar.json

    Pulls feature_order/cards/id_maps from the dataset (preferred) or model.
    """
    # ---------- feature order ----------
    feature_order: Sequence[str] = []
    if hasattr(ds, "feature_order") and ds.feature_order:
        feature_order = list(ds.feature_order)
    elif hasattr(ds, "x_cols") and ds.x_cols:
        feature_order = list(ds.x_cols)
    elif hasattr(model, "hparams") and getattr(model.hparams, "feature_order", None):
        feature_order = list(model.hparams.feature_order)

    # ---------- cards (categorical cardinalities) ----------
    cards: Dict[str, int] = {}
    if hasattr(ds, "cards") and callable(getattr(ds, "cards", None)):
        try:
            cards = dict(ds.cards() or {})
        except Exception:
            cards = {}
    elif hasattr(ds, "cards_info") and getattr(ds.cards_info, "cards", None):
        cards = dict(ds.cards_info.cards)
    elif hasattr(ds, "cards") and isinstance(getattr(ds, "cards"), dict):
        cards = dict(getattr(ds, "cards"))
    elif hasattr(model, "cards") and isinstance(getattr(model, "cards"), dict):
        cards = dict(getattr(model, "cards"))

    # ---------- id_maps (raw_value -> id) optional but very useful ----------
    id_maps: Optional[Dict[str, Dict[str, int]]] = None
    if hasattr(ds, "id_maps") and callable(getattr(ds, "id_maps", None)):
        try:
            id_maps = ds.id_maps()
        except Exception:
            id_maps = None

    # If we can't describe inputs properly, don't emit a misleading sidecar
    if not feature_order or not cards:
        return None

    # ---------- extra metadata ----------
    extra: Dict[str, Any] = {
        "targets": ["p_win", "p_tie", "p_lose"],
        "soft_labels": True,
        "weight_col": getattr(ds, "weight_col", "weight"),
        "notes": "EquityNet trained on soft labels (win/tie/lose).",
    }

    return save_sidecar_json(
        ckpt_path=best_ckpt,
        model_name=model_name,
        feature_order=list(feature_order),
        cards=cards,
        id_maps=id_maps,  # key is "id_maps" (not "encoders")
        extra=extra,
    )