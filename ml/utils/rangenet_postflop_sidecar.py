import json
from pathlib import Path
from typing import Sequence, Mapping, Optional, Any, Dict

from ml.models.policy_consts import ACTION_VOCAB

SIDECAR_FILENAME = "sidecar.json"

def make_postflop_policy_sidecar(
    *,
    feature_order: Sequence[str],
    cards: Mapping[str, int],
    id_maps: Mapping[str, Mapping[str, int]],
    # ⬇️ remove board_cluster_id from default cont_features
    cont_features: Sequence[str] = ("board_mask_52", "pot_bb", "eff_stack_bb"),
    action_vocab: Sequence[str] = ACTION_VOCAB,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    feature_order = list(feature_order)
    cont_features = list(cont_features)

    # If board_cluster_id is a categorical, ensure it is NOT in cont_features
    if "board_cluster_id" in feature_order and "board_cluster_id" in cont_features:
        cont_features.remove("board_cluster_id")

    cards_norm = {str(k): int(v) for k, v in cards.items()}
    id_maps_norm = {str(k): {str(a): int(b) for a, b in m.items()} for k, m in id_maps.items()}

    sc: Dict[str, Any] = {
        "model_name": "PostflopPolicy",
        "feature_order": feature_order,
        "cards": cards_norm,
        "cat_feature_order": feature_order,  # legacy alias
        "card_sizes": cards_norm,            # legacy alias
        "id_maps": id_maps_norm,
        "cont_features": cont_features,      # now clean
        "action_vocab": list(action_vocab),
        "notes": "Categoricals via id_maps; cont features as-is.",
    }

    sc["extras"] = dict(extras) if extras else {
        "board_cluster": {"type": "kmeans","artifact": "data/artifacts/board_clusters_kmeans_128.json","n_clusters": 128}
    }
    return sc


def write_postflop_policy_sidecar(
    *,
    ckpt_dir: Path | str,
    feature_order: Sequence[str],
    cards: Mapping[str, int],
    id_maps: Mapping[str, Mapping[str, int]],
    cont_features: Sequence[str] = ("board_mask_52", "pot_bb", "eff_stack_bb"),
    action_vocab: Sequence[str] = ACTION_VOCAB,
    extras: Optional[Mapping[str, Any]] = None,
    filename: str = SIDECAR_FILENAME,
) -> Path:
    """
    Write sidecar JSON next to checkpoints (e.g., checkpoints/postflop_policy/).
    Returns the written path.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sc = make_postflop_policy_sidecar(
        feature_order=feature_order,
        cards=cards,
        id_maps=id_maps,
        cont_features=cont_features,
        action_vocab=action_vocab,
        extras=extras,
    )
    path = ckpt_dir / filename
    path.write_text(json.dumps(sc, indent=2))
    return path
