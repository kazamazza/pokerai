import json
from pathlib import Path
from typing import Sequence, Mapping, Optional, Any, Dict

from ml.models.policy_consts import ACTION_VOCAB

SIDECAR_FILENAME = "postflop_policy_sidecar.json"
SIDECAR_VERSION  = 1

def make_postflop_policy_sidecar(
    *,
    feature_order: Sequence[str],
    cards: Mapping[str, int],
    id_maps: Mapping[str, Mapping[str, int]],
    cont_features: Sequence[str] = ("board_mask_52", "pot_bb", "eff_stack_bb"),
    action_vocab: Sequence[str] = ACTION_VOCAB,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Canonical, compact sidecar for postflop policy.
    Contains everything inference needs to rebuild encoders and tensors.
    """
    sc: Dict[str, Any] = {
        "sidecar_version": SIDECAR_VERSION,
        "model_name": "PostflopPolicy",
        "cat_feature_order": list(feature_order),     # order for x_cat dict
        "card_sizes": {str(k): int(v) for k, v in cards.items()},
        "id_maps": {str(k): {str(a): int(b) for a, b in m.items()} for k, m in id_maps.items()},
        "cont_features": list(cont_features),         # names expected in x_cont
        "action_vocab": list(action_vocab),
        "notes": "Categoricals are integer IDs via id_maps; cont features as-is.",
    }
    if extras:
        sc["extras"] = dict(extras)
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
