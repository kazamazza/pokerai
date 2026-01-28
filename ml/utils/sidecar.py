# ml/utils/sidecar.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Mapping, Optional, Any, Dict


class ModelSidecarBuilder:
    """
    Generic sidecar builder for *all* models (equity, preflop, postflop, etc).

    This class is intentionally boring.
    """

    def __init__(
        self,
        *,
        model_name: str,
        feature_order: Sequence[str],
        cont_features: Sequence[str],
        id_maps: Mapping[str, Mapping[str, int]],
        action_vocab: Optional[Sequence[str]] = None,
        cards: Optional[Mapping[str, int]] = None,
        extras: Optional[Mapping[str, Any]] = None,
    ):
        self.model_name = model_name
        self.feature_order = list(feature_order)
        self.cont_features = list(cont_features)
        self.id_maps = {
            str(k): {str(a): int(b) for a, b in m.items()}
            for k, m in id_maps.items()
        }
        self.action_vocab = list(action_vocab) if action_vocab else None
        self.cards = {str(k): int(v) for k, v in cards.items()} if cards else None
        self.extras = dict(extras) if extras else {}

        self._validate()

    # --------------------------------------------------
    # Validation (fail fast, once)
    # --------------------------------------------------
    def _validate(self) -> None:
        # board_cluster_id must not be both cat + cont
        if "board_cluster_id" in self.feature_order and "board_cluster_id" in self.cont_features:
            raise ValueError("board_cluster_id cannot be both categorical and continuous")

        if not self.feature_order:
            raise ValueError("feature_order cannot be empty")

    # --------------------------------------------------
    # Build payload
    # --------------------------------------------------
    def build(self) -> Dict[str, Any]:
        sc: Dict[str, Any] = {
            "model_name": self.model_name,
            "feature_order": self.feature_order,
            "cat_feature_order": self.feature_order,  # legacy alias
            "id_maps": self.id_maps,
            "cont_features": self.cont_features,
            "notes": "Categorical features via id_maps; continuous features passed as-is.",
        }

        if self.action_vocab is not None:
            sc["action_vocab"] = self.action_vocab

        if self.cards is not None:
            sc["cards"] = self.cards
            sc["card_sizes"] = self.cards  # legacy alias

        if self.extras:
            sc["extras"] = self.extras

        return sc

    # --------------------------------------------------
    # Write
    # --------------------------------------------------
    def write(self, ckpt_dir: Path | str, filename: str = "sidecar.json") -> Path:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        path = ckpt_dir / filename
        path.write_text(json.dumps(self.build(), indent=2))
        return path