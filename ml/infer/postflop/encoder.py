from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PostflopInferArtifacts:
    """
    Loaded artifacts needed for inference encoding:
      - sidecar.json written at training time
      - cat_vocabs.json produced by the vocab builder
    """
    sidecar: Dict[str, Any]
    cat_vocabs: Dict[str, Dict[str, int]]  # col -> {token -> id}

    feature_order: List[str]
    cont_features: List[str]
    action_vocab: List[str]


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_postflop_infer_artifacts(
    *,
    sidecar_path: str | Path,
    cat_vocabs_path: str | Path,
) -> PostflopInferArtifacts:
    sidecar = _load_json(Path(sidecar_path))
    voc = _load_json(Path(cat_vocabs_path))

    # We expect vocabs JSON like: {"vocabs": {"ctx": {...}, "ip_pos": {...}, ...}}
    cat_vocabs = voc.get("vocabs", voc)  # tolerate either shape

    feature_order = list(sidecar.get("feature_order") or sidecar.get("cat_feature_order") or [])
    cont_features = list(sidecar.get("cont_features") or [])
    action_vocab = list(sidecar.get("action_vocab") or [])

    if not feature_order:
        raise ValueError("sidecar missing feature_order")
    if not isinstance(cat_vocabs, dict) or not cat_vocabs:
        raise ValueError("cat_vocabs_json missing/invalid")

    return PostflopInferArtifacts(
        sidecar=sidecar,
        cat_vocabs=cat_vocabs,
        feature_order=feature_order,
        cont_features=cont_features,
        action_vocab=action_vocab,
    )


def _get_vocab_id(vocab: Mapping[str, int], value: Any) -> int:
    """
    Map raw categorical value -> id.
    Unknowns map to 0 if present, else to 0 anyway.
    """
    key = "" if value is None else str(value)
    if key in vocab:
        return int(vocab[key])
    # common unknown token conventions
    for unk in ("<UNK>", "UNK", "__UNK__", "unknown", "UNKNOWN"):
        if unk in vocab:
            return int(vocab[unk])
    return 0


def encode_postflop_state_dense(
    *,
    state: Dict[str, Any],
    artifacts: PostflopInferArtifacts,
) -> torch.Tensor:
    """
    Encode a postflop state into a single dense float tensor [input_dim].

    Encoding strategy (simple + stable):
      - categorical columns in feature_order that exist in cat_vocabs:
          -> id normalized to [0..1] by id/(vocab_size-1)
      - scalar numeric columns (in feature_order but not in cat_vocabs and not in cont_features):
          -> float(value)
      - cont_features (vectors, e.g. board_mask_52):
          -> concatenated as floats

    This matches the “MLP expects dense numeric features already encoded” approach.
    """
    x: List[float] = []

    # 1) scalar features in feature_order
    for col in artifacts.feature_order:
        if col in artifacts.cont_features:
            # cont vectors handled later
            continue

        if col in artifacts.cat_vocabs:
            vocab = artifacts.cat_vocabs[col]
            vid = _get_vocab_id(vocab, state.get(col))
            denom = max(1, len(vocab) - 1)
            x.append(float(vid) / float(denom))
        else:
            v = state.get(col)
            try:
                x.append(float(v))
            except Exception:
                x.append(0.0)

    # 2) cont vectors
    for col in artifacts.cont_features:
        v = state.get(col)
        if v is None:
            # If missing, emit zeros of known length when possible (board_mask_52=52)
            if col == "board_mask_52":
                x.extend([0.0] * 52)
            else:
                # unknown cont size => skip (or you could raise)
                pass
            continue

        if isinstance(v, (list, tuple)):
            x.extend([float(a) for a in v])
        else:
            # tolerate numpy arrays or strings that are json lists
            try:
                if hasattr(v, "tolist"):
                    arr = v.tolist()
                    x.extend([float(a) for a in arr])
                else:
                    x.extend([float(v)])
            except Exception:
                if col == "board_mask_52":
                    x.extend([0.0] * 52)

    return torch.tensor(x, dtype=torch.float32)