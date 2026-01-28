from __future__ import annotations

from typing import Any, Dict, Literal

from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB

def make_root_policy_payload(
    *,
    common: Dict[str, Any],
    solver_key: str,
    solver_version: str,
    size_pct: int,
    probs: Dict[str, float],
    weight: float = 1.0,
    valid: bool = True,
) -> Dict[str, Any]:
    # Flatten vocab probabilities into columns
    out = {
        "sha1": common["sha1"],
        "s3_key": solver_key,
        "solver_version": str(solver_version),

        "street": int(common["street"]),
        "board": str(common["board"]),
        "board_mask_52": common["board_mask_52"],
        "pot_bb": float(common["pot_bb"]),
        "effective_stack_bb": float(common["effective_stack_bb"]),

        # policy rows are from OOP root actor by construction
        "hero_pos": "OOP",
        "villain_pos": "IP",
        "ctx": str(common["ctx"]),

        "size_pct": int(size_pct),

        "action_vocab": "ROOT",
        "weight": float(weight),
        "valid": bool(valid),
    }

    # init all vocab cols
    for a in ROOT_ACTION_VOCAB:
        out[a] = float(probs.get(a, 0.0))

    # choose label action for convenience
    if valid:
        best = max(ROOT_ACTION_VOCAB, key=lambda a: out[a])
        out["action"] = best
    else:
        out["action"] = "CHECK"

    return out


def make_facing_policy_payload(
    *,
    common: Dict[str, Any],
    solver_key: str,
    solver_version: str,
    faced_size_pct: int,
    probs: Dict[str, float],
    weight: float = 1.0,
    valid: bool = True,
) -> Dict[str, Any]:
    out = {
        "sha1": common["sha1"],
        "s3_key": solver_key,
        "solver_version": str(solver_version),

        "street": int(common["street"]),
        "board": str(common["board"]),
        "board_mask_52": common["board_mask_52"],
        "pot_bb": float(common["pot_bb"]),
        "effective_stack_bb": float(common["effective_stack_bb"]),

        # facing rows: “hero” is whoever is acting in facing node for your dataset.
        # v1: keep it consistent with root (OOP viewpoint).
        "hero_pos": "OOP",
        "villain_pos": "IP",
        "ctx": str(common["ctx"]),

        "faced_size_pct": int(faced_size_pct),

        "action_vocab": "FACING",
        "weight": float(weight),
        "valid": bool(valid),
    }

    for a in FACING_ACTION_VOCAB:
        out[a] = float(probs.get(a, 0.0))

    if valid:
        best = max(FACING_ACTION_VOCAB, key=lambda a: out[a])
        out["action"] = best
    else:
        out["action"] = "CALL"

    return out