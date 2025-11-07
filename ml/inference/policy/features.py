from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .board import parse_board_str, make_board_mask_52, map_cluster_id
from .types import PolicyRequest


@dataclass
class Sidecar:
    feature_order: list
    cont_features: list | None
    id_maps: Dict[str, Dict[str, int]] | None
    action_vocab: list | None
    board_cluster_feat: Optional[str] = "board_cluster_id"

def build_postflop_row(req: "PolicyRequest", sidecar: Sidecar, clusterer: Optional[Any]) -> Dict[str, Any]:
    # why: align to model expectations deterministically
    board_cards = parse_board_str(req.board)
    bm52 = req.board_mask_52 or make_board_mask_52(board_cards)
    hero_is_ip = PolicyRequest.is_hero_ip(req.hero_pos or "", req.villain_pos or "")
    ip_pos = (req.hero_pos if hero_is_ip else req.villain_pos) or ""
    oop_pos = (req.villain_pos if hero_is_ip else req.hero_pos) or ""
    row = {
        "hero_pos": (req.hero_pos or "").upper(),
        "ip_pos": ip_pos,
        "oop_pos": oop_pos,
        "ctx": (req.ctx or req.raw.get("ctx") or "VS_OPEN"),
        "street": int(req.street),
        "stakes_id": req.stakes,  # if categorical, router should map via sidecar.id_maps
        "board_mask_52": bm52,
        "pot_bb": float(req.pot_bb),
        "stack_bb": float(req.eff_stack_bb),
        "size_frac": float(req.faced_size_frac or (req.faced_size_pct or 0.0) / 100.0),
    }
    # cluster id
    if sidecar.board_cluster_feat:
        cid_raw = 0
        if clusterer is not None and any(c != "__" for c in board_cards):
            try:
                cid_raw = int(clusterer.predict_one(board_cards))
            except Exception:
                cid_raw = 0
        cid = cid_raw
        if sidecar.id_maps and "board_cluster_id" in sidecar.id_maps:
            cid = map_cluster_id(cid_raw, sidecar.id_maps["board_cluster_id"])  # remap through id_map
        row[sidecar.board_cluster_feat] = cid
    return row