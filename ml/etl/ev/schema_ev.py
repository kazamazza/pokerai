# file: ml/etl/ev/schema_ev.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Sequence

from ml.models.vocab_actions import PREFLOP_ACTION_VOCAB


@dataclass(frozen=True)
class PreflopEVSchema:
    # Column order for emitted parquet
    cat_cols: Sequence[str] = (
        "stakes_id",       # string id-map token e.g. "2" for NL10 per sidecar style
        "hero_pos_raw",    # UTG/HJ/CO/BTN/SB/BB
        "villain_pos_raw", # UTG/HJ/CO/BTN/SB/BB
    )
    cont_cols: Sequence[str] = (
        "street",          # 0
        "pot_bb",          # starting pot (e.g., 1.5)
        "eff_stack_bb",
        "faced_size_bb",   # 0 if unopened
    )
    aux_cols: Sequence[str] = (
        "ctx",             # e.g., VS_OPEN
        "action_seq_1", "action_seq_2", "action_seq_3",
        "facing_bet",      # bool
        "hero_hand",       # optional; may be ""
        "rowsrc",          # provenance tag
    )
    # Label vector columns aligned to vocab
    label_cols: Sequence[str] = tuple(f"ev_{t}" for t in PREFLOP_ACTION_VOCAB)
    mask_cols:  Sequence[str] = tuple(f"legal_{t}" for t in PREFLOP_ACTION_VOCAB)

    @property
    def all_cols(self) -> List[str]:
        return list(self.cat_cols) + list(self.cont_cols) + list(self.aux_cols) + list(self.label_cols) + list(self.mask_cols)

SCHEMA_PREFLOP = PreflopEVSchema()