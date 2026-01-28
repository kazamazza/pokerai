from dataclasses import dataclass
from typing import Dict, List, Literal


# ============================
# ROOT POLICY (no bet faced)
# ============================

@dataclass(frozen=True)
class PostflopPolicyRootRow:
    # --- identity / trace ---
    sha1: str
    s3_key: str
    solver_version: str

    # --- game state ---
    street: int                    # 1=flop, 2=turn, 3=river
    board: str                     # e.g. "Ah9hAs"
    board_mask_52: List[float]     # len=52, 0/1
    pot_bb: float
    effective_stack_bb: float

    # --- positions ---
    hero_pos: str                  # "IP" or "OOP"
    villain_pos: str               # opposite of hero
    ctx: str                       # SRP / VS_3BET / VS_4BET / LIMPED_*

    # --- sizing ---
    size_pct: int                  # concrete size used for this solve (e.g. 33)

    # --- policy ---
    action_vocab: Literal["ROOT"]
    action_probs: Dict[str, float] # keys ⊆ ROOT_ACTION_VOCAB

    # --- training ---
    weight: float                  # usually 1.0
    valid: bool                    # False for sentinel rows


# ============================
# FACING POLICY (bet faced)
# ============================

@dataclass(frozen=True)
class PostflopPolicyFacingRow:
    # --- identity / trace ---
    sha1: str
    s3_key: str
    solver_version: str

    # --- game state ---
    street: int
    board: str
    board_mask_52: List[float]
    pot_bb: float
    effective_stack_bb: float

    # --- positions ---
    hero_pos: str
    villain_pos: str
    ctx: str

    # --- sizing ---
    faced_size_pct: int            # size of bet we are facing

    # --- policy ---
    action_vocab: Literal["FACING"]
    action_probs: Dict[str, float] # keys ⊆ FACING_ACTION_VOCAB

    # --- training ---
    weight: float
    valid: bool