# ml/config/types.py
from enum import IntEnum
from typing import Dict


class Stakes(IntEnum):
    """Table stakes / blind levels (expandable)."""
    NL2 = 0
    NL5 = 1
    NL10 = 2
    NL25 = 3
    # add NL50, NL100... as needed


class Pos(IntEnum):
    """Hero or villain position at a 6-max table (canonical order)."""
    UTG = 0
    HJ = 1
    CO = 2
    BTN = 3
    SB  = 4
    BB  = 5


class Street(IntEnum):
    """Which betting round the action occurs in."""
    PREFLOP = 0
    FLOP    = 1
    TURN    = 2
    RIVER   = 3


class Ctx(IntEnum):
    OPEN = 0
    VS_OPEN = 1
    VS_3BET = 2
    VS_4BET = 3
    # NEW: split limps
    BLIND_VS_STEAL = 4
    LIMPED_SINGLE = 5   # exactly one limper before any raise
    LIMPED_MULTI  = 6   # two or more limpers before any raise

    VS_CBET = 10
    VS_CBET_TURN = 11
    VS_CHECK_RAISE = 13
    VS_DONK = 14

    # (Optional: add VS_DELAYED_CBET, etc. later if needed)


class Act(IntEnum):
    """Actions available to hero."""
    FOLD        = 0
    CALL        = 1
    RAISE       = 2       # Generic raise (can be split later)
    CHECK       = 3       # Postflop only
    BET         = 4       # Postflop only
    ALL_IN      = 5       # Explicit shove (separate from RAISE)


class Flag(IntEnum):
    """Binary/auxiliary flags that may condition the context."""
    SINGLEWAY   = 0   # Default: only opener + hero
    MULTIWAY    = 1   # Multiple villains in pot


POSITIONS: Dict[int, list[Pos]] = {
    2: [Pos.SB, Pos.BB],                           # heads-up
    3: [Pos.SB, Pos.BB, Pos.BTN],                  # 3-handed
    4: [Pos.SB, Pos.BB, Pos.UTG, Pos.BTN],         # 4-handed
    5: [Pos.SB, Pos.BB, Pos.UTG, Pos.CO, Pos.BTN], # 5-handed
    6: [Pos.SB, Pos.BB, Pos.UTG, Pos.HJ, Pos.CO, Pos.BTN],
}