# ml/range/solvers/context_profiles.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Sequence, Optional

Street = Literal["flop","turn","river"]
Role   = Literal["ip","oop"]

BetMenu = dict[Street, dict[Role, dict[str, Sequence[float] | bool]]]

@dataclass(frozen=True)
class ContextProfile:
    name: str
    pot_bb: float                 # default flop pot for this context
    min_spr: float                # guard; below this we either strip AI or refuse
    strip_allins_when_spr_ge: float  # e.g. 3.0 → no flop/turn shove at healthy SPR
    accuracy: float               # target eps
    max_iter: int
    allin_threshold: float
    bet_menu_id: str
    bet_menu: BetMenu

STD_MENU: BetMenu = {
    "flop": {
        "oop": {"bet": [33, 50, 75], "raise": [66, 100, 150], "allin": True},
        "ip":  {"bet": [25, 33, 50, 75], "raise": [66, 100, 150], "allin": True},
    },
    "turn": {
        "oop": {"bet": [50, 66, 100], "raise": [100, 150], "allin": True},
        "ip":  {"bet": [50, 66, 100], "raise": [100, 150], "allin": True},
    },
    "river": {
        "oop": {"bet": [50, 66, 100], "raise": [100, 150], "allin": True},
        "ip":  {"bet": [50, 66, 100], "raise": [100, 150], "allin": True},
    },
}

# Slightly tighter flop sizings for 3b/4b pots (shallower SPR, bigger sizes)
THREEBET_MENU: BetMenu = {
    "flop": {
        "oop": {"bet": [33, 50, 66], "raise": [100, 150], "allin": True},
        "ip":  {"bet": [33, 50, 66], "raise": [100, 150], "allin": True},
    },
    "turn": STD_MENU["turn"],
    "river": STD_MENU["river"],
}

FOURBET_MENU: BetMenu = {
    "flop": {
        "oop": {"bet": [33, 50], "raise": [100, 150], "allin": True},
        "ip":  {"bet": [33, 50], "raise": [100, 150], "allin": True},
    },
    "turn": STD_MENU["turn"],
    "river": STD_MENU["river"],
}

PROFILES: Dict[str, ContextProfile] = {
    # “VS_OPEN” = SRP heads-up
    "VS_OPEN": ContextProfile(
        name="VS_OPEN",
        pot_bb=6.5,                 # template SRP pot
        min_spr=3.0,
        strip_allins_when_spr_ge=3.0,
        accuracy=0.01,
        max_iter=15000,
        allin_threshold=0.67,
        bet_menu_id="std",
        bet_menu=STD_MENU,
    ),
    # 3-bet pot
    "VS_3BET": ContextProfile(
        name="VS_3BET",
        pot_bb=20.0,                # typical HU 3b pot
        min_spr=1.5,
        strip_allins_when_spr_ge=2.5,
        accuracy=0.01,
        max_iter=18000,
        allin_threshold=0.67,
        bet_menu_id="3b",
        bet_menu=THREEBET_MENU,
    ),
    # 4-bet pot
    "VS_4BET": ContextProfile(
        name="VS_4BET",
        pot_bb=48.0,
        min_spr=0.8,
        strip_allins_when_spr_ge=2.0,
        accuracy=0.009,
        max_iter=20000,
        allin_threshold=0.67,
        bet_menu_id="4b",
        bet_menu=FOURBET_MENU,
    ),
    # Limped pots are often multiway; keep but expect lower SPR & more donk
    "LIMPED_SINGLE": ContextProfile(
        name="LIMPED_SINGLE",
        pot_bb=3.0,
        min_spr=3.0,
        strip_allins_when_spr_ge=3.0,
        accuracy=0.012,
        max_iter=12000,
        allin_threshold=0.67,
        bet_menu_id="std",
        bet_menu=STD_MENU,
    ),
    "LIMPED_MULTI": ContextProfile(
        name="LIMPED_MULTI",
        pot_bb=4.0,
        min_spr=2.5,
        strip_allins_when_spr_ge=3.0,
        accuracy=0.012,
        max_iter=12000,
        allin_threshold=0.67,
        bet_menu_id="std",
        bet_menu=STD_MENU,
    ),
}