import os
from enum import Enum
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

class ActionType(str, Enum):
    FOLD     = "fold"
    CHECK    = "check"
    CALL     = "call"
    BET      = "bet"
    RAISE    = "raise"
    POST_SB  = "post_sb"
    POST_BB  = "post_bb"

ACTION_TYPES = list(ActionType)
ACTION_TO_INDEX = {a: i for i, a in enumerate(ACTION_TYPES)}

class Action(BaseModel):
    player_id: str
    type: ActionType
    amount: Optional[float] = None

class Status(str, Enum):
    active = "active"
    sitting_out = "sitting out"

class Seat(BaseModel):
    seat_number: int
    player_id: str
    stack_size: float
    status: Status

class Hand(BaseModel):
    hand_id:    str
    seats:      List[Seat]
    button_seat:int
    actions:    List[Action]    # <-- now a list of structured Actions
    board:      List[str]
    street:     str


class RangeFeatureRequest(BaseModel):
    history: List[Hand]         # full session history for stats
    current_hand: Hand          # the single hand to extract features for
    player_id: str              # villain whose range we're estimating


RANK_ORDER = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                  '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11,
                  'Q': 12, 'K': 13, 'A': 14}


class SolverRequest(BaseModel):
    board: List[str]
    hero_cards: List[str]
    ip_range: str
    oop_range: str
    position: str  # "IP" or "OOP"
    stack_depth: float
    pot_size: float
    bet_sizes: Dict[Tuple[str, str, str], float]  # (street, role, act): size

ACTIONS = [
    "fold",
    "open",
    "call",
    "3bet",
    "4bet",
    "jam",
    "iso",
    "limp"
]

STACK_BUCKETS = [10, 20, 40, 75, 100, 150, 200]  # in big blinds

VILLAIN_PROFILES = [
    "GTO",      # Balanced, solver-based baseline
    "TAG",      # Tight-aggressive
    "LAG",      # Loose-aggressive
    "NIT",      # Very tight and passive
    "MANIAC",   # Wildly aggressive and loose
    "FISH",     # Loose-passive or unpredictable
]

MULTIWAY_CONTEXTS = [
    "HU",        # Heads-up (default)
    "3WAY",      # 3 players to the flop
    "4WAY_PLUS", # 4 or more players
]

EXPLOIT_SETTINGS = [
    "GTO",              # Play default solver strategy
    "EXPLOIT_LIGHT",    # Slight deviation to exploit tendencies
    "EXPLOIT_HEAVY",    # Aggressively punish leaks
]

POPULATION_TYPES = ["RECREATIONAL", "REGULAR"]

ACTION_CONTEXTS = [
    "OPEN",
    "VS_LIMP",
    "VS_OPEN",
    "VS_ISO",
    "VS_3BET",
    "VS_4BET"
]

POSITIONS = [
    "UTG",   # Under the Gun
    "MP",    # Middle Position
    "CO",    # Cutoff
    "BTN",   # Button
    "SB",    # Small Blind
    "BB"     # Big Blind
]