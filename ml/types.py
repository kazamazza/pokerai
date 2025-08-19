import os
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any, Union, Literal, cast

from pydantic import BaseModel, field_validator, Field

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


SizeVal = Union[int, float, str, List[Union[int, float, str]], None]
BetKey = Tuple[
    Literal["flop", "turn", "river"],
    Literal["ip", "oop"],
    Literal["bet", "raise", "donk", "allin"],
]

class SolverRequest(BaseModel):
    # Core spot description
    pot_size: float
    stack_depth: float
    board: Union[List[str], str]
    ip_range: Union[str, List[str]]
    oop_range: Union[str, List[str]]

    # Who acts first on each street is inferred by the solver; keep for future use
    position: Literal["IP", "OOP"] = "IP"
    hero_cards: List[str] = Field(default_factory=list)

    # Tree config: (street, role, act) -> size
    # size is percent-of-pot as int/float/str, list of sizes, or None for "allin"
    bet_sizes: Dict[BetKey, SizeVal]

    # Solver knobs (safe defaults)
    allin_threshold: float = 0.67
    threads: int = 1
    accuracy: float = 0.5
    max_iterations: int = 200
    print_interval: int = 10
    use_isomorphism: bool = True
    dump_rounds: int = 2
    output_path: str = "output_result.json"

    # --- Normalizers / validators ---

    @field_validator("board")
    @classmethod
    def _normalize_board(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, list):
            return v
        s = str(v).replace(",", "").strip()
        if len(s) != 6:
            raise ValueError("board must be 3 cards like '2c4dQh' or ['2c','4d','Qh']")
        return [s[0:2], s[2:4], s[4:6]]

    @field_validator("ip_range", "oop_range")
    @classmethod
    def _normalize_range(cls, v: Union[str, List[str]]) -> str:
        if isinstance(v, list):
            return ",".join(v)
        return str(v)

    @field_validator("bet_sizes")
    @classmethod
    def _normalize_bet_sizes(cls, m: Dict[Tuple[Any, Any, Any], SizeVal]) -> Dict[BetKey, SizeVal]:
        def low(x: Any) -> str:
            return str(x).lower()

        norm: Dict[BetKey, SizeVal] = {}
        for k, size in m.items():
            if not isinstance(k, tuple) or len(k) != 3:
                raise ValueError(f"bet_sizes key must be a 3-tuple (street, role, act), got {k}")

            street, role, act = low(k[0]), low(k[1]), low(k[2])

            if street not in {"flop", "turn", "river"}:
                raise ValueError(f"invalid street: {street}")
            if role not in {"ip", "oop"}:
                raise ValueError(f"invalid role: {role}")
            if act not in {"bet", "raise", "donk", "allin"}:
                raise ValueError(f"invalid act: {act}")

            key = cast(BetKey, (street, role, act))  # <-- the important bit

            if act == "allin":
                norm[key] = None  # writer emits the no-size command
            else:
                norm[key] = size  # scalar/list/str; writer will percent-normalize
        return norm

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