from typing import List, Dict, Optional, Literal
from pydantic import BaseModel

# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class RawSeat(BaseModel):
    seat_number: int
    player_id: str
    stack_size: float
    status: Optional[Literal['active', 'sitting out']] = 'active'


class ActionEvent(BaseModel):
    street: Literal['preflop','flop','turn','river']
    actor: str
    act: Literal['fold','check','call','bet','raise','allin']
    amount_bb: Optional[float] = None   # unified normalized amount in BB


class HandSchema(BaseModel):
    hand_id: str
    table_name: Optional[str]
    stake_tag: str
    button_seat: Optional[int]
    seats: List[RawSeat]
    street: Literal['preflop','flop','turn','river']
    hero: Optional[str]
    hole_cards_by_player: Dict[str, List[str]]
    board: List[str]

    # CHANGED: actions are structured
    actions: List[ActionEvent]

    players: List[str]

    # NEW fields (since parser now emits them)
    position_by_player: Dict[str, Literal['SB','BB','UTG','HJ','CO','BTN']] = {}
    results_by_player: Dict[str, float] = {}   # net won in BB by player id
    pot_bb: float = 0.0
    rake_bb: float = 0.0

    min_bet: float
    stakes: str