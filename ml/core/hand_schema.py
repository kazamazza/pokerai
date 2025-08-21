from typing import List, Dict, Optional, Literal
from pydantic import BaseModel

class PosObj(BaseModel):
    id: int
    name: str

class StakeObj(BaseModel):
    id: int
    name: str

class RawSeat(BaseModel):
    seat_number: int
    player_id: str
    stack_size: float
    status: Optional[Literal['active', 'sitting out']] = 'active'

class ActionEvent(BaseModel):
    street: int
    actor: str
    act: int
    amount_bb: Optional[float] = None   # unified normalized amount in BB

class HandSchema(BaseModel):
    hand_id: str
    table_name: Optional[str]
    button_seat: Optional[int]
    seats: List[RawSeat]
    street: int
    hero: Optional[str]
    hole_cards_by_player: Dict[str, List[str]]
    board: List[str]
    actions: List[ActionEvent]
    players: List[str]
    position_by_player: Dict[str, PosObj] = {}
    results_by_player: Dict[str, float] = {}
    pot_bb: float = 0.0
    rake_bb: float = 0.0
    min_bet: float
    stakes: StakeObj