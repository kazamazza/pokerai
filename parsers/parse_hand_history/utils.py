from typing import List, Dict, Optional, Literal
from pydantic import BaseModel

# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class RawSeat(BaseModel):
    seat_number: int
    player_id: str
    stack_size: float
    status: Optional[Literal['active','sitting out']] = 'active'

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
    actions: List[str]
    players: List[str]
    winnings: List[dict]
    min_bet: float
    stakes: str