from pydantic import BaseModel, Field


class EquityNetFeatures(BaseModel):
    hero_hand: list[str]                 # e.g., "AhKs"
    board: list[str]              # e.g., ["8h", "7d", "2s"]
    position: str                 # e.g., "BTN", "BB"
    stack_bb: int                 # e.g., 40
    pot_size: float               # e.g., 15.5
    num_players: int              # e.g., 2
    has_initiative: bool          # Hero was last aggressor?


class EquityNetLabel(BaseModel):
    raw_equity: float             # e.g., 0.72
    normalized_equity: float     # e.g., 0.91