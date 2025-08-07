import hashlib
import json

from pydantic import BaseModel, Field


class EquityNetFeatures(BaseModel):
    hero_hand: list[str]                 # e.g., "AhKs"
    board: list[str]              # e.g., ["8h", "7d", "2s"]
    position: str                 # e.g., "BTN", "BB"
    stack_bb: int                 # e.g., 40
    pot_size: float               # e.g., 15.5
    num_players: int              # e.g., 2
    has_initiative: bool          # Hero was last aggressor?

    def hash(self) -> str:
        key_data = {
            "hero_hand": self.hero_hand,
            "board": self.board,
            "position": self.position,
            "stack_bb": self.stack_bb,
            "pot_size": self.pot_size,
            "num_players": self.num_players,
            "has_initiative": self.has_initiative,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha1(key_str.encode()).hexdigest()

class EquityNetLabel(BaseModel):
    raw_equity: float             # e.g., 0.72
    normalized_equity: float     # e.g., 0.91