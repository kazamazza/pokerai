from pydantic import BaseModel, Field
from typing import Literal


class PopulationNetFeatures(BaseModel):
    """
    Input features for population-level strategy tendencies, aggregated from hand histories.
    """
    stake_level: Literal["NL2", "NL5", "NL10", "NL25", "NL50", "NL100", "NL200"]
    action_context: Literal[
        "OPEN", "VS_LIMP", "VS_OPEN", "VS_ISO", "VS_3BET", "VS_4BET"
    ]
    position: Literal["UTG", "MP", "CO", "BTN", "SB", "BB"]
    player_count: int = Field(..., ge=2, le=6)


class PopulationNetLabel(BaseModel):
    """
    Output label: aggregate statistics from the player pool for a given context.
    Values represent average tendencies at the population level.
    """
    fold_pct: float = Field(..., ge=0.0, le=1.0)
    call_pct: float = Field(..., ge=0.0, le=1.0)
    raise_pct: float = Field(..., ge=0.0, le=1.0)
    avg_bet_sizing: float = Field(..., ge=0.0, description="Average bet sizing relative to pot")