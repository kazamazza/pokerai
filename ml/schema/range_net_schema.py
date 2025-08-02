from pydantic import BaseModel, Field
from typing import List, Literal


class RangeNetFeatures(BaseModel):
    # Contextual features (typically one-hot encoded)
    cluster_id: int = Field(..., ge=0, le=127)
    ip_position: str
    oop_position: str
    stack_bb: int
    action_context: str
    multiway_context: str
    population_type: str
    villain_profile: str
    exploit_setting: str

    # Flop texture features (normalized float values)
    is_paired: float
    is_triplet: float
    is_monotone: float
    is_two_tone: float
    high_rank: float
    low_rank: float
    gap1: float
    gap2: float


class RangeNetLabel(BaseModel):
    # Action label to be learned (e.g., from GTO strategy)
    action: Literal["FOLD", "CALL", "RAISE"]