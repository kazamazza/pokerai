from dataclasses import dataclass
from enum import Enum

class TextureClass(Enum):
    DRY = "dry"
    SEMI_DRY = "semi_dry"
    WET = "wet"
    PAIR_HIGH = "pair_high"
    PAIR_LOW = "pair_low"
    TRIPLE_SUITED = "triple_suited"
    ACE_HIGH = "ace_high"
    LOW_CONNECT = "low_connect"
    UNKNOWN = "unknown"  # fallback


class Connectivity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StraightDrawPotential(Enum):
    NONE = "none"
    GUTSHOT = "gutshot"
    OPEN_ENDED = "open_ended"


@dataclass
class FlopFeatures:
    is_paired: bool
    has_ace: bool
    has_king: bool
    has_flush_draw: bool
    is_monotone: bool
    is_two_tone: bool
    high_card_count: int  # number of T/J/Q/K/A
    straight_draw_potential: StraightDrawPotential
    wetness_score: float  # 0.0 (dry) → 1.0 (wet)
    connectivity: Connectivity
    texture_class: TextureClass