from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict

# ── Enums ─────────────────────────────────────────────────────────────────
Street = Literal["PREFLOP", "FLOP", "TURN", "RIVER"]
Position = Literal["UTG", "MP", "CO", "BTN", "SB", "BB"]
ActionType = Literal[
    # atomic actions (raw):
    "FOLD", "CHECK", "CALL", "BET", "RAISE", "ALLIN",
    # preflop qualifiers (appear in history; we still normalize server-side):
    "OPEN", "LIMP", "ISO", "THREE_BET", "FOUR_BET"
]
Site = Literal["POKERSTARS", "GG", "PARTY", "OTHER"]
StakeLevel = Literal["NL2", "NL5", "NL10", "NL25", "NL50", "NL100", "NL200"]

# ── Core Table Structures ────────────────────────────────────────────────
@dataclass
class Seat:
    player_id: str
    position: Position
    stack_bb: float                 # stack in big blinds at decision time
    is_hero: bool = False
    sitting_out: bool = False

@dataclass
class BlindsAntes:
    sb_bb: float                    # small blind in bb (usually 0.5)
    bb_bb: float                    # big blind in bb (usually 1.0)
    ante_bb: float = 0.0            # total per-player ante in bb (0 if none)

@dataclass
class ActionEvent:
    street: Street
    player_id: str
    action: ActionType              # raw label (FOLD/CALL/BET/RAISE/etc.)
    amount_bb: Optional[float] = None   # amount put in with this atomic action (bb)
    to_total_bb: Optional[float] = None # player's total invested after this action (bb)
    raw_text: Optional[str] = None      # original line if you have it

# ── “Who’s in” & Matchup Snapshot (static, not derived client-side) ───────
@dataclass
class StreetActors:
    street: Street
    active_player_ids: List[str]           # currently not folded
    to_act_player_id: str                  # who must act now
    last_aggressor_player_id: Optional[str] = None  # last bettor/raiser on this street (initiative)

@dataclass
class MatchupHint:
    """
    Optional confidence helper. We verify server-side. Pass if easy for you.
    If HU, include ip_position / oop_position; otherwise list opponent_positions.
    """
    hero_position: Position
    opponent_positions: List[Position]             # remaining villains
    ip_position: Optional[Position] = None         # if heads-up & known
    oop_position: Optional[Position] = None        # if heads-up & known

# ── Street State (static snapshot) ────────────────────────────────────────
@dataclass
class StreetState:
    street: Street
    board: List[str] = field(default_factory=list)  # e.g. ["Jh","5c","3c"] (0–5 cards depending on street)
    pot_bb: Optional[float] = None                  # total pot in bb at decision time (optional)
    bet_to_call_bb: Optional[float] = None         # amount hero must call now in bb (optional)
    min_raise_to_bb: Optional[float] = None        # minimum total raise-to size in bb (optional)

# ── Optional live stats (for ExploitNet) ──────────────────────────────────
@dataclass
class VillainLiveStats:
    player_id: str
    hands_sampled: int
    vpip: Optional[float] = None
    pfr: Optional[float] = None
    three_bet: Optional[float] = None
    flop_cbet: Optional[float] = None
    fold_to_cbet: Optional[float] = None
    wtsd: Optional[float] = None
    wsd: Optional[float] = None

# ── Population context (for PopulationNet) ────────────────────────────────
@dataclass
class PopulationContext:
    site: Site
    stake_level: StakeLevel

# ── Request Wrapper ──────────────────────────────────────────────────────
@dataclass
class PolicyRequest:
    # Identity
    table_id: Optional[str]
    hand_id: Optional[str]

    # Stakes & seats
    blinds: BlindsAntes
    button_position: Position
    seats: List[Seat]

    # Current decision context
    state: StreetState
    actors: StreetActors
    history: List[ActionEvent]           # full action history so far (all streets)

    # Optional clarity (not trusted blindly)
    matchup_hint: Optional[MatchupHint] = None

    # Optional inputs for exploit & population priors
    population: Optional[PopulationContext] = None
    live_stats: Optional[List[VillainLiveStats]] = None

    # (Optional) legal action constraints if your client tracks them
    legal_actions: Optional[Dict[str, List[float]]] = None  # e.g. {"BET":[0.33,0.5,0.75], "RAISE_TO":[2.2,3.0]}

    # Hero hand is required for EV/policy; if you do not want to send preflop, send when revealed
    hero_hand: Optional[List[str]] = None  # e.g. ["Ah","Ks"]
    timestamp_iso: Optional[str] = None  # ISO 8601 if you have it (server will accept None)