from typing import Literal, Optional, List, Dict, Tuple
from pydantic import BaseModel, Field, conint, confloat

# --- Shared enums (reuse your existing ones if you’ve centralized them) ---
Street = Literal["preflop","flop","turn","river"]
Position6 = Literal["SB","BB","UTG","HJ","CO","BTN"]
ActionClass = Literal["fold","check","call","bet","raise","allin"]

# ---------- FEATURES (X) ----------
class PopNetFeatures(BaseModel):
    # Game / meta
    stake_tag: str = Field(..., description="e.g., 'NL10'")
    players: conint(ge=2, le=6) = 6
    street: Street

    # Actor identity & positions
    actor: str = Field(..., description="player id as appears in HH (for traceability)")
    actor_pos: Position6
    btn_pos: Position6 | None = None  # optional if you want it; not needed for model
    positions: Dict[str, Position6] = Field(..., description="player_id -> position")

    # Stack / pot context at decision time (all normalized to BB)
    effective_stack_bb: confloat(ge=0.0)
    pot_bb: confloat(ge=0.0)
    amount_to_call_bb: confloat(ge=0.0)  # 0 if not facing a bet/raise
    is_3bet_pot: bool = False
    is_4bet_plus: bool = False

    # Preflop open/iso flags (small, useful signals)
    is_first_in: bool = False          # actor opens the pot (no prior action)
    facing_open: bool = False
    facing_3bet: bool = False
    facing_4bet_plus: bool = False

    # Postflop board context (optional for v1 preflop-only, recommended later)
    board_cluster_id: Optional[conint(ge=0)] = None   # use your flop clusters (turn/river later)
    board_cards: Optional[List[str]] = None           # keep raw cards for debug; model won’t use them

    # Sizing grist (what sizes are already in the pot)
    last_bet_bb: Optional[confloat(ge=0.0)] = None    # size of the last bet/raise, if any
    min_raise_to_bb: Optional[confloat(ge=0.0)] = None  # “to” amount required for a legal raise

    # Optional history-lite (count these within the hand up to this decision)
    bets_this_street: conint(ge=0) = 0
    raises_this_street: conint(ge=0) = 0
    checks_this_street: conint(ge=0) = 0
    calls_this_street: conint(ge=0) = 0

    # Optional population keying (lets you slice later)
    table_name: Optional[str] = None
    stakes: Optional[str] = None  # duplicate of stake_tag if you like


# ---------- LABEL (y) ----------
class PopNetLabel(BaseModel):
    action: ActionClass                          # what the actor actually did
    amount_bb: Optional[confloat(ge=0.0)] = None # if action in {bet, raise, allin, call}
                                                 # (call/raise are “to” amounts; call==amount_to_call_bb)

    # Optional discretization for a 2-head model (classification + regression)
    size_bucket: Optional[conint(ge=0)] = None   # e.g., your own bucketing (1/3 pot, 1/2, 3/4, pot, shove, …)


# ---------- A whole training sample ----------
class PopNetSample(BaseModel):
    x: PopNetFeatures
    y: PopNetLabel