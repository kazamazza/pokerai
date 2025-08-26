# api/schemas.py
from __future__ import annotations
from typing import Literal, Optional, Dict, List
from pydantic import BaseModel, Field, root_validator, validator

# ---- little helpers ----

_STREET_TO_INT = {
    "preflop": 0,
    "flop":    1,
    "turn":    2,
    "river":   3,
}

def _norm_street(x: int | str) -> int:
    if isinstance(x, int):
        if x not in (0, 1, 2, 3):
            raise ValueError("street int must be one of 0,1,2,3")
        return x
    x = str(x).strip().lower()
    if x in _STREET_TO_INT:
        return _STREET_TO_INT[x]
    raise ValueError("street must be one of {0,1,2,3,'preflop','flop','turn','river'}")


# ---- containers for structured fields ----

class VillainProfile(BaseModel):
    # all optional; ExploitNet can decide which it needs
    vpip: Optional[float] = None
    pfr: Optional[float] = None
    three_bet: Optional[float] = None
    fold_to_cbet: Optional[float] = None
    fold_to_3bet: Optional[float] = None
    wwsf: Optional[float] = None
    hands: Optional[int] = None  # sample size

class ActionEvent(BaseModel):
    # e.g. {"actor":"BTN","street":"flop","name":"BET","size_bb":10}
    actor: Literal["UTG","HJ","CO","BTN","SB","BB","HERO","VILLAIN"]
    street: Literal["preflop","flop","turn","river",0,1,2,3]
    name: str                     # "OPEN","CALL","3BET","BET","CHECK","RAISE","FOLD",...
    size_bb: Optional[float] = None

    @validator("street", pre=True)
    def _v_street(cls, v):
        return _norm_street(v)

class ExploitFeatures(BaseModel):
    # If you pre-bucket these externally, pass IDs directly.
    spr_bin: Optional[int] = None
    vpip_bin: Optional[int] = None
    pfr_bin: Optional[int] = None
    three_bet_bin: Optional[int] = None
    # Or pass continuous features and let ExploitNetInfer bucket them:
    spr: Optional[float] = None


# ---- main request ----

class PolicyRequest(BaseModel):
    # Core context
    stakes_id: int = Field(..., description="Internal stakes/context enum id for PopulationNet")
    ctx_id: Optional[int] = Field(10, description="Optional context id used by PopulationNet")
    street: int | Literal["preflop","flop","turn","river"] = Field(...)

    # Positions & state
    hero_pos: Literal["UTG","HJ","CO","BTN","SB","BB"]
    villain_pos: Literal["UTG","HJ","CO","BTN","SB","BB"]
    stack_bb: float = Field(..., description="Effective stack in BB")
    pot_bb: float = Field(..., description="Current pot in BB")

    # Cards
    board: Optional[str] = Field(None, description="e.g. 'QsJh2h' (no spaces/commas)")
    hero_hand: Optional[str] = Field(None, description="e.g. 'AhKh' – required to use EquityNet")

    # Preflop-only extras for Range/Equity
    opener_pos: Optional[Literal["UTG","HJ","CO","BTN","SB","BB"]] = None
    opener_action: Optional[str] = Field(None, description="e.g. 'OPEN','3BET','4BET','LIMP'")

    # Profiles & history
    villain_profile: Optional[VillainProfile] = None
    last_actions: List[ActionEvent] = Field(default_factory=list)

    # Exploit preprocessing (optional). If omitted, ExploitNetInfer can derive/bucket.
    exploit_features: Optional[ExploitFeatures] = None

    # ---- validators / normalizers ----

    @validator("street", pre=True)
    def _normalize_street(cls, v):
        return _norm_street(v)

    @validator("board")
    def _normalize_board(cls, v):
        if v is None:
            return v
        s = str(v).replace(",", "").replace(" ", "")
        if s and (len(s) not in (0, 6, 8, 10)):  # flop=6, turn=8, river=10 chars
            # allow but warn by raising ValueError if you want to be strict
            return s
        return s

    @root_validator
    def _check_by_street(cls, values):
        street = values.get("street")
        board = values.get("board")
        opener_pos = values.get("opener_pos")
        opener_action = values.get("opener_action")
        hero_hand = values.get("hero_hand")

        # Preflop: opener_* help Range/Equity preflop variants
        if street == 0:
            # not strictly required, but recommended
            if opener_pos is None:
                # you can relax this if your preflop models don't require it
                values.setdefault("opener_pos", values.get("villain_pos"))
            if opener_action is None:
                values.setdefault("opener_action", "OPEN")
        else:
            # Postflop must have a board
            if not board:
                raise ValueError("Postflop requests require 'board' like 'QsJh2h'")

        # EquityNet needs hero_hand; if missing, PolicyInfer should downweight equity
        values["_has_hero_hand"] = hero_hand is not None
        return values

    # Convenience: emit a dict exactly as PolicyInfer expects today
    def to_infer_dict(self) -> dict:
        return {
            "stakes_id": self.stakes_id,
            "ctx_id": self.ctx_id,
            "street": int(self.street),
            "hero_pos": self.hero_pos,
            "villain_pos": self.villain_pos,
            "stack_bb": float(self.stack_bb),
            "pot_bb": float(self.pot_bb),
            "board": self.board or "",
            "hero_hand": self.hero_hand or "",
            "opener_pos": self.opener_pos or "",
            "opener_action": self.opener_action or "",
            "villain_profile": (self.villain_profile.dict() if self.villain_profile else {}),
            "last_actions": [a.dict() for a in self.last_actions],
            "exploit_features": (self.exploit_features.dict() if self.exploit_features else None),
        }