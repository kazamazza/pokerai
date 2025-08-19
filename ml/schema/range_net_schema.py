from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, conlist, conint, confloat, model_validator

# -------- enums (align with the rest of your codebase) --------
Street     = Literal["preflop","flop","turn","river"]  # kept for symmetry, (not used here)
Position6  = Literal["SB","BB","UTG","HJ","CO","BTN"]
ActionCtx  = Literal["OPEN","VS_OPEN","VS_3BET","VS_4BET","VS_LIMP","VS_ISO","VS_SQUEEZE"]
MultiwayCtx= Literal["HU","3WAY","4WAY_PLUS"]

# -------- FEATURES (X) --------
class RangeNetFeatures(BaseModel):
    """
    One row describes a *preflop node* + a specific hand bucket (169 grid) or combo.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Meta / routing
    version: str = "rangenet.v1"
    stake_tag: str = Field(..., description="e.g. 'NL10'")
    players: conint(ge=2, le=6) = 6

    # Game economy
    rake_tier: Optional[str] = Field(None, description="e.g. 'MICRO/HIGH'")
    ante_bb: confloat(ge=0.0) = 0.0
    open_size_policy: Optional[str] = Field(
        None, description="e.g. 'SMALL_2_2' or 'STD_2_5' from settings"
    )

    # Table geometry (preflop only)
    hero_pos: Position6
    btn_pos: Optional[Position6] = None
    positions: Optional[Dict[str, Position6]] = None  # optional in dev

    # Spot definition
    ctx: ActionCtx                                   # <-- canonical name
    multiway: MultiwayCtx = "HU"
    effective_stack_bb: confloat(ge=0.0)             # <-- canonical name
    pot_bb: confloat(ge=0.0) = 0.0
    amount_to_call_bb: confloat(ge=0.0) = 0.0
    last_raise_to_bb: Optional[confloat(ge=0.0)] = None
    min_raise_to_bb: Optional[confloat(ge=0.0)] = None

    # Opponent roles (optional depending on ctx)
    opener_pos: Optional[Position6] = None
    three_bettor_pos: Optional[Position6] = Field(None, alias="threebettor_pos")
    four_bettor_pos: Optional[Position6]  = Field(None, alias="fourbettor_pos")
    cold_callers: conint(ge=0) = 0
    squeezers: conint(ge=0) = 0

    # Specific hand
    hand_bucket: str = Field(..., description="169 grid label: 'AKs', 'A5s', 'KQo', ...")
    hand_combo: Optional[str] = Field(None, description="1326 combo (e.g. 'AsKh'); optional")

    # ---------- compatibility shim ----------
    @model_validator(mode="before")
    def _coerce_vendor_meta(cls, data: dict):
        """
        Accept exporter/vendored keys and coerce into canonical names.
        This lets your current exporter run without exploding.
        """
        if not isinstance(data, dict):
            return data

        # seats -> players
        if "seats" in data and "players" not in data:
            data["players"] = data["seats"]

        # action_ctx -> ctx
        if "action_ctx" in data and "ctx" not in data:
            data["ctx"] = data["action_ctx"]

        # stack_bb -> effective_stack_bb (preflop effective ≈ stack depth in vendor charts)
        if "stack_bb" in data and "effective_stack_bb" not in data:
            data["effective_stack_bb"] = data["stack_bb"]

        # threebettor_pos/fourbettor_pos already handled via Field alias,
        # but accept camel variants too just in case:
        if "threeBettor_pos" in data and "three_bettor_pos" not in data:
            data["three_bettor_pos"] = data["threeBettor_pos"]
        if "fourBettor_pos" in data and "four_bettor_pos" not in data:
            data["four_bettor_pos"] = data["fourBettor_pos"]

        # default positions map if missing (identity map of 6-max)
        if not data.get("positions"):
            data["positions"] = {p: p for p in ["UTG","HJ","CO","BTN","SB","BB"]}

        return data


# -------- LABEL (y) --------
class RangeNetLabel(BaseModel):
    """
    Supervision for the preflop node. Minimal v1 expects action_probs.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Distribution over actions for this hand in this node:
    # [fold, call, raise_bucket_0, ..., raise_bucket_{k-1}]
    action_probs: conlist(float, min_length=2, max_length=16) = Field(
        ..., description="normalized probabilities over [fold, call, raise_buckets...]"
    )

    # Optional regression targets
    exp_raise_bb: Optional[confloat(ge=0.0)] = None   # expected raise size (if raises happen)

    # Optional EVs (if available from solver/vendor export)
    ev_bb: Optional[float] = None


class RangeNetSample(BaseModel):
    x: RangeNetFeatures
    y: RangeNetLabel