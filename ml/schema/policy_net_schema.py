from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, conint, confloat, conlist

class PolicyNetFeatures(BaseModel):
    # Meta / routing
    version: str = "policynet.v1"
    stake_tag: str = Field(..., description="e.g. NL10, NL25")
    players: conint(ge=2, le=6) = 6
    rake_tier: Optional[str] = Field(None, description="e.g. MICRO/HIGH")

    # Game state (node)
    street: List[str]
    hero_pos: List[str]
    btn_pos: Optional[List[str]] = None
    positions: Optional[Dict[str, List[str]]] = None

    # Money geometry
    effective_stack_bb: confloat(ge=0.0)
    pot_bb: confloat(ge=0.0)
    amount_to_call_bb: confloat(ge=0.0) = 0.0
    last_bet_to_bb: Optional[confloat(ge=0.0)] = None
    spr: Optional[confloat(ge=0.0)] = None

    # Hand & board representation (choose dev/prod granularity in ETL)
    # Dev-friendly: hand_bucket_169 + board_cluster_k
    hand_bucket_169: str = Field(..., description="AKs/AQo/... 169-grid key")
    board_cluster_id: Optional[int] = Field(None, description="cluster index for flop/turn/river nodes")
    # Optional raw board for analysis/debug (not required for training)
    board_cards: Optional[str] = Field(None, description="e.g. 'Ah7d2c|Ts|5h'")

    # Action head description (so label length is deterministic)
    action_vocab: List[str] = Field(
        ...,
        description="Ordered names for logits/probs (e.g. ['fold','check_call','b_sz0','b_sz1','b_sz2'])"
    )
    size_bucket_edges: Optional[List[float]] = Field(
        None, description="BB or pot-fraction edges used to define size buckets for this node"
    )

    # Optional model-side priors/features (plug-ins from other nets)
    equity_feats: Optional[Dict[str, float]] = None        # e.g., E[eq], eq_vs_call, draw flags...
    pop_feats_by_player: Optional[Dict[str, List[float]]] = None
    exploit_feats_by_player: Optional[Dict[str, List[float]]] = None
    range_summaries: Optional[Dict[str, List[float]]] = None  # e.g., villain range shape stats

    # Legality (trainer uses this to mask targets/loss on illegal actions)
    legal_mask: Optional[List[int]] = Field(
        None, description="Same length as action_vocab; 1=legal, 0=illegal"
    )


class PolicyNetLabel(BaseModel):
    # Primary training target: distribution over action head
    action_probs: conlist(float, min_length=2, max_length=64) = Field(
        ..., description="Normalized probabilities aligned with x.action_vocab"
    )

    # Optional auxiliaries (help with stability / diagnostics)
    exp_size_bb: Optional[confloat(ge=0.0)] = Field(
        None, description="Conditional expected size (BB) given an aggressive action"
    )
    ev_bb: Optional[float] = None                       # Node EV (optional if solver export provides)
    cf_values_bb: Optional[Dict[str, float]] = None     # Optional counterfactuals (per action), for analysis only

    # Optional label-side legality (ETL can copy x.legal_mask here for redundancy)
    legal_mask: Optional[List[int]] = None


class PolicyNetSample(BaseModel):
    x: PolicyNetFeatures
    y: PolicyNetLabel