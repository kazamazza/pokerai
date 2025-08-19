from pydantic import BaseModel, conlist, Field
from typing import Optional, Literal, Tuple, List


# ---- Inputs (features) ----
class EquityNetFeatures(BaseModel):
    # Hand / street
    hand_id: int = Field(..., ge=0, le=168)
    street: Literal["flop", "turn", "river"] = "flop"

    # Board representation
    board_cluster_id: int = Field(..., ge=0)  # keep strict: must be 0..K-1

    # Optional: allow None or any short vector (we're not using it yet)
    board_feats: Optional[List[float]] = None  # ← no min_length; None is fine

    # Opponent range (choose ONE representation)
    opp_range_bucket_id: Optional[int] = Field(None, ge=0)

    # For now accept full 169-d vector (quick path). Later we can switch to PCA-16/32.
    opp_range_emb: Optional[conlist(float, min_length=169, max_length=169)] = None

    players: int = Field(2, ge=2, le=2)

    def key(self) -> Tuple[int, int, Optional[int], Optional[Tuple[float, ...]]]:
        emb = tuple(self.opp_range_emb) if self.opp_range_emb is not None else None
        return (self.hand_id, self.board_cluster_id, self.opp_range_bucket_id, emb)

    @property
    def has_valid_opp(self) -> bool:
        return (self.opp_range_bucket_id is not None) ^ (self.opp_range_emb is not None)

    @property
    def has_valid_opp(self) -> bool:
        return (self.opp_range_bucket_id is not None) ^ (self.opp_range_emb is not None)


# ---- Label (target) ----
class EquityNetLabel(BaseModel):
    """
    Ground-truth equity label.
    - equity ∈ [0,1] is required.
    - Store n_samples/var when using Monte Carlo (helps training weight/noise handling).
    """
    equity: float = Field(..., ge=0.0, le=1.0)                   # mean equity vs given opponent range
    n_samples: Optional[int] = Field(None, ge=1)                  # MC trials used to produce the label
    variance: Optional[float] = Field(None, ge=0.0)               # optional: per-label variance from MC