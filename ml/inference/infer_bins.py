from pathlib import Path
from typing import Optional, Dict, Any
from ml.core.buckets import bucketize_spr, bucketize_rate
from ml.core.io import load_bins_sidecar


class ExploitBucketizer:
    def __init__(self, bins_source: str | Path):
        meta = load_bins_sidecar(Path(bins_source))
        self.spr_bins = meta["spr_bins"]
        self.rate_bins = meta["rate_bins"]

    def to_feature_row(
        self,
        *,
        stakes_id: int,
        street: int,
        villain_pos: str,
        spr: float,
        vpip: Optional[float],
        pfr: Optional[float],
        three_bet: Optional[float],
    ) -> Dict[str, Any]:
        """
        Map live numeric stats → categorical bins used by ExploitNet.
        Returns the X-feature dict expected by the dataset/model.
        """
        return {
            "stakes_id": int(stakes_id),
            "street": int(street),
            "villain_pos": str(villain_pos),
            "spr_bin": bucketize_spr(float(spr), spr_bins=self.spr_bins),
            "vpip_bin": bucketize_rate(vpip, rate_bins=self.rate_bins),
            "pfr_bin": bucketize_rate(pfr, rate_bins=self.rate_bins),
            "three_bet_bin": bucketize_rate(three_bet, rate_bins=self.rate_bins),
        }