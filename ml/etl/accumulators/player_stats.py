# etl/accumulators/player_stats.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional
import math

from ml.schema.exploit_net_schema import POP_FEAT_NAMES


class PlayerStatsAccumulator:
    """
    Tracks and accumulates per-player poker statistics (VPIP, PFR, 3Bet%, WTSD, etc.)
    with optional exponential decay for time-weighted recency.
    Designed for feeding ExploitNet features.
    """

    DEFAULT_FEATS = [
        "vpip", "pfr", "threebet", "fold_to_3bet",
        "cbet_flop", "fold_to_cbet_flop",
        "wtsd", "w$sd", "agg_factor"
    ]

    def __init__(self, decay: float = 0.97, feat_names: Optional[list[str]] = None):
        """
        Args:
            decay: Exponential decay factor applied each hand (0.97 ≈ 3% weight drop per hand).
            feat_names: Custom feature order; defaults to DEFAULT_FEATS.
        """
        self.decay = decay
        self.feat_names = feat_names or self.DEFAULT_FEATS

        # Counters (per player)
        self.counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.showdowns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # -------------------------------
    # Event ingestion
    # -------------------------------

    def decay_all(self):
        """Apply exponential decay to all players' stats."""
        for pid in self.counts:
            for k in self.counts[pid]:
                self.counts[pid][k] *= self.decay
        for pid in self.showdowns:
            for k in self.showdowns[pid]:
                self.showdowns[pid][k] *= self.decay

    def record_action(self, player_id: str, street: str, action: str, to_call: float, bet_size: float, is_first_raise: bool):
        """
        Record a single betting action.
        Args:
            player_id: str unique id
            street: 'preflop','flop','turn','river'
            action: 'fold','check','call','bet','raise','allin'
            to_call: amount to call before action (bb)
            bet_size: amount committed (bb)
            is_first_raise: True if this was the first raise (open or 3bet)
        """
        c = self.counts[player_id]

        # VPIP (voluntarily put money in pot)
        if action in ("call", "raise", "bet", "allin") and street == "preflop":
            c["vpip_hands"] += 1

        # PFR (preflop raise)
        if action in ("raise", "allin") and street == "preflop" and is_first_raise:
            c["pfr_hands"] += 1

        # 3bet
        if action in ("raise", "allin") and street == "preflop" and not is_first_raise:
            c["threebet_hands"] += 1

        # Fold to 3bet
        if action == "fold" and street == "preflop" and to_call > 0:
            c["fold_to_3bet_opps"] += 1
            if is_first_raise:  # folding to villain’s 3bet after opening
                c["fold_to_3bet"] += 1

        # C-bet flop
        if action in ("bet", "raise") and street == "flop" and is_first_raise:
            c["cbet_flop"] += 1

        # Fold to c-bet
        if action == "fold" and street == "flop" and to_call > 0:
            c["fold_to_cbet_flop"] += 1

        # Aggression factor components
        if action in ("bet", "raise", "allin"):
            c["agg_acts"] += 1
        elif action == "call":
            c["passive_acts"] += 1

        # Total hands observed
        c["hands"] += 1

    def record_showdown(self, player_id: str, won: bool):
        """Track showdown appearances and outcomes."""
        s = self.showdowns[player_id]
        s["showdowns"] += 1
        if won:
            s["won_showdowns"] += 1

    # -------------------------------
    # Feature extraction
    # -------------------------------

    def get_features(self, player_id: str) -> list[float]:
        """Return normalized feature vector in the configured order."""
        c = self.counts[player_id]
        s = self.showdowns[player_id]

        hands = max(1, c.get("hands", 0))  # avoid div0

        vpip = c.get("vpip_hands", 0) / hands
        pfr = c.get("pfr_hands", 0) / hands
        threebet = c.get("threebet_hands", 0) / max(1, c.get("hands", 0))
        fold_to_3bet = c.get("fold_to_3bet", 0) / max(1, c.get("fold_to_3bet_opps", 0))

        cbet_flop = c.get("cbet_flop", 0) / max(1, c.get("hands", 0))
        fold_to_cbet_flop = c.get("fold_to_cbet_flop", 0) / max(1, c.get("hands", 0))

        wtsd = s.get("showdowns", 0) / hands
        wsd = s.get("won_showdowns", 0) / max(1, s.get("showdowns", 0))

        agg_factor = c.get("agg_acts", 0) / max(1, c.get("passive_acts", 0))

        feats = {
            "vpip": vpip,
            "pfr": pfr,
            "threebet": threebet,
            "fold_to_3bet": fold_to_3bet,
            "cbet_flop": cbet_flop,
            "fold_to_cbet_flop": fold_to_cbet_flop,
            "wtsd": wtsd,
            "w$sd": wsd,
            "agg_factor": math.tanh(agg_factor / 5.0),  # clamp for stability
        }

        return [float(feats.get(k, 0.0)) for k in self.feat_names]

    def get_all_features(self) -> Dict[str, list[float]]:
        """Return normalized feature vectors for all players."""
        return {pid: self.get_features(pid) for pid in self.counts.keys()}

    def infer_profile(self, player_id: str) -> str:
        """
        Map accumulated stats → coarse profile name.
        Profiles: NIT, TAG, LAG, MANIAC, FISH (fallback = "UNKNOWN").
        """
        feats_vec = self.get_features(player_id)
        if not feats_vec:
            return "UNKNOWN"

        # zip vector → dict
        feats = dict(zip(POP_FEAT_NAMES, feats_vec))

        vpip = feats.get("vpip", 0.0)
        pfr = feats.get("pfr", 0.0)
        threebet = feats.get("threebet", 0.0)
        agg_factor = feats.get("agg_factor", 1.0)

        # --- heuristics ---
        if vpip < 0.15 and pfr < 0.12:
            return "NIT"
        if 0.15 <= vpip <= 0.28 and 0.12 <= pfr <= 0.22:
            return "TAG"
        if vpip > 0.28 and pfr >= 0.20 and threebet >= 0.06:
            return "LAG"
        if vpip > 0.35 and agg_factor > 2.5:
            return "MANIAC"
        if vpip > 0.40 and pfr < 0.12:
            return "FISH"

        return "UNKNOWN"