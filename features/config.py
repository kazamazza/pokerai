from dataclasses import dataclass, field
from typing import List


@dataclass
class RangeFeatureConfig:
    """
    Lists of feature keys for your Range-Net.

    - foundation: the core handful you need up front
    - situational: high-value context features
    - supplemental: nice-to-have extras
    """
    foundation: List[str] = field(default_factory=lambda: [
        # Basic player profile
        "vpip_pct",
        "pfr_pct",
        "three_bet_pct",
        # Flop betting tendencies
        "flop_cbet_pct",
        "fold_to_flop_cbet_pct",
        # Table context
        "position_onehot",        # 6-dim
        "last4_actions_onehot",   # 4 actions × 4 verbs = 16-dim
        # Board texture
        "board_paired",
        "board_connected",
        "board_uncoordinated",
        "board_monotone",
        "board_two_tone",
        "board_rainbow",
    ])

    situational: List[str] = field(default_factory=lambda: [
        # Stack / pot dynamics
        "exact_spr",  # float
        "pot_size",  # float
        "num_opponents",  # int

        # Turn betting tendencies
        "turn_cbet_pct",  # float
        "fold_to_turn_cbet_pct",  # float
        "aggression_factor",  # float  (bets+raises)/calls

        # New situational features
        "ip_vs_last_raiser",  # bool: in-position vs. last raiser
        "did_flop_cbet",  # bool: did this player c-bet the flop in this hand
        "turn_bet_size_pct",  # float: avg turn bet size as % of pot
        "pot_odds_pct",  # float: current pot odds offered
        "spr_bin",  # int: bucketed SPR (e.g. SPR<2,2–4,4–8,8+)
        "street_onehot",  # one-hot: which street (flop, turn, river)
        "actions_since_last_aggression",  # int: number of actions since their last bet/raise
        "prev_showdown_tendency",  # float: did they go to showdown last time they faced a turn c-bet
    ])

    supplemental: List[str] = field(default_factory=lambda: [
        # Showdown habits
        "wtsd_pct",           # Went To Showdown %
        "wsd_pct",            # Won Showdown %
        # Odds & dead money
        "implied_odds",       # float
        "dead_money",         # float
    ])

    @property
    def all_features(self) -> List[str]:
        """Flattened list of every feature key, in order."""
        return self.foundation + self.situational + self.supplemental


# Example usage:
if __name__ == "__main__":
    cfg = RangeFeatureConfig()
    print("Foundation features:", cfg.foundation)
    print("Situational features:", cfg.situational)
    print("Supplemental features:", cfg.supplemental)
    print("Total features:", len(cfg.all_features))