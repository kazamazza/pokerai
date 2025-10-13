from enum import IntEnum
from typing import Dict, Any

class Stakes(IntEnum):
    NL2=0; NL5=1; NL10=2; NL25=3  # extend later

STAKE_CFG: Dict[Stakes, Dict[str, Any]] = {
    Stakes.NL10: {
        # Preflop size model (x = raise-to in BB)
        "open_x":     {"UTG":3.0, "HJ":2.5, "CO":2.5, "BTN":2.5, "SB":3.0},
        "threebet_x": {"IP":7.5,  "OOP":9.0},
        "fourbet_x":  24.0,

        # Flop menus (fractions of pot) and raise ladders (to-multiple of facing bet)
        "bet_menus": {
            "srp_hu.PFR_IP":         [0.33, 0.66],
            "srp_hu.Caller_OOP":     [0.33],
            "3bet_hu.Aggressor_IP":  [0.33, 0.66],
            "3bet_hu.Aggressor_OOP": [0.33, 0.66],
            "4bet_hu.Aggressor_IP":  [0.33],
            "4bet_hu.Aggressor_OOP": [0.33],
            "limped_single.SB_IP":   [0.33],
            "limped_multi.Any":      [0.33],
        },
        "raise_mult": [1.5, 2.0, 3.0],
        "flop_allin": True,

        # Optional stake tuning for pot calc
        "pot_adj": {"srp":1.00, "threebet":1.00, "fourbet":1.00},
    },

    Stakes.NL25: {
        "open_x":     {"UTG":2.5, "HJ":2.3, "CO":2.3, "BTN":2.2, "SB":2.5},
        "threebet_x": {"IP":7.0,  "OOP":8.5},
        "fourbet_x":  22.0,

        "bet_menus": {
            "srp_hu.PFR_IP":         [0.33, 0.66, 1.00],
            "srp_hu.Caller_OOP":     [0.33],
            "3bet_hu.Aggressor_IP":  [0.33, 0.66],
            "3bet_hu.Aggressor_OOP": [0.33, 0.66],
            "4bet_hu.Aggressor_IP":  [0.33],
            "4bet_hu.Aggressor_OOP": [0.33],
            "limped_single.SB_IP":   [0.33],
            "limped_multi.Any":      [0.33],
        },
        "raise_mult": [1.5, 2.0, 3.0, 4.0],
        "flop_allin": True,

        "pot_adj": {"srp":0.95, "threebet":0.95, "fourbet":0.90},
    },
}