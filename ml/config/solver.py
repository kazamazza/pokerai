from typing import Dict, Any

from ml.core.types import Stakes

STAKE_CFG: Dict[Stakes, Dict[str, Any]] = {
Stakes.NL10: {
  "board_clusters": {"flop": 64, "turn": 128, "river": 128},

  "open_x":     {"UTG": 2.5, "HJ": 2.5, "CO": 2.5, "BTN": 2.5, "SB": 3.0},
  "threebet_x": {"IP": 7.5, "OOP": 9.0},
  "fourbet_x":  24.0,

  "bet_menus": {
    # SRP
    "srp_hu.PFR_IP":         {"flop":[0.25,0.50,0.75], "turn":[0.50,1.25], "river":[0.75,1.25]},
    "srp_hu.Caller_OOP":     {"flop":[0.33],          "turn":[0.50],      "river":[0.75]},

    # Blind vs steal
    "bvs.Any":               {"flop":[0.25,0.66],     "turn":[0.66,1.25],  "river":[0.75,1.25]},

    # 3bet pots
    "3bet_hu.Aggressor_IP":  {"flop":[0.25,0.66],     "turn":[0.66,1.25],  "river":[0.75,1.25]},
    "3bet_hu.Aggressor_OOP": {"flop":[0.25,0.66],     "turn":[0.66,1.25],  "river":[0.75,1.25]},

    # 4bet pots
    "4bet_hu.Aggressor_IP":  {"flop":[0.33],          "turn":[0.50],       "river":[0.50]},
    "4bet_hu.Aggressor_OOP": {"flop":[0.33],          "turn":[0.50],       "river":[0.50]},

    # Limped
    "limped_single.BB_IP":   {"flop":[0.33],          "turn":[0.50],       "river":[0.75]},
    "limped_multi.Any":      {"flop":[0.33],          "turn":[0.50],       "river":[0.75]},
  },

  "raise_mult": [2.0, 3.0, 4.5],
  "allin_gate_spr": 2.5,

  "pot_adj": {"srp": 1.00, "threebet": 1.00, "fourbet": 1.00},
  "stacks_by_ctx": {
    "VS_OPEN":        [25, 60, 100, 150],
    "BLIND_VS_STEAL": [25, 60, 100, 150],
    "VS_3BET":        [25, 60, 100, 150],
    "VS_4BET":        [60, 100, 150],
    "LIMPED_SINGLE":  [25, 60, 100],
    "LIMPED_MULTI":   [25, 60, 100],
  }
},
    Stakes.NL25: {
        "board_clusters": 128,
        "open_x":     {"UTG": 2.5, "HJ": 2.3, "CO": 2.3, "BTN": 2.2, "SB": 2.5},
        "threebet_x": {"IP": 7.0,  "OOP": 8.5},
        "fourbet_x":  22.0,
        "bet_menus": {
            "srp_hu.PFR_IP":         [0.33, 0.66, 1.00],  # allow 100% at deep
            "srp_hu.Caller_OOP":     [0.33],
            "bvs.Any":               [0.33, 0.66],
            "3bet_hu.Aggressor_IP":  [0.33, 0.66],
            "3bet_hu.Aggressor_OOP": [0.33, 0.66],
            "4bet_hu.Aggressor_IP":  [0.33],
            "4bet_hu.Aggressor_OOP": [0.33],
            "limped_single.SB_IP":   [0.33],
            "limped_multi.Any":      [0.33],
        },
        "raise_mult": [1.5, 2.0, 3.0, 4.0],
        "flop_allin": True,
        "pot_adj": {"srp": 0.95, "threebet": 0.95, "fourbet": 0.90},
        "stacks_by_ctx": {
            "VS_OPEN":        [25, 60, 100, 150],
            "BLIND_VS_STEAL": [25, 60, 100, 150],
            "VS_3BET":        [25, 60, 100, 150],
            "VS_4BET":        [60, 100, 150],
            "LIMPED_SINGLE":  [25, 60, 100],
            "LIMPED_MULTI":   [25, 60, 100],
        },
    },
}