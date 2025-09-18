# tools/rangenet/etl/_solver_profiles.py  (new small module)

from typing import Dict

# Per-context solver settings (tuned to cap file size/runtime).
# You can tweak these as you observe sizes; start conservative.
SOLVER_PROFILES: Dict[str, Dict] = {
    # --- SRP ---
    "srp_hu.PFR_IP":     {"accuracy": 0.015, "max_iter": 6000, "allin_threshold": 0.67},
    "srp_hu.PFR_OOP":    {"accuracy": 0.015, "max_iter": 6000, "allin_threshold": 0.67},
    "srp_hu.Caller_IP":  {"accuracy": 0.020, "max_iter": 5000, "allin_threshold": 0.67},
    "srp_hu.Caller_OOP": {"accuracy": 0.020, "max_iter": 5000, "allin_threshold": 0.67},

    # --- 3-bet ---
    "3bet_hu.Aggressor_IP":  {"accuracy": 0.020, "max_iter": 6000, "allin_threshold": 0.67},
    "3bet_hu.Aggressor_OOP": {"accuracy": 0.020, "max_iter": 6000, "allin_threshold": 0.67},
    "3bet_hu.Caller_IP":     {"accuracy": 0.025, "max_iter": 5000, "allin_threshold": 0.67},
    "3bet_hu.Caller_OOP":    {"accuracy": 0.025, "max_iter": 5000, "allin_threshold": 0.67},

    # --- 4-bet (keep tiny) ---
    "4bet_hu.Aggressor_IP":  {"accuracy": 0.030, "max_iter": 3000, "allin_threshold": 0.67},
    "4bet_hu.Aggressor_OOP": {"accuracy": 0.030, "max_iter": 3000, "allin_threshold": 0.67},
    "4bet_hu.Caller_IP":     {"accuracy": 0.030, "max_iter": 3000, "allin_threshold": 0.67},
    "4bet_hu.Caller_OOP":    {"accuracy": 0.030, "max_iter": 3000, "allin_threshold": 0.67},

    # --- Limped ---
    "limped_single.SB_IP": {"accuracy": 0.020, "max_iter": 4000, "allin_threshold": 0.67},
    "limped_multi.Any":    {"accuracy": 0.020, "max_iter": 4000, "allin_threshold": 0.67},
}

# Fallback if id not listed
SOLVER_DEFAULT = {"accuracy": 0.020, "max_iter": 5000, "allin_threshold": 0.67}

def profile_for(menu_id: str) -> Dict:
    return SOLVER_PROFILES.get((menu_id or "").strip(), SOLVER_DEFAULT).copy()