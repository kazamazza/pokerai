# tools/rangenet/etl/_solver_profiles.py
from typing import Dict

# Lean, context-aware knobs that keep runtime & file sizes under control
# Targets (typical on laptop): SRP 6–20s, 3bet 3–10s, 4bet/limp 2–6s; gz ≤ ~2–6 MB
SOLVER_PROFILES: Dict[str, Dict] = {
    # --- SRP ---
    "srp_hu.PFR_IP":     {"accuracy": 0.020, "max_iter": 4000, "allin_threshold": 0.67},
    "srp_hu.PFR_OOP":    {"accuracy": 0.020, "max_iter": 4000, "allin_threshold": 0.67},
    "srp_hu.Caller_IP":  {"accuracy": 0.020, "max_iter": 3800, "allin_threshold": 0.67},
    "srp_hu.Caller_OOP": {"accuracy": 0.020, "max_iter": 3800, "allin_threshold": 0.67},

    # --- 3-bet HU ---
    "3bet_hu.Aggressor_IP":  {"accuracy": 0.025, "max_iter": 3500, "allin_threshold": 0.67},
    "3bet_hu.Aggressor_OOP": {"accuracy": 0.025, "max_iter": 3500, "allin_threshold": 0.67},
    "3bet_hu.Caller_IP":     {"accuracy": 0.025, "max_iter": 3200, "allin_threshold": 0.67},
    "3bet_hu.Caller_OOP":    {"accuracy": 0.025, "max_iter": 3200, "allin_threshold": 0.67},

    # --- 4-bet HU (keep tiny) ---
    "4bet_hu.Aggressor_IP":  {"accuracy": 0.030, "max_iter": 2500, "allin_threshold": 0.67},
    "4bet_hu.Aggressor_OOP": {"accuracy": 0.030, "max_iter": 2500, "allin_threshold": 0.67},
    "4bet_hu.Caller_IP":     {"accuracy": 0.030, "max_iter": 2500, "allin_threshold": 0.67},
    "4bet_hu.Caller_OOP":    {"accuracy": 0.030, "max_iter": 2500, "allin_threshold": 0.67},

    # --- Limped ---
    "limped_single.SB_IP": {"accuracy": 0.020, "max_iter": 3000, "allin_threshold": 0.67},
    "limped_multi.Any":    {"accuracy": 0.020, "max_iter": 3000, "allin_threshold": 0.67},
}

# Fallback if id not listed
SOLVER_DEFAULT = {"accuracy": 0.025, "max_iter": 3500, "allin_threshold": 0.67}

def profile_for(menu_id: str) -> Dict:
    return SOLVER_PROFILES.get((menu_id or "").strip(), SOLVER_DEFAULT).copy()