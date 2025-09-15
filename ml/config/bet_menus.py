BET_SIZE_MENUS = {
    # --- Single Raised Pots (SRP) ---
    "srp_hu.PFR_IP":     [0.25, 0.33, 0.50, 0.75, 1.00],  # in position c-bets
    "srp_hu.PFR_OOP":    [0.33, 0.50, 0.75, 1.00],        # out of position c-bets
    "srp_hu.Caller_IP":  [0.25, 0.50, 0.75],              # caller IP probes
    "srp_hu.Caller_OOP": [0.25, 0.50, 0.75],              # caller OOP donks

    # --- 3-Bet Pots ---
    "3bet_hu.Aggressor_IP":  [0.25, 0.50, 0.75],
    "3bet_hu.Aggressor_OOP": [0.25, 0.50, 0.75],
    "3bet_hu.Caller_IP":     [0.33, 0.66, 1.00],
    "3bet_hu.Caller_OOP":    [0.33, 0.66, 1.00],

    # --- 4-Bet Pots ---
    "4bet_hu.Aggressor_IP":  [0.25, 0.50, 0.75],
    "4bet_hu.Aggressor_OOP": [0.25, 0.50, 0.75],
    "4bet_hu.Caller_IP":     [0.33, 0.66, 1.00],
    "4bet_hu.Caller_OOP":    [0.33, 0.66, 1.00],

    # --- Limped Pots ---
    "limped_single.SB_IP": [0.33, 0.66, 1.00],   # SB limps, BB checks
    "limped_multi.Any":    [0.33, 0.66, 1.00],   # generic for multiway limped pots
}

# fallback if context not mapped
DEFAULT_MENU = [0.33, 0.66, 1.00]