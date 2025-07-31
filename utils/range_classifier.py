def classify_by_equity(
    equity: float,
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str,
    action_context: str
) -> str:
    mode = action_context.upper()

    if mode == "OPEN":
        return _classify_open_equity(
            equity, position, stack_bb, villain_profile,
            exploit_setting, multiway_context, population_type
        )
    elif mode == "VS_OPEN":
        return _classify_vs_open_equity(
            equity, position, stack_bb, villain_profile,
            exploit_setting, multiway_context, population_type
        )
    elif mode == "VS_ISO":
        return _classify_vs_iso_equity(
            equity, position, stack_bb, villain_profile,
            exploit_setting, multiway_context, population_type
        )
    elif mode == "VS_LIMP":
        return _classify_vs_limp_equity(
            equity, position, stack_bb, villain_profile,
            exploit_setting, multiway_context, population_type
        )
    elif mode == "VS_3BET":
        return _classify_vs_3bet_equity(
            equity, position, stack_bb, villain_profile,
            exploit_setting, multiway_context, population_type
        )
    elif mode == "VS_4BET":
        return _classify_vs_4bet_equity(
            equity, position, stack_bb, villain_profile,
            exploit_setting, multiway_context, population_type
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


# --- Internal Mode Handlers ---

def _classify_open_equity(
    equity: float,
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str
) -> str:
    """
    Classify hero's action when first to act (open spot) based on equity and context.
    Returns: 'open' or 'fold'
    """
    # --- Normalize input casing ---
    position = position.upper()
    exploit_setting = exploit_setting.upper()
    villain_profile = villain_profile.upper()
    multiway_context = multiway_context.upper()
    population_type = population_type.upper()

    # --- Base threshold ---
    threshold = 0.56  # Default for average depth and mid position

    # --- Adjust for position ---
    if position in ("UTG", "MP"):
        threshold += 0.02  # tighter from early position
    elif position == "CO":
        threshold += 0.01  # in between
    elif position == "BTN":
        threshold -= 0.02  # button opens wider
    elif position == "SB":
        threshold -= 0.03  # SB can open very wide
    elif position == "BB":
        threshold -= 0.01  # BB opening is rare, but not impossible in special cases

    # --- Adjust for stack depth ---
    if stack_bb <= 15:
        threshold += 0.02  # ultra short
    elif stack_bb <= 25:
        threshold += 0.01
    elif stack_bb >= 150:
        threshold -= 0.02
    elif stack_bb >= 100:
        threshold -= 0.01

    # --- Exploit settings ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold -= 0.02

    # --- Villain profile ---
    if villain_profile == "NIT":
        threshold -= 0.01  # open wider against tight players
    elif villain_profile == "MANIAC":
        threshold += 0.01  # avoid marginal hands vs aggression

    # --- Multiway context ---
    if multiway_context == "3WAY":
        threshold += 0.01
    elif multiway_context == "4WAY_PLUS":
        threshold += 0.02

    # --- Population tendencies ---
    if population_type == "RECREATIONAL":
        threshold -= 0.005
    elif population_type == "REGULAR":
        threshold += 0.005

    # --- Final decision ---
    return "open" if equity >= threshold else "fold"

def _classify_vs_open_equity(
    equity: float,
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str
) -> str:
    """
    Classify hero's action when facing an open (preflop) based on equity and context.
    Returns: '3bet', 'flat', or 'fold'
    """
    # --- Normalize inputs ---
    position = position.upper()
    exploit_setting = exploit_setting.upper()
    villain_profile = villain_profile.upper()
    multiway_context = multiway_context.upper()
    population_type = population_type.upper()

    # --- Base thresholds ---
    threshold_3bet = 0.58
    threshold_flat = 0.52

    # --- Stack depth adjustments ---
    if stack_bb <= 15:
        threshold_3bet += 0.01
        threshold_flat += 0.01
    elif stack_bb >= 100:
        threshold_3bet -= 0.01
        threshold_flat -= 0.01
    elif stack_bb >= 150:
        threshold_3bet -= 0.02
        threshold_flat -= 0.02

    # --- Exploit settings ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_3bet -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_3bet -= 0.02
        threshold_flat -= 0.01

    # --- Villain profile ---
    if villain_profile == "NIT":
        threshold_3bet -= 0.01  # more incentive to 3bet if they fold a lot
    elif villain_profile == "MANIAC":
        threshold_3bet += 0.01  # avoid bloating pots vs loose-aggressive

    # --- Multiway adjustments ---
    if multiway_context == "3WAY":
        threshold_3bet += 0.01
        threshold_flat += 0.005
    elif multiway_context == "4WAY_PLUS":
        threshold_3bet += 0.015
        threshold_flat += 0.01

    # --- Population tendencies ---
    if population_type == "RECREATIONAL":
        threshold_3bet -= 0.005
    elif population_type == "REGULAR":
        threshold_3bet += 0.005

    # --- Position-specific adjustments ---
    if position in ("BTN", "SB"):
        threshold_flat -= 0.01  # more flats in LP
    elif position in ("UTG", "MP"):
        threshold_flat += 0.01  # more value-heavy ranges

    # --- Final classification ---
    if equity >= threshold_3bet:
        return "3bet"
    elif equity >= threshold_flat:
        return "flat"
    else:
        return "fold"

def _classify_vs_iso_equity(
    equity: float,
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str
) -> str:
    """
    Classify hero's action when facing a limp (option to isolate).
    Returns: 'iso', 'overlimp', or 'fold'
    """
    # --- Normalize inputs ---
    position = position.upper()
    exploit_setting = exploit_setting.upper()
    villain_profile = villain_profile.upper()
    multiway_context = multiway_context.upper()
    population_type = population_type.upper()

    # --- Base thresholds ---
    threshold_iso = 0.56
    threshold_overlimp = 0.50

    # --- Stack depth ---
    if stack_bb <= 15:
        threshold_iso += 0.01
        threshold_overlimp += 0.01
    elif stack_bb >= 100:
        threshold_iso -= 0.01
        threshold_overlimp -= 0.005
    elif stack_bb >= 150:
        threshold_iso -= 0.015
        threshold_overlimp -= 0.01

    # --- Exploit settings ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_iso -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_iso -= 0.02
        threshold_overlimp -= 0.01

    # --- Villain profile ---
    if villain_profile == "FISH":
        threshold_iso -= 0.015  # isolate weaker opponents more aggressively
    elif villain_profile == "MANIAC":
        threshold_iso += 0.01  # play tighter when isolation could backfire

    # --- Multiway ---
    if multiway_context == "3WAY":
        threshold_iso += 0.005
    elif multiway_context == "4WAY_PLUS":
        threshold_iso += 0.01
        threshold_overlimp += 0.01

    # --- Population type ---
    if population_type == "RECREATIONAL":
        threshold_iso -= 0.005
    elif population_type == "REGULAR":
        threshold_iso += 0.005

    # --- Position specific ---
    if position == "BTN":
        threshold_iso -= 0.01
    elif position == "SB":
        threshold_iso -= 0.005
    elif position == "UTG":
        threshold_iso += 0.01

    # --- Final classification ---
    if equity >= threshold_iso:
        return "iso"
    elif equity >= threshold_overlimp:
        return "overlimp"
    else:
        return "fold"

def _classify_vs_limp_equity(
    equity: float,
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str
) -> str:
    """
    Classify hero's action when facing one or more limpers.
    Returns: 'iso', 'overlimp', or 'fold'
    """
    # --- Normalize inputs ---
    position = position.upper()
    exploit_setting = exploit_setting.upper()
    villain_profile = villain_profile.upper()
    multiway_context = multiway_context.upper()
    population_type = population_type.upper()

    # --- Base thresholds ---
    threshold_iso = 0.58
    threshold_overlimp = 0.51

    # --- Stack depth adjustments ---
    if stack_bb <= 15:
        threshold_iso += 0.01
        threshold_overlimp += 0.01
    elif stack_bb >= 100:
        threshold_iso -= 0.01
        threshold_overlimp -= 0.005
    elif stack_bb >= 150:
        threshold_iso -= 0.015
        threshold_overlimp -= 0.01

    # --- Exploit adjustments ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_iso -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_iso -= 0.02
        threshold_overlimp -= 0.01

    # --- Villain profile adjustments ---
    if villain_profile == "FISH":
        threshold_iso -= 0.015
    elif villain_profile == "MANIAC":
        threshold_iso += 0.01

    # --- Multiway pressure ---
    if multiway_context == "3WAY":
        threshold_iso += 0.005
    elif multiway_context == "4WAY_PLUS":
        threshold_iso += 0.01
        threshold_overlimp += 0.01

    # --- Population tendencies ---
    if population_type == "RECREATIONAL":
        threshold_iso -= 0.005
    elif population_type == "REGULAR":
        threshold_iso += 0.005

    # --- Position-specific adjustments ---
    if position == "BTN":
        threshold_iso -= 0.01
        threshold_overlimp -= 0.005
    elif position == "SB":
        threshold_iso -= 0.005
    elif position == "UTG":
        threshold_iso += 0.01
        threshold_overlimp += 0.005

    # --- Final classification ---
    if equity >= threshold_iso:
        return "iso"
    elif equity >= threshold_overlimp:
        return "overlimp"
    else:
        return "fold"

def _classify_vs_3bet_equity(
    equity: float,
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str
) -> str:
    """
    Classify hero's action when facing a 3-bet after opening.
    Returns: 'call', '4bet', or 'fold'
    """
    # --- Normalize inputs ---
    position = position.upper()
    exploit_setting = exploit_setting.upper()
    villain_profile = villain_profile.upper()
    multiway_context = multiway_context.upper()
    population_type = population_type.upper()

    # --- Base thresholds ---
    threshold_4bet = 0.62
    threshold_call = 0.54

    # --- Stack depth adjustments ---
    if stack_bb <= 20:
        threshold_4bet += 0.02  # prefer shove/fold
        threshold_call += 0.01
    elif stack_bb >= 100:
        threshold_4bet -= 0.01
        threshold_call -= 0.01
    elif stack_bb >= 150:
        threshold_4bet -= 0.015
        threshold_call -= 0.01

    # --- Exploit adjustments ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_4bet -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_4bet -= 0.02
        threshold_call -= 0.01

    # --- Villain profile adjustments ---
    if villain_profile == "NIT":
        threshold_call -= 0.01  # rarely bluffs → fold more
    elif villain_profile == "MANIAC":
        threshold_call += 0.01  # bluff-heavy → call more

    # --- Multiway pressure (rare, but present in some formats) ---
    if multiway_context == "3WAY":
        threshold_4bet += 0.01
        threshold_call += 0.01
    elif multiway_context == "4WAY_PLUS":
        threshold_4bet += 0.015
        threshold_call += 0.01

    # --- Population tendencies ---
    if population_type == "RECREATIONAL":
        threshold_call -= 0.005
    elif population_type == "REGULAR":
        threshold_call += 0.005

    # --- Position-based caution ---
    if position == "UTG":
        threshold_4bet += 0.01
    elif position == "BTN":
        threshold_4bet -= 0.01

    # --- Final decision ---
    if equity >= threshold_4bet:
        return "4bet"
    elif equity >= threshold_call:
        return "call"
    else:
        return "fold"

def _classify_vs_4bet_equity(
    equity: float,
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str
) -> str:
    """
    Classify hero's action when facing a 4-bet after 3-betting.
    Returns: 'jam', 'call', or 'fold'
    """
    # --- Normalize inputs ---
    position = position.upper()
    exploit_setting = exploit_setting.upper()
    villain_profile = villain_profile.upper()
    multiway_context = multiway_context.upper()
    population_type = population_type.upper()

    # --- Base thresholds ---
    threshold_jam = 0.66
    threshold_call = 0.58

    # --- Stack depth logic ---
    if stack_bb <= 20:
        threshold_jam -= 0.01  # short stack → more shoving
        threshold_call += 0.01
    elif stack_bb >= 100:
        threshold_jam += 0.01
        threshold_call -= 0.01
    elif stack_bb >= 150:
        threshold_jam += 0.015
        threshold_call -= 0.01

    # --- Exploit settings ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_jam -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_jam -= 0.02
        threshold_call -= 0.01

    # --- Villain profile ---
    if villain_profile == "NIT":
        threshold_jam += 0.01  # tighter vs tight 4-bet range
    elif villain_profile == "MANIAC":
        threshold_call -= 0.01  # looser call vs wide 4-bet

    # --- Multiway context ---
    if multiway_context == "3WAY":
        threshold_jam += 0.01
        threshold_call += 0.01
    elif multiway_context == "4WAY_PLUS":
        threshold_jam += 0.015
        threshold_call += 0.01

    # --- Population tendencies ---
    if population_type == "RECREATIONAL":
        threshold_call -= 0.005
    elif population_type == "REGULAR":
        threshold_call += 0.005

    # --- Position sensitivity ---
    if position == "BTN":
        threshold_jam -= 0.005
    elif position == "UTG":
        threshold_jam += 0.01

    # --- Final decision ---
    if equity >= threshold_jam:
        return "jam"
    elif equity >= threshold_call:
        return "call"
    else:
        return "fold"