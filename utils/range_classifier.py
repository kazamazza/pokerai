def classify_by_equity(
    equity: float,
    position: str,
    stack_bb: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str,
    action_context: str,
    opponent_position: str | None = None,
    hand_combo: str | None = None,
) -> str:
    mode = action_context.upper()

    if mode == "OPEN":
        return _classify_open_equity(
            equity, position, stack_bb,
            villain_profile, exploit_setting, multiway_context, population_type,
            opponent_position=opponent_position, hand_combo=hand_combo,   # passed (ignored inside)
        )

    elif mode == "VS_OPEN":
        return _classify_vs_open_equity(
            equity, position, stack_bb,
            villain_profile, exploit_setting, multiway_context, population_type,
            opponent_position=opponent_position, hand_combo=hand_combo,   # used
        )

    elif mode == "VS_ISO":
        return _classify_vs_iso_equity(
            equity, position, stack_bb,
            villain_profile, exploit_setting, multiway_context, population_type,
            opponent_position=opponent_position, hand_combo=hand_combo,   # likely used
        )

    elif mode == "VS_LIMP":
        return _classify_vs_limp_equity(
            equity, position, stack_bb,
            villain_profile, exploit_setting, multiway_context, population_type,
            opponent_position=opponent_position, hand_combo=hand_combo,   # likely used
        )

    elif mode == "VS_3BET":
        return _classify_vs_3bet_equity(
            equity, position, stack_bb,
            villain_profile, exploit_setting, multiway_context, population_type,
            opponent_position=opponent_position, hand_combo=hand_combo,   # useful
        )

    elif mode == "VS_4BET":
        return _classify_vs_4bet_equity(
            equity, position, stack_bb,
            villain_profile, exploit_setting, multiway_context, population_type,
            opponent_position=opponent_position, hand_combo=hand_combo,   # useful
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
    population_type: str,
    opponent_position: str | None = None,  # accepted (unused for OPEN)
    hand_combo: str | None = None,         # accepted (unused for OPEN)
) -> str:
    """
    Classify hero's action when first to act (OPEN) based on equity and context.
    Returns: 'open' or 'fold'.

    Notes:
      - `opponent_position` and `hand_combo` are accepted for uniform handler signatures,
        but are not used in OPEN classification.
    """
    # --- Normalize input casing ---
    position = position.upper()
    exploit_setting = exploit_setting.upper()
    villain_profile = villain_profile.upper()
    multiway_context = multiway_context.upper()
    population_type = population_type.upper()

    # --- Base threshold ---
    threshold = 0.56  # default for average depth and mid position

    # --- Adjust for position ---
    if position in ("UTG", "MP"):
        threshold += 0.02  # tighter from early
    elif position == "CO":
        threshold += 0.01
    elif position == "BTN":
        threshold -= 0.02  # button opens wider
    elif position == "SB":
        threshold -= 0.03  # SB can open very wide
    elif position == "BB":
        threshold -= 0.01  # BB opening is rare; kept for completeness

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
        threshold -= 0.01  # open wider vs tight pools
    elif villain_profile == "MANIAC":
        threshold += 0.01  # avoid marginal vs aggression

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
    population_type: str,
    opponent_position: str | None = None,  # new (opener's seat)
    hand_combo: str | None = None,         # new (e.g., 'KQo', 'A5s')
) -> str:
    """
    Classify hero's action when facing an open.
    Returns: '3bet', 'flat', or 'fold'.
    """

    # --- Normalize inputs ---
    position          = (position or "").upper()
    opponent_position = (opponent_position or "").upper()
    exploit_setting   = exploit_setting.upper()
    villain_profile   = villain_profile.upper()
    multiway_context  = multiway_context.upper()
    population_type   = population_type.upper()

    # --- Base thresholds ---
    threshold_3bet = 0.58
    threshold_flat = 0.52

    # --- Stack depth adjustments ---
    if stack_bb <= 15:
        threshold_3bet += 0.01
        threshold_flat += 0.01
    elif stack_bb >= 150:
        threshold_3bet -= 0.02
        threshold_flat -= 0.02
    elif stack_bb >= 100:
        threshold_3bet -= 0.01
        threshold_flat -= 0.01

    # --- Exploit settings ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_3bet -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_3bet -= 0.02
        threshold_flat -= 0.01

    # --- Villain profile (opener tendencies) ---
    if villain_profile == "NIT":
        threshold_3bet -= 0.01  # more incentive to 3bet if they overfold
    elif villain_profile == "MANIAC":
        threshold_3bet += 0.01  # avoid bloating pots vs very loose opens

    # --- Multiway pressure ---
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

    # --- Defender position specific ---
    if position in ("BTN", "SB"):
        threshold_flat -= 0.01  # more flats in LP / SB defend vs LP opens
    elif position in ("UTG", "MP"):
        threshold_flat += 0.01  # EP continues narrower, bias to 3bet/fold

    if position == "BB":
        # BB gets price to defend wider by flatting
        threshold_flat -= 0.01

    # --- NEW: Opener position sensitivity (tight EP vs loose LP opens) ---
    if opponent_position in ("UTG", "MP"):
        threshold_3bet += 0.01
        threshold_flat += 0.01
    elif opponent_position == "CO":
        threshold_3bet -= 0.005
        threshold_flat  -= 0.005
    elif opponent_position == "BTN":
        threshold_3bet -= 0.01
        threshold_flat  -= 0.015
    elif opponent_position == "SB":
        # SB opens are typically wide in many pools
        threshold_3bet -= 0.01
        threshold_flat  -= 0.01

    # --- NEW (very light gating): offsuit broadway bias outside BTN ---
    # discourage thin offsuit flats in non-LP seats (keeps ranges saner)
    if hand_combo:
        hc = hand_combo.strip().upper()
        if len(hc) == 3 and hc.endswith('O') and position not in ("BTN", "BB"):
            threshold_flat += 0.005

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
    population_type: str,
    opponent_position: str | None = None,  # limper's seat
    hand_combo: str | None = None,         # e.g. 'A5s', 'KQo'
) -> str:
    """
    Facing a limp: decide between 'iso', 'overlimp', or 'fold'.
    """

    # --- Normalize inputs ---
    position          = (position or "").upper()
    opponent_position = (opponent_position or "").upper()
    exploit_setting   = exploit_setting.upper()
    villain_profile   = villain_profile.upper()
    multiway_context  = multiway_context.upper()
    population_type   = population_type.upper()
    hc                = (hand_combo or "").upper()

    # --- Base thresholds ---
    threshold_iso      = 0.56
    threshold_overlimp = 0.50

    # --- Stack depth ---
    if stack_bb <= 15:
        threshold_iso += 0.01
        threshold_overlimp += 0.01
    elif stack_bb >= 150:
        threshold_iso      -= 0.015
        threshold_overlimp -= 0.01
    elif stack_bb >= 100:
        threshold_iso      -= 0.01
        threshold_overlimp -= 0.005

    # --- Exploit settings ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_iso -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_iso      -= 0.02
        threshold_overlimp -= 0.01

    # --- Villain profile (the limper) ---
    if villain_profile == "FISH":
        threshold_iso -= 0.015  # iso wider vs weak limps
    elif villain_profile == "MANIAC":
        threshold_iso += 0.01   # avoid marginal isos

    # --- Multiway pressure ---
    if multiway_context == "3WAY":
        threshold_iso      += 0.005
    elif multiway_context == "4WAY_PLUS":
        threshold_iso      += 0.01
        threshold_overlimp += 0.01

    # --- Population tendencies ---
    if population_type == "RECREATIONAL":
        threshold_iso -= 0.005
    elif population_type == "REGULAR":
        threshold_iso += 0.005

    # --- Hero position specifics ---
    if position == "BTN":
        threshold_iso      -= 0.012
        threshold_overlimp -= 0.005
    elif position == "CO":
        threshold_iso      -= 0.008
    elif position == "SB":
        threshold_iso      -= 0.005
    elif position == "BB":
        # BB often prefers cheap overlimps to bloating pots OOP
        threshold_iso      += 0.005
        threshold_overlimp -= 0.01

    # --- NEW: limper seat sensitivity (EP limps stronger than LP/SB limps generally) ---
    if opponent_position in ("UTG", "MP"):
        threshold_iso      += 0.008
        threshold_overlimp += 0.004
    elif opponent_position in ("CO", "BTN", "SB"):
        threshold_iso      -= 0.008
        threshold_overlimp -= 0.004

    # --- NEW: light hand-aware nudges to keep ranges sane ---
    if hc:
        # Suited wheel aces are premium iso candidates
        if re.fullmatch(r"A[2-5]S", hc):
            threshold_iso -= 0.01
        # Suited connectors: small nudge toward iso, especially in position
        elif hc in {"T9S","98S","87S","76S","65S","54S"}:
            threshold_iso -= 0.005
        # Offsuit broadways / marginal offsuit: prefer overlimp or fold in non-LP
        elif hc.endswith("O") and position not in {"BTN", "CO"}:
            threshold_iso += 0.006

    # --- Final decision ---
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
    population_type: str,
    opponent_position: str | None = None,  # first limper's seat
    hand_combo: str | None = None          # e.g. 'A5s', 'KQo'
) -> str:
    """
    Facing one or more limpers: decide between 'iso', 'overlimp', or 'fold'.
    """

    # --- Normalize inputs ---
    position          = (position or "").upper()
    opponent_position = (opponent_position or "").upper()
    exploit_setting   = exploit_setting.upper()
    villain_profile   = villain_profile.upper()
    multiway_context  = multiway_context.upper()
    population_type   = population_type.upper()
    hc                = (hand_combo or "").upper()

    # --- Base thresholds ---
    threshold_iso      = 0.58
    threshold_overlimp = 0.51

    # --- Stack depth adjustments ---
    if stack_bb <= 15:
        threshold_iso      += 0.01
        threshold_overlimp += 0.01
    elif stack_bb >= 150:
        threshold_iso      -= 0.015
        threshold_overlimp -= 0.01
    elif stack_bb >= 100:
        threshold_iso      -= 0.01
        threshold_overlimp -= 0.005

    # --- Exploit adjustments ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_iso -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_iso      -= 0.02
        threshold_overlimp -= 0.01

    # --- Villain profile adjustments ---
    if villain_profile == "FISH":
        threshold_iso -= 0.015  # iso wider vs weak limpers
    elif villain_profile == "MANIAC":
        threshold_iso += 0.01   # tighten up vs aggressive limp-raisers

    # --- Multiway pressure ---
    if multiway_context == "3WAY":
        threshold_iso += 0.005
    elif multiway_context == "4WAY_PLUS":
        threshold_iso      += 0.01
        threshold_overlimp += 0.01

    # --- Population tendencies ---
    if population_type == "RECREATIONAL":
        threshold_iso -= 0.005
    elif population_type == "REGULAR":
        threshold_iso += 0.005

    # --- Hero position adjustments ---
    if position == "BTN":
        threshold_iso      -= 0.01
        threshold_overlimp -= 0.005
    elif position == "SB":
        threshold_iso -= 0.005
    elif position == "UTG":
        threshold_iso      += 0.01
        threshold_overlimp += 0.005

    # --- Limper position impact ---
    if opponent_position in ("UTG", "MP"):
        threshold_iso      += 0.008  # early limps stronger
        threshold_overlimp += 0.004
    elif opponent_position in ("CO", "BTN", "SB"):
        threshold_iso      -= 0.008
        threshold_overlimp -= 0.004

    # --- Hand-specific nudges ---
    if hc:
        import re
        # Suited wheel aces are premium iso tools
        if re.fullmatch(r"A[2-5]S", hc):
            threshold_iso -= 0.01
        # Suited connectors get a small iso boost in position
        elif hc in {"T9S", "98S", "87S", "76S", "65S", "54S"}:
            threshold_iso -= 0.005
        # Offsuit junk broadways — avoid iso unless in LP
        elif hc.endswith("O") and position not in {"BTN", "CO"}:
            threshold_iso += 0.006

    # --- Final decision ---
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
    population_type: str,
    opponent_position: str | None = None,  # seat of the 3-bettor
    hand_combo: str | None = None          # e.g. 'AKo','QQ','A5s'
) -> str:
    """
    Facing a 3-bet after opening: decide '4bet', 'call', or 'fold'.
    """

    # --- Normalize ---
    position          = (position or "").upper()
    opponent_position = (opponent_position or "").upper()
    exploit_setting   = exploit_setting.upper()
    villain_profile   = villain_profile.upper()
    multiway_context  = multiway_context.upper()
    population_type   = population_type.upper()
    hc                = (hand_combo or "").upper()

    # --- Base thresholds (rough SRP defaults) ---
    threshold_4bet = 0.62
    threshold_call = 0.54

    # --- Stack depth adjustments ---
    if stack_bb <= 20:
        # shallow: bias to shove/fold over flats
        threshold_4bet += 0.02
        threshold_call += 0.01
    elif stack_bb >= 150:
        # deep: more flats/4-bet bluffs possible
        threshold_4bet -= 0.015
        threshold_call -= 0.01
    elif stack_bb >= 100:
        threshold_4bet -= 0.01
        threshold_call -= 0.01

    # --- Exploit settings ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_4bet -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_4bet -= 0.02
        threshold_call -= 0.01

    # --- Villain profile ---
    if villain_profile == "NIT":
        # tighter 3-bets → prefer fold/4-bet for value
        threshold_call += 0.01
    elif villain_profile == "MANIAC":
        # wider 3-bets → call more, mix 4-bet bluff a bit
        threshold_call -= 0.01
        threshold_4bet -= 0.005

    # --- Multiway pressure (rare but possible) ---
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

    # --- Hero position sensitivity (OOP vs IP matters for flatting) ---
    # Early opens flat a bit less vs 3-bet; Button can flat more.
    if position in {"UTG", "MP"}:
        threshold_call += 0.01
    elif position == "BTN":
        threshold_call -= 0.01

    # --- 3-bettor seat matters (range width) ---
    # Blinds tend to 3-bet wider vs LP; EP 3-bets are tighter.
    if opponent_position in {"SB", "BB"}:
        threshold_4bet -= 0.008
        threshold_call -= 0.006
    elif opponent_position in {"UTG", "MP"}:
        threshold_4bet += 0.008
        threshold_call += 0.006

    # --- Hand-specific nudges (light but helpful) ---
    if hc:
        import re
        pairs = {"AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22"}
        if hc in {"AA","KK"}:
            threshold_4bet -= 0.04
        elif hc in {"QQ","JJ"}:
            threshold_4bet -= 0.02
            threshold_call  -= 0.01
        elif hc == "AKS" or hc == "AKO":
            # AK often indifferent: make both actions slightly easier
            threshold_4bet -= 0.01
            threshold_call  -= 0.01
        elif hc in pairs and stack_bb >= 100 and hc not in {"AA","KK","QQ"}:
            # set-mining deep: encourage calling a touch
            threshold_call -= 0.006
        elif re.fullmatch(r"A[2-5]S", hc):
            # suited wheel bluff candidates
            threshold_4bet -= 0.006
        elif hc in {"AQO","AJS","AQs","AJo"}:
            # AQ/AJ tend to call more than 4-bet vs tight ranges
            threshold_call -= 0.004
            threshold_4bet += 0.004
        elif hc.endswith("O") and position not in {"BTN","CO"}:
            # offsuit marginal broadways OOP → discourage flats
            threshold_call += 0.006

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
    population_type: str,
    opponent_position: str | None = None,  # seat of the 4-bettor
    hand_combo: str | None = None          # e.g. 'AKs','QQ','A5s'
) -> str:
    """
    Facing a 4-bet after 3-betting: decide 'jam', 'call', or 'fold'.
    """

    # --- Normalize inputs ---
    position          = (position or "").upper()
    opponent_position = (opponent_position or "").upper()
    exploit_setting   = exploit_setting.upper()
    villain_profile   = villain_profile.upper()
    multiway_context  = multiway_context.upper()
    population_type   = population_type.upper()
    hc                = (hand_combo or "").upper()

    # --- Base thresholds ---
    threshold_jam  = 0.66
    threshold_call = 0.58

    # --- Stack depth logic ---
    if stack_bb <= 20:
        # short: shove more, call less
        threshold_jam  -= 0.02
        threshold_call += 0.01
    elif stack_bb <= 40:
        threshold_jam  -= 0.01
    elif stack_bb >= 150:
        # deep: flatter more, jam tighter
        threshold_jam  += 0.02
        threshold_call -= 0.01
    elif stack_bb >= 100:
        threshold_jam  += 0.01
        threshold_call -= 0.005

    # --- Exploit settings ---
    if exploit_setting == "EXPLOIT_LIGHT":
        threshold_jam  -= 0.01
    elif exploit_setting == "EXPLOIT_HEAVY":
        threshold_jam  -= 0.02
        threshold_call -= 0.01

    # --- Villain profile ---
    if villain_profile == "NIT":
        # tighter 4-bets → prefer jam only for strong value, fewer calls
        threshold_jam  += 0.01
        threshold_call += 0.005
    elif villain_profile == "MANIAC":
        # wider 4-bets → jam/call slightly easier
        threshold_jam  -= 0.01
        threshold_call -= 0.005

    # --- Multiway context (rare) ---
    if multiway_context == "3WAY":
        threshold_jam  += 0.01
        threshold_call += 0.01
    elif multiway_context == "4WAY_PLUS":
        threshold_jam  += 0.015
        threshold_call += 0.01

    # --- Population tendencies ---
    if population_type == "RECREATIONAL":
        threshold_call -= 0.005
    elif population_type == "REGULAR":
        threshold_call += 0.005

    # --- Hero position sensitivity ---
    # EP tends to jam tighter vs 4-bets; BTN can mix more calls.
    if position in {"UTG", "MP"}:
        threshold_jam  += 0.008
        threshold_call += 0.004
    elif position == "BTN":
        threshold_jam  -= 0.006
        threshold_call -= 0.004
    elif position == "SB":
        # out of position post → bias away from thin calls
        threshold_call += 0.006

    # --- 4-bettor seat matters (range width) ---
    # Blinds/BTN often 4-bet wider than EP.
    if opponent_position in {"SB", "BB", "BTN"}:
        threshold_jam  -= 0.006
        threshold_call -= 0.004
    elif opponent_position in {"UTG", "MP"}:
        threshold_jam  += 0.006
        threshold_call += 0.004

    # --- Hand-specific nudges ---
    if hc:
        pairs = {"AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22"}
        if hc in {"AA","KK"}:
            threshold_jam  -= 0.05
            threshold_call += 0.01  # mostly jam, but don’t forbid trapping entirely
        elif hc == "QQ":
            threshold_jam  -= 0.02
            # deep stacks: call a bit more with QQ
            if stack_bb >= 120:
                threshold_call -= 0.008
        elif hc == "JJ":
            # more call than jam, esp. deep
            threshold_jam  += 0.006
            if stack_bb >= 100:
                threshold_call -= 0.01
        elif hc in {"AKS","AKO"}:
            # AK is close: slightly ease both, more shove shallow
            threshold_jam  -= 0.012 if stack_bb <= 50 else 0.006
            threshold_call -= 0.006
        elif hc in pairs and hc not in {"AA","KK","QQ","JJ"}:
            # set-mining deep: incentivize calling a touch when deep
            if stack_bb >= 120:
                threshold_call -= 0.006
        elif hc.endswith("S") and hc.startswith("A") and hc in {"A5S","A4S"}:
            # suited wheel blockers as bluff jams occasionally
            threshold_jam  -= 0.006

    # --- Final decision ---
    if equity >= threshold_jam:
        return "jam"
    elif equity >= threshold_call:
        return "call"
    else:
        return "fold"