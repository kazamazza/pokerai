import json
import os
from typing import Dict, Any, List, Union
from typing import Literal
from features.types import RangeFeatureRequest, Action, ActionType, Hand
from utils.range_parser import expand_range_syntax

def sum_action_amounts(actions: List[Action]) -> float:
    return sum(a.amount or 0.0 for a in actions
               if a.type in {
                   ActionType.CALL, ActionType.BET,
                   ActionType.RAISE, ActionType.POST_SB,
                   ActionType.POST_BB
               })

BUCKET_EDGES = {
    'vpip_pct':              [0.05, 0.15, 0.30, 0.50],
    'pfr_pct':               [0.05, 0.15, 0.30, 0.50],
    'three_bet_pct':         [0.01, 0.03, 0.07, 0.15],
    'flop_cbet_pct':         [0.25, 0.50, 0.75],
    'fold_to_flop_cbet_pct': [0.25, 0.50, 0.75],
}

def _bucket_index(value: float, edges: List[float]) -> int:
    """Return bucket index for value given sorted edges."""
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def compute_profile_stats(
    history: List[Union[Hand, Dict[str, Any]]]
) -> Dict[str, Dict[str, int]]:
    """
    Compute bucketed VPIP, PFR, 3-bet, flop C-bet and fold-to-flop-C-bet rates per player.
    Accepts a list of either Hand models or plain dicts with the same keys.
    Returns: { player_id: { 'vpip_bin':…, 'pfr_bin':…, … } }
    """
    stats: Dict[str, Dict[str, float]] = {}

    # 1) Count raw events
    for raw_hand in history:
        # unpack seats, actions, street
        if isinstance(raw_hand, Hand):
            seats, actions, street = raw_hand.seats, raw_hand.actions, raw_hand.street
        else:
            seats   = raw_hand.get("seats", [])
            actions = raw_hand.get("actions", [])
            street  = raw_hand.get("street", "")

        # init per-player counters
        for s in seats:
            pid = s.player_id if hasattr(s, "player_id") else s["player_id"]
            stats.setdefault(pid, {
                "hands": 0, "vpip": 0, "pfr": 0, "3bet": 0,
                "flop_cbet": 0, "flop_cbet_faced": 0, "fold_to_flop_cbet": 0
            })

        # everyone saw this hand
        for pid in stats:
            stats[pid]["hands"] += 1

        preflop_raiser = None
        for act in actions:
            # unify both Action objects and legacy strings
            if isinstance(act, Action):
                pid, act_type = act.player_id, act.type
            else:
                parts = act.split()
                if len(parts) < 2:
                    continue
                pid = parts[0]
                verb = parts[1].lower().rstrip("s")
                try:
                    act_type = ActionType(verb)
                except ValueError:
                    continue

            if pid not in stats:
                continue
            st = stats[pid]

            # Preflop
            if street == "preflop":
                if act_type in (ActionType.CALL, ActionType.BET, ActionType.RAISE):
                    st["vpip"] += 1
                if act_type == ActionType.RAISE:
                    if preflop_raiser is None:
                        preflop_raiser = pid
                        st["pfr"] += 1
                    else:
                        # any subsequent raise in the same hand is a 3-bet
                        st["3bet"] += 1

            # Flop
            if street == "flop":
                if act_type == ActionType.BET:
                    st["flop_cbet"] += 1
                    for other in stats:
                        if other != pid:
                            stats[other]["flop_cbet_faced"] += 1
                elif act_type == ActionType.FOLD and st["flop_cbet_faced"] > 0:
                    st["fold_to_flop_cbet"] += 1

    # 2) Convert counts into bucket indices
    rates: Dict[str, Dict[str, int]] = {}
    for pid, c in stats.items():
        hands = max(c["hands"], 1)
        faced = max(c["flop_cbet_faced"], 1)
        vpip_pct = c["vpip"] / hands
        pfr_pct = c["pfr"] / hands
        three_bet_pct = c["3bet"] / hands
        flop_cbet_pct = c["flop_cbet"] / faced
        fold_to_flop_pct = c["fold_to_flop_cbet"] / faced
        rates[pid] = {
            'vpip_bin':              _bucket_index(vpip_pct, BUCKET_EDGES['vpip_pct']),
            'pfr_bin':               _bucket_index(pfr_pct, BUCKET_EDGES['pfr_pct']),
            'three_bet_bin':         _bucket_index(three_bet_pct, BUCKET_EDGES['three_bet_pct']),
            'flop_cbet_bin':         _bucket_index(flop_cbet_pct, BUCKET_EDGES['flop_cbet_pct']),
            'fold_to_flop_cbet_bin': _bucket_index(fold_to_flop_pct, BUCKET_EDGES['fold_to_flop_cbet_pct']),
        }
    return rates


SITUATIONAL_BUCKET_EDGES = {
    # Turn bet sizing as % of pot
    'turn_bet_size_pct':            [0.20, 0.40, 0.60, 0.80],  # → 5 bins

    # Pot odds offered to hero
    'pot_odds_pct':                 [0.20, 0.33, 0.50, 0.67],  # → 5 bins

    # SPR buckets
    'spr':                          [2.0, 4.0, 8.0, 16.0],     # → 5 bins

    # Actions since last aggression
    'actions_since_last_aggression':[1, 2, 4, 8],              # → 5 bins

    # Previous showdown tendency (WTSD)
    'prev_showdown_tendency':       [0.10, 0.30, 0.60, 0.90],  # → 5 bins
}

def compute_situational_stats(
    req: RangeFeatureRequest,
    profile_stats: Dict[str, Dict[str, int]]
) -> Dict[str, Any]:
    """
    Compute all situational features for Range-Net:
      - Boolean: ip_vs_last_raiser, did_flop_cbet
      - Bucketed: spr_bin, pot_odds_pct_bin, turn_bet_size_pct_bin,
                  actions_since_last_aggression_bin,
                  prev_showdown_tendency_bin
      - One-hot: street_onehot
      - Raw: pot_size, num_opponents, turn_cbet_pct, fold_to_turn_cbet_pct,
             aggression_factor
    """
    hand = req.current_hand
    pid  = req.player_id

    # 1) Pot & SPR
    pot = sum_action_amounts(hand.actions)
    stacks = [s.stack_size for s in hand.seats]
    eff_stack = min(stacks) if stacks else 0.0
    spr = eff_stack / pot if pot > 0 else 0.0

    # 2) Opponents
    num_opponents = max(len(hand.seats) - 1, 0)

    # 3) Historical turn stats
    stats = profile_stats.get(pid, {})
    turn_cbet_pct         = stats.get('turn_cbet_pct', 0.0)
    fold_to_turn_cbet_pct = stats.get('fold_to_turn_cbet_pct', 0.0)
    # 4) Aggression factor
    wins = ties = 0
    bets_and_raises = sum(
        1 for past in req.history
          for a in past.actions
          if getattr(a, 'player_id', None)==pid and a.type in (ActionType.BET, ActionType.RAISE)
    )
    calls = sum(
        1 for past in req.history
          for a in past.actions
          if getattr(a, 'player_id', None)==pid and a.type == ActionType.CALL
    )
    aggression_factor = (bets_and_raises / calls) if calls > 0 else 0.0

    # 5) Boolean flags
    # did_flop_cbet?
    did_flop_cbet = any(
        getattr(a, 'player_id', None)==pid and a.type == ActionType.BET and past.street=='flop'
        for past in req.history for a in past.actions
    )
    # ip_vs_last_raiser (placeholder: always 0, implement seat-order logic)
    ip_vs_last_raiser = 0

    # 6) Actions since last aggression on this street
    # find last BET/RAISE index
    acts = [a for a in hand.actions if getattr(a,'player_id',None)==pid]
    last_agg = next(
        (i for i, a in enumerate(acts[::-1]) if a.type in (ActionType.BET, ActionType.RAISE)),
        None
    )
    actions_since = last_agg if last_agg is not None else len(acts)

    # 7) Pot odds pct (hero-to-act facing a bet of size min_bet? stub as 0)
    pot_odds_pct = 0.0  # TODO: compute based on pending bet vs. pot

    # 8) Turn bet size pct (avg bet size / pot on turn) → stub
    turn_bets = [a.amount for a in hand.actions if hasattr(a,'street') and a.street=='turn' and a.type==ActionType.BET]
    turn_bet_size_pct = (sum(turn_bets)/len(turn_bets)/pot) if turn_bets else 0.0

    # 9) Prev showdown tendency (from profile_stats if stored)
    prev_showdown = stats.get('wtsd_pct', 0.0)

    # 10) Street one-hot
    streets = ['flop','turn','river']
    street_onehot = [1 if hand.street==s else 0 for s in streets]

    # Now bucket where needed
    out: Dict[str, Any] = {
        'pot_size': pot,
        'num_opponents': num_opponents,
        'turn_cbet_pct': turn_cbet_pct,
        'fold_to_turn_cbet_pct': fold_to_turn_cbet_pct,
        'aggression_factor': aggression_factor,
        'did_flop_cbet': int(did_flop_cbet),
        'ip_vs_last_raiser': ip_vs_last_raiser,
        'street_onehot': street_onehot,
    }

    # Helper to bucket & one-hot
    def _bucket_feature(name: str, raw: float):
        edges = SITUATIONAL_BUCKET_EDGES[name]
        idx = _bucket_index(raw, edges)
        onehot = [0.0]*(len(edges)+1)
        onehot[idx] = 1.0
        for i, v in enumerate(onehot):
            out[f"{name}_bin_{i}"] = v

    # Bucketed
    _bucket_feature('spr', spr)
    _bucket_feature('pot_odds_pct', pot_odds_pct)
    _bucket_feature('turn_bet_size_pct', turn_bet_size_pct)
    _bucket_feature('actions_since_last_aggression', actions_since)
    _bucket_feature('prev_showdown_tendency', prev_showdown)

    return out


SUPPLEMENTAL_BUCKET_EDGES = {
    # Showdown frequencies (if you choose to bucket; optional)
    'wtsd_pct':               [0.20, 0.40, 0.60, 0.80],  # → 5 bins
    'wsd_pct':                [0.20, 0.40, 0.60, 0.80],  # → 5 bins

    # Implied odds: extra chips won beyond pot, as ratio
    'implied_odds':           [0.50, 1.00, 2.00, 3.00],  # → 5 bins

    # Dead money: avg pot when they first act, in BB
    'dead_money':             [1, 2, 5, 10],             # → 5 bins

    # Post-flop reaction rates
    'fold_to_3bet_pct':       [0.10, 0.25, 0.50, 0.75],  # → 5 bins
    'check_raise_pct':        [0.05, 0.15, 0.30, 0.50],  # → 5 bins

    # River continuation-bet & fold rates
    'river_cbet_pct':         [0.10, 0.30, 0.60, 0.90],  # → 5 bins
    'fold_to_river_cbet_pct': [0.10, 0.30, 0.60, 0.90],  # → 5 bins

    # Overall check-raise across streets
    'raise_after_check_pct':  [0.05, 0.15, 0.30, 0.50],  # → 5 bins
}

def compute_supplemental_stats(
    req: RangeFeatureRequest,
    profile_stats: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute supplemental features for Range-Net, bucketed according to SUPPLEMENTAL_BUCKET_EDGES.
    Expects profile_stats[pid] to contain raw keys:
      'wtsd_pct', 'wsd_pct', 'implied_odds', 'dead_money',
      'fold_to_3bet_pct', 'check_raise_pct',
      'river_cbet_pct', 'fold_to_river_cbet_pct', 'raise_after_check_pct'
    """
    pid = req.player_id
    stats = profile_stats.get(pid, {})

    # Pull raw values (default 0.0)
    wtsd    = stats.get('wtsd_pct', 0.0)
    wsd     = stats.get('wsd_pct', 0.0)
    implied = stats.get('implied_odds', 0.0)
    dead    = stats.get('dead_money', 0.0)
    f3b     = stats.get('fold_to_3bet_pct', 0.0)
    cr      = stats.get('check_raise_pct', 0.0)
    rcb     = stats.get('river_cbet_pct', 0.0)
    frcb    = stats.get('fold_to_river_cbet_pct', 0.0)
    rac     = stats.get('raise_after_check_pct', 0.0)

    # Base output: keep raw wtsd/wsd as floats (or bucket if desired)
    out: Dict[str, Any] = {
        'wtsd_pct': wtsd,
        'wsd_pct':  wsd,
    }

    # Helper to bucket + one-hot
    def _bucket(name: str, raw: float):
        edges = SUPPLEMENTAL_BUCKET_EDGES[name]
        idx = _bucket_index(raw, edges)
        onehot = [0.0] * (len(edges) + 1)
        onehot[idx] = 1.0
        for i, v in enumerate(onehot):
            out[f"{name}_bin_{i}"] = v

    # Bucket the rest
    _bucket('implied_odds', implied)
    _bucket('dead_money', dead)
    _bucket('fold_to_3bet_pct', f3b)
    _bucket('check_raise_pct', cr)
    _bucket('river_cbet_pct', rcb)
    _bucket('fold_to_river_cbet_pct', frcb)
    _bucket('raise_after_check_pct', rac)

    return out

def compute_position_onehot(seats: List[Dict[str, Any]], button_seat: int, player_id: str) -> List[int]:
    """
    Compute a one-hot position vector for a 6-max table.
    - seats: list of dicts with 'seat_number' and 'player_id'
    - button_seat: the seat number of the button
    - player_id: the player to encode

    Returns a list of length len(seats) with a single 1 at the player's relative position.
    """
    # Sort seats by their seat_number
    sorted_seats = sorted(seats, key=lambda s: s['seat_number'])
    # Find index of button in sorted order
    btn_idx = next((i for i, s in enumerate(sorted_seats)
                    if s['seat_number'] == button_seat), 0)
    # Rotate so button is at position 0
    rotated = sorted_seats[btn_idx:] + sorted_seats[:btn_idx]
    # Build one-hot for player_id
    vec = [0] * len(rotated)
    for i, s in enumerate(rotated):
        if s['player_id'] == player_id:
            vec[i] = 1
            break
    return vec

Position = Literal["UTG", "MP", "CO", "BTN", "SB", "BB"]


def get_preflop_range(
    position: str,
    stack_depth: int,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str,
    action_context: str
) -> list[str]:
    """
    Return full combo-expanded GTO-like range as list of concrete hand strings.
    Loads from pre-generated JSON using full configuration key.
    """
    key = f"{position}|{stack_depth}|{villain_profile}|{exploit_setting}|{multiway_context}|{action_context}|{population_type}"
    print(f"[DEBUG] Lookup key: {key}")
    file_path = "data/villain_range_map.json"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Could not find preflop range data: {file_path}")

    with open(file_path, "r") as f:
        all_ranges = json.load(f)

    if key not in all_ranges:
        raise KeyError(f"❌ Range not found for key: {key}")

    raw_range = all_ranges[key]
    print(f"[DEBUG] raw range: {raw_range}")
    return expand_range_syntax(raw_range)

def get_stack_bucket_label(stack_depth: float) -> str:
    """
    Return a bucket label given a stack depth in bb.
    """
    if stack_depth <= 12:
        return "ultra_short"
    elif stack_depth <= 25:
        return "short"
    elif stack_depth <= 50:
        return "mid"
    elif stack_depth <= 90:
        return "deep"
    elif stack_depth <= 125:
        return "standard"
    elif stack_depth <= 175:
        return "deepstack"
    else:
        return "very_deep"