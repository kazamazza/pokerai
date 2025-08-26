from ml.core.types import Street, Act, Flag, Ctx
from typing import Dict, List, Optional

EPS = 1e-9

def _is_raise(ev) -> bool:
    return ev["act"] in (Act.RAISE.value, Act.ALL_IN.value)

def _to_bb(ev) -> float:
    # for raises we store the "to" size in amount_bb; default to 0.0 if missing
    return float(ev.get("amount_bb") or 0.0)

def _next_distinct_raise(pre, start_idx: int, last_raiser: str, last_to_bb: float):
    """
    Find next raise AFTER start_idx where:
      - actor != last_raiser
      - raise-to size increases: to_bb > last_to_bb + EPS
    Returns (idx, actor, to_bb) or (None, None, None)
    """
    for i in range(start_idx + 1, len(pre)):
        ev = pre[i]
        if not _is_raise(ev):
            continue
        actor = ev["actor"]
        to_bb = _to_bb(ev)
        if actor == last_raiser:
            continue
        if to_bb <= last_to_bb + EPS:
            continue
        return i, actor, to_bb
    return None, None, None

def _seat_to_player_map(hand) -> Dict[int, str]:
    return {s["seat_number"]: s["player_id"] for s in hand["seats"]}

def _seat_order_after(player_id: str, hand) -> List[str]:
    """Clockwise player order starting *after* player_id, wrapping, stopping before player_id."""
    seat2pl = _seat_to_player_map(hand)
    ring = sorted(seat2pl.keys())
    # find player's seat
    pl_seat = None
    for seat, pid in seat2pl.items():
        if pid == player_id:
            pl_seat = seat
            break
    if pl_seat is None:
        return []

    # rotate to start after pl_seat
    idx = ring.index(pl_seat)
    rotated = ring[idx+1:] + ring[:idx+1]  # ... then player_id seat is last
    rotated = rotated[:-1]  # drop the player_id seat itself
    return [seat2pl[s] for s in rotated]

def _pos_of(hand, player_id: str) -> Optional[int]:
    """Return Pos enum id for player or None."""
    obj = hand["position_by_player"].get(player_id)
    if obj is None:
        return None
    # supports both {id:int,name:str} or raw int (if you change later)
    return obj["id"] if isinstance(obj, dict) else int(obj)

def _pos_name_of(hand, player_id: str) -> Optional[str]:
    """Return position name like 'UTG','HJ','CO','BTN','SB','BB' or None."""
    obj = hand["position_by_player"].get(player_id)
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("name")
    # if you ever store only ids, map id->name here
    return None


def build_preflop_decision_rows(hand, infer_silent_folds: bool = True) -> List[dict]:
    """
    Emit decision rows for preflop nodes:
      - OPEN (hero acts before any raise; usually folds)
      - VS_OPEN (facing the opener)
      - VS_3BET (facing a 3-bet)
      - VS_4BET (optional; facing a 4-bet)
      - LIMPED_SINGLE / LIMPED_MULTI (iso/over-limp decisions before any raise)

    Each row also carries 'raises_before_hero' for easier auditing.
    """
    rows: List[dict] = []

    if not hand.get("position_by_player"):
        return rows

    valid_players = set(hand["position_by_player"].keys())
    pre = [e for e in hand["actions"]
           if e["street"] == Street.PREFLOP.value and e["actor"] in valid_players]
    if not pre:
        return rows

    hand_id   = hand["hand_id"]
    stakes_id = hand["stakes"]["id"]
    street_id = Street.PREFLOP.value

    # ---------- LIMPED POTS (CALL before any RAISE) ----------
    first_call_idx = next((i for i, e in enumerate(pre) if e["act"] == Act.CALL.value), None)
    first_raise_idx = next((i for i, e in enumerate(pre) if _is_raise(e)), None)

    if first_call_idx is not None and (first_raise_idx is None or first_call_idx < first_raise_idx):
        # Window ends at the first raise (if any); otherwise end of preflop
        end_idx = first_raise_idx if first_raise_idx is not None else len(pre)

        # Count limpers strictly before the first raise
        n_limpers = sum(1 for e in pre[:end_idx] if e["act"] == Act.CALL.value)

        # First limper (villain for these decisions)
        limper0 = pre[first_call_idx]["actor"]
        limper0_pos = _pos_of(hand, limper0)
        if limper0_pos is None:
            return rows

        ctx_id = Ctx.LIMPED_SINGLE.value if n_limpers == 1 else Ctx.LIMPED_MULTI.value
        base_flag = Flag.SINGLEWAY.value if n_limpers == 1 else Flag.MULTIWAY.value

        # Emit ONLY explicit actions that occur before the first raise.
        # If there is NO raise at all (pure limped pot), then we may optionally
        # infer silent folds for players who never acted in the limped round.
        explicit_window = pre[first_call_idx + 1: end_idx]
        actors_with_ev = {e["actor"] for e in explicit_window}

        def _emit(hero_id: str, ev: dict | None):
            hero_pos = _pos_of(hand, hero_id)
            if hero_pos is None:
                return
            act_id = ev["act"] if ev else Act.FOLD.value
            amt = ev.get("amount_bb") if ev else None
            rows.append({
                "hand_id": hand_id,
                "stakes_id": stakes_id,
                "street_id": street_id,
                "ctx_id": ctx_id,
                "flag_id": base_flag,
                "hero_pos_id": hero_pos,
                "villain_pos_id": limper0_pos,  # facing first limper
                "act_id": act_id,
                "amount_bb": amt,
                "n_limpers": int(n_limpers),
                "raises_before_hero": 0,  # by definition within limped window
            })

        # 1) Always include players who ACTUALLY acted before the raise (over-limp or iso)
        for ev in explicit_window:
            if ev["actor"] == limper0:
                continue
            _emit(ev["actor"], ev)

        # 2) Only if NO raise exists at all, optionally infer silent folds for
        #    the rest of the table after the first limper.
        if first_raise_idx is None and infer_silent_folds:
            for hero in _seat_order_after(limper0, hand):
                if hero == limper0 or hero in actors_with_ev:
                    continue
                _emit(hero, None)

        # Do not also build VS_OPEN/VS_3BET/VS_4BET for limped pots
        return rows

    # ---------- NO LIMP BEFORE RAISE: HANDLE OPEN + VS_OPEN (+ 3/4bet) ----------
    if first_raise_idx is None:
        # no open at all (walks / folds to BB). We could emit OPEN for those who folded,
        # but typically we skip entirely in v1.
        return rows

    i_open    = first_raise_idx
    opener_ev = pre[i_open]
    opener    = opener_ev["actor"]
    open_to   = _to_bb(opener_ev)

    opener_pos     = _pos_of(hand, opener)
    opener_posname = _pos_name_of(hand, opener)
    if opener_pos is None:
        return rows

    # (A) Emit OPEN rows for anyone who acted BEFORE the first raise.
    # In this branch (no limp), events before i_open should be folds.
    for ev in pre[:i_open]:
        hero = ev["actor"]
        if ev["act"] != Act.FOLD.value:
            # Defensive: if you ever see CALL/RAISE here, your upstream normalization is off.
            continue
        hero_pos = _pos_of(hand, hero)
        if hero_pos is None:
            continue
        rows.append({
            "hand_id": hand_id,
            "stakes_id": stakes_id,
            "street_id": street_id,
            "ctx_id": Ctx.OPEN.value,          # <-- key fix
            "flag_id": Flag.SINGLEWAY.value,   # before any callers exist
            "hero_pos_id": hero_pos,
            "villain_pos_id": opener_pos,      # tie to eventual opener for grouping
            "act_id": Act.FOLD.value,
            "amount_bb": None,
            "n_limpers": 0,
            "raises_before_hero": 0,
        })

    # (B) VS_OPEN responders: from after opener until first distinct 3-bet (if any)
    i_3b, three_bettor, three_to = _next_distinct_raise(pre, i_open, opener, open_to)
    end_idx_open = i_3b if i_3b is not None else len(pre)
    cold_call_seen = False

    # resolve opener pos/name once
    opener_posname = _pos_name_of(hand, opener)  # e.g., "UTG","HJ","CO","BTN","SB","BB"
    opener_is_lp = opener_posname in {"CO", "BTN", "SB"}  # SB steal vs BB included

    for hero in _seat_order_after(opener, hand):
        hero_pos = _pos_of(hand, hero)
        if hero_pos is None:
            continue

        # find hero's first explicit action in the VS_OPEN window
        ev = next((e for e in pre[i_open + 1:end_idx_open] if e["actor"] == hero), None)

        # Only infer silent folds if there is NO 3-bet; otherwise skip missing acts
        if ev is None:
            if i_3b is None and infer_silent_folds:
                act_id, amt = Act.FOLD.value, None
            else:
                continue
        else:
            act_id, amt = ev["act"], ev.get("amount_bb")

        # Blind-vs-steal refinement: hero in blinds, opener in LP, and no cold call yet
        hero_posname = _pos_name_of(hand, hero)
        hero_is_blind = hero_posname in {"SB", "BB"}
        no_cold_call = not cold_call_seen
        ctx_id = (Ctx.BLIND_VS_STEAL.value
                  if (hero_is_blind and opener_is_lp and no_cold_call)
                  else Ctx.VS_OPEN.value)

        flag_id = Flag.MULTIWAY.value if cold_call_seen else Flag.SINGLEWAY.value

        rows.append({
            "hand_id": hand_id,
            "stakes_id": stakes_id,
            "street_id": street_id,
            "ctx_id": ctx_id,
            "flag_id": flag_id,
            "hero_pos_id": hero_pos,
            "villain_pos_id": opener_pos,
            "act_id": act_id,
            "amount_bb": amt,
            "n_limpers": 0,  # by definition in open-raise tree
            "raises_before_hero": 1,  # facing the open
        })

        if act_id == Act.CALL.value:
            cold_call_seen = True

    # (C) VS_3BET responders (if any)
    if i_3b is not None and three_bettor is not None:
        three_pos = _pos_of(hand, three_bettor)
        if three_pos is not None:
            # bound VS_3BET by first distinct 4-bet (if any)
            i_4b, four_bettor, four_to = _next_distinct_raise(pre, i_3b, three_bettor, three_to)
            end_idx_3b = i_4b if i_4b is not None else len(pre)

            cold_call_seen = False
            for hero in _seat_order_after(three_bettor, hand):
                if hero == three_bettor:
                    continue  # bettor is not a responder in this node

                hero_pos = _pos_of(hand, hero)
                if hero_pos is None:
                    continue

                # hero's first explicit action in the 3-bet window
                ev = next((e for e in pre[i_3b + 1:end_idx_3b] if e["actor"] == hero), None)

                # If a downstream raise (4-bet) exists, DO NOT infer silent folds
                if ev is None:
                    if i_4b is None and infer_silent_folds:
                        act_id, amt = Act.FOLD.value, None
                    else:
                        continue
                else:
                    act_id, amt = ev["act"], ev.get("amount_bb")

                flag_id = Flag.MULTIWAY.value if cold_call_seen else Flag.SINGLEWAY.value

                rows.append({
                    "hand_id": hand_id,
                    "stakes_id": stakes_id,
                    "street_id": street_id,
                    "ctx_id": Ctx.VS_3BET.value,
                    "flag_id": flag_id,
                    "hero_pos_id": hero_pos,
                    "villain_pos_id": three_pos,  # facing the 3-bettor
                    "act_id": act_id,
                    "amount_bb": amt,
                    "n_limpers": 0,
                    "raises_before_hero": 2,  # open + 3-bet
                })

                if act_id == Act.CALL.value:
                    cold_call_seen = True

            # (D) VS_4BET responders (optional)
            if i_4b is not None and four_bettor is not None:
                four_pos = _pos_of(hand, four_bettor)
                if four_pos is not None:
                    # bound VS_4BET by first distinct 5-bet (if any)
                    i_5b, _, _ = _next_distinct_raise(pre, i_4b, four_bettor, four_to)
                    end_idx_4b = i_5b if i_5b is not None else len(pre)

                    cold_call_seen = False
                    for hero in _seat_order_after(four_bettor, hand):
                        if hero == four_bettor:
                            continue  # bettor is not a responder in this node

                        hero_pos = _pos_of(hand, hero)
                        if hero_pos is None:
                            continue

                        ev = next((e for e in pre[i_4b + 1:end_idx_4b] if e["actor"] == hero), None)

                        # If a downstream raise (5-bet) exists, DO NOT infer silent folds
                        if ev is None:
                            if i_5b is None and infer_silent_folds:
                                act_id, amt = Act.FOLD.value, None
                            else:
                                continue
                        else:
                            act_id, amt = ev["act"], ev.get("amount_bb")

                        flag_id = Flag.MULTIWAY.value if cold_call_seen else Flag.SINGLEWAY.value

                        rows.append({
                            "hand_id": hand_id,
                            "stakes_id": stakes_id,
                            "street_id": street_id,
                            "ctx_id": Ctx.VS_4BET.value,
                            "flag_id": flag_id,
                            "hero_pos_id": hero_pos,
                            "villain_pos_id": four_pos,  # facing the 4-bettor
                            "act_id": act_id,
                            "amount_bb": amt,
                            "n_limpers": 0,
                            "raises_before_hero": 3,  # open + 3-bet + 4-bet
                        })

                        if act_id == Act.CALL.value:
                            cold_call_seen = True

    return rows