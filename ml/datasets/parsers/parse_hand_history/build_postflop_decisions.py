from typing import List, Dict, Optional

from ml.core.types import Street, Act, Ctx, Flag


def _pos_of(hand, pid: str) -> Optional[int]:
    obj = hand["position_by_player"].get(pid)
    return obj["id"] if isinstance(obj, dict) else (int(obj) if obj is not None else None)

def _players_alive_before_street(hand, street_id: int) -> List[str]:
    """Players who have not folded before <street_id> starts."""
    alive = set(hand["position_by_player"].keys())
    for e in hand["actions"]:
        if e["street"] >= street_id:  # only consider folds before this street
            continue
        if e["act"] == Act.FOLD.value and e["actor"] in alive:
            alive.remove(e["actor"])
    # Return in table (seat) order for consistency
    seat2pl = {s["seat_number"]: s["player_id"] for s in hand["seats"] if s["player_id"] in alive}
    return [seat2pl[s] for s in sorted(seat2pl.keys())]

def _preflop_aggressor(hand) -> Optional[str]:
    """Last raiser or all-in on preflop; None if no raise."""
    last = None
    for e in hand["actions"]:
        if e["street"] == Street.PREFLOP.value and e["act"] in (Act.RAISE.value, Act.ALL_IN.value):
            last = e["actor"]
    return last

def build_postflop_decision_rows(hand) -> List[dict]:
    """
    Emit postflop decision rows for:
      - VS_CBET: facing an aggressor's first bet on the street
      - VS_DONK: facing a donk (bet into the preflop aggressor before they act)
      - VS_CHECK_RAISE: the initial bettor's response after getting raised

    Notes:
      - We do NOT infer silent folds postflop: only explicit actions become rows.
      - We bound responder windows from the first bet to the first raise (if any).
      - For VS_CHECK_RAISE we look for the bettor's first action AFTER the raise and emit that.
    """
    rows: List[dict] = []
    hand_id = hand["hand_id"]
    stakes_id = hand["stakes"]["id"]

    agg = _preflop_aggressor(hand)  # player_id with initiative, or None
    if agg is None:
        return rows  # no initiative → skip postflop contexts for v1

    valid_players = set(hand["position_by_player"].keys())

    # helper: surviving players when street starts (by seat-number order)
    def _alive_order_for_street(street_id: int) -> List[str]:
        alive = _players_alive_before_street(hand, street_id)
        seat2pl = {s["seat_number"]: s["player_id"]
                   for s in hand["seats"] if s["player_id"] in alive}
        return [seat2pl[s] for s in sorted(seat2pl.keys())]

    for street_id in (Street.FLOP.value, Street.TURN.value, Street.RIVER.value):
        events = [e for e in hand["actions"]
                  if e["street"] == street_id and e["actor"] in valid_players]
        if not events:
            continue

        order = _alive_order_for_street(street_id)
        if agg not in order:
            # preflop aggressor is no longer in the hand
            continue

        # Find the first aggressive bet on this street
        first_bet_idx = next(
            (i for i, e in enumerate(events)
             if e["act"] in (Act.BET.value, Act.RAISE.value, Act.ALL_IN.value)),
            None
        )
        if first_bet_idx is None:
            # no betting occurred (check-through street)
            continue

        first_bet = events[first_bet_idx]
        bettor = first_bet["actor"]

        # Donk/cbet classification
        is_cbet = (bettor == agg)
        # donk = a player other than agg puts in first bet BEFORE agg has acted on this street
        agg_acted_before_bet = any(ev["actor"] == agg for ev in events[:first_bet_idx])
        is_donk = (not is_cbet) and (not agg_acted_before_bet)

        # Bound responders: until the first raise after the initial bet (if any)
        next_raise_idx = next(
            (i for i in range(first_bet_idx + 1, len(events))
             if events[i]["act"] in (Act.RAISE.value, Act.ALL_IN.value)),
            None
        )
        end_idx = next_raise_idx if next_raise_idx is not None else len(events)

        ctx_id = None
        if is_cbet:
            ctx_id = Ctx.VS_CBET.value
        elif is_donk:
            ctx_id = Ctx.VS_DONK.value
        # else: skip labeling for other first-bet patterns in v1

        if ctx_id is not None:
            villain_pos = _pos_of(hand, bettor)
            if villain_pos is not None:
                flag_id = Flag.MULTIWAY.value if len(order) > 2 else Flag.SINGLEWAY.value
                # Emit explicit responders between first bet and first raise/end
                for ev in events[first_bet_idx + 1:end_idx]:
                    hero = ev["actor"]
                    if hero == bettor:
                        continue
                    hero_pos = _pos_of(hand, hero)
                    if hero_pos is None:
                        continue
                    rows.append({
                        "hand_id": hand_id,
                        "stakes_id": stakes_id,
                        "street_id": street_id,
                        "ctx_id": ctx_id,
                        "flag_id": flag_id,
                        "hero_pos_id": hero_pos,
                        "villain_pos_id": villain_pos,   # the bettor you're facing
                        "act_id": ev["act"],             # hero's action
                        "amount_bb": ev.get("amount_bb"),
                    })

        # VS_CHECK_RAISE: if there was a raise over the initial bet, record the **bettor's response**
        if next_raise_idx is not None:
            initial_bettor = bettor
            raiser = events[next_raise_idx]["actor"]

            # Find the initial bettor's first action AFTER the raise within this street
            reply_ev = next(
                (e for e in events[next_raise_idx + 1:]
                 if e["actor"] == initial_bettor),
                None
            )
            if reply_ev is not None:
                init_pos = _pos_of(hand, initial_bettor)
                raiser_pos = _pos_of(hand, raiser)
                if init_pos is not None and raiser_pos is not None:
                    rows.append({
                        "hand_id": hand_id,
                        "stakes_id": stakes_id,
                        "street_id": street_id,
                        "ctx_id": Ctx.VS_CHECK_RAISE.value,
                        "flag_id": Flag.MULTIWAY.value if len(order) > 2 else Flag.SINGLEWAY.value,
                        "hero_pos_id": init_pos,          # bettor now facing a raise
                        "villain_pos_id": raiser_pos,     # raiser
                        "act_id": reply_ev["act"],        # bettor's response (fold/call/raise)
                        "amount_bb": reply_ev.get("amount_bb"),
                    })

    return rows