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
    rows: List[dict] = []
    stakes_id = hand["stakes"]["id"]

    agg = _preflop_aggressor(hand)  # preflop aggressor (initiative)
    if agg is None:
        return rows  # no initiative → skip postflop contexts for v1

    valid_players = set(hand["position_by_player"].keys())

    # helper: seat order for alive players (SB..BB..UTG.. in seat number order)
    def _alive_order_for_street(street_id: int) -> List[str]:
        alive = _players_alive_before_street(hand, street_id)
        # return in seat number order
        seat2pl = {s["seat_number"]: s["player_id"] for s in hand["seats"] if s["player_id"] in alive}
        return [seat2pl[s] for s in sorted(seat2pl.keys())]

    for street_id in (Street.FLOP.value, Street.TURN.value, Street.RIVER.value):
        events = [e for e in hand["actions"] if e["street"] == street_id and e["actor"] in valid_players]
        if not events:
            continue

        order = _alive_order_for_street(street_id)
        if agg not in order:
            # aggressor folded earlier; skip this street for v1
            continue

        # find first aggressive action on the street (BET/RAISE/ALL_IN)
        first_bet_idx = next((i for i, e in enumerate(events)
                              if e["act"] in (Act.BET.value, Act.RAISE.value, Act.ALL_IN.value)), None)
        if first_bet_idx is None:
            continue

        first_bet = events[first_bet_idx]
        bettor = first_bet["actor"]

        # find first raise after that bet (for bounding responders and check-raise detection)
        next_raise_idx = next((i for i in range(first_bet_idx + 1, len(events))
                               if events[i]["act"] in (Act.RAISE.value, Act.ALL_IN.value)), None)
        end_idx = next_raise_idx if next_raise_idx is not None else len(events)

        is_cbet = (bettor == agg)
        # DONK = first bettor acts BEFORE the aggressor in seat order and the aggressor has NOT acted yet on this street
        bettor_before_agg = order.index(bettor) < order.index(agg) if (bettor in order and agg in order) else False
        agg_acted_before_bet = any(ev["actor"] == agg for ev in events[:first_bet_idx])
        is_donk = (not is_cbet) and bettor_before_agg and (not agg_acted_before_bet)

        ctx_id = None
        if is_cbet:
            ctx_id = Ctx.VS_CBET.value
        elif is_donk:
            ctx_id = Ctx.VS_DONK.value
        else:
            # Not a c-bet and not a donk (e.g., IP bet after OOP checked) → skip labeling for v1
            ctx_id = None

        if ctx_id is not None:
            villain_pos = _pos_of(hand, bettor)
            flag_id = Flag.MULTIWAY.value if len(order) > 2 else Flag.SINGLEWAY.value

            # Emit one row per explicit responder between the bet and the first raise/end
            # (we do NOT infer silent folds postflop in v1)
            for ev in events[first_bet_idx + 1:end_idx]:
                hero = ev["actor"]
                if hero == bettor:
                    continue
                hero_pos = _pos_of(hand, hero)
                if hero_pos is None:
                    continue
                rows.append({
                    "stakes_id": stakes_id,
                    "street_id": street_id,
                    "ctx_id": ctx_id,
                    "flag_id": flag_id,
                    "hero_pos_id": hero_pos,
                    "villain_pos_id": villain_pos,
                    "act_id": ev["act"],
                    "amount_bb": ev.get("amount_bb"),
                })

        # VS_CHECK_RAISE: if a raise occurs over the initial bet, record that the initial bettor is now facing a raise
        if next_raise_idx is not None:
            initial_bettor = bettor
            raiser = events[next_raise_idx]["actor"]
            init_pos = _pos_of(hand, initial_bettor)
            raiser_pos = _pos_of(hand, raiser)
            if init_pos is not None and raiser_pos is not None:
                rows.append({
                    "stakes_id": stakes_id,
                    "street_id": street_id,
                    "ctx_id": Ctx.VS_CHECK_RAISE.value,
                    "flag_id": Flag.MULTIWAY.value if len(order) > 2 else Flag.SINGLEWAY.value,
                    "hero_pos_id": init_pos,          # the bettor who got raised (now facing decision)
                    "villain_pos_id": raiser_pos,     # the raiser
                    "act_id": Act.RAISE.value,        # trigger is a raise (we can later join to the bettor's response)
                    "amount_bb": events[next_raise_idx].get("amount_bb"),
                })

    return rows