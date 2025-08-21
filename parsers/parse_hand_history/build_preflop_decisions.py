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

def build_preflop_decision_rows(hand, infer_silent_folds: bool = True) -> List[dict]:
    """
    Emit decision rows for preflop nodes:
      - VS_OPEN (facing opener)
      - VS_3BET (facing 3-bettor)
    VS_4BET can be added later with same pattern.

    Output row schema:
      {
        "stakes_id": int,
        "street_id": int,
        "ctx_id": int,
        "flag_id": int,              # SINGLEWAY / MULTIWAY
        "hero_pos_id": int,
        "villain_pos_id": int,       # opener for VS_OPEN, 3-bettor for VS_3BET
        "act_id": int,               # FOLD/CALL/RAISE/ALL_IN
        "amount_bb": float | None,
      }
    """
    rows: List[dict] = []

    # Guard: need positions and preflop actions
    if not hand.get("position_by_player"):
        return rows

    valid_players = set(hand["position_by_player"].keys())
    pre = [e for e in hand["actions"] if e["street"] == Street.PREFLOP.value and e["actor"] in valid_players]
    stakes_id = hand["stakes"]["id"]
    street_id = Street.PREFLOP.value

    # ---------- PREFLOP NODES (VS_OPEN / VS_3BET / VS_4BET) ----------

    # filter preflop events to valid players already done: pre = [...]
    if not pre:
        return rows

    # Limp detection: any CALL before any RAISE ⇒ skip preflop for v1
    first_call_idx = next((i for i, e in enumerate(pre) if e["act"] == Act.CALL.value), None)
    first_raise_idx = next((i for i, e in enumerate(pre) if _is_raise(e)), None)
    if first_call_idx is not None and (first_raise_idx is None or first_call_idx < first_raise_idx):
        # Count limpers before any raise
        n_limpers = sum(1 for e in pre[: (first_raise_idx if first_raise_idx is not None else len(pre))]
                        if e["act"] == Act.CALL.value)

        limper0 = pre[first_call_idx]["actor"]  # the first limper
        limper0_pos = _pos_of(hand, limper0)
        if limper0_pos is None:
            return rows

        # Window ends at the first raise (if any); otherwise end of preflop
        end_idx = first_raise_idx if first_raise_idx is not None else len(pre)

        # Choose context by number of limpers
        ctx_id = Ctx.LIMPED_SINGLE.value if n_limpers == 1 else Ctx.LIMPED_MULTI.value
        # Flag: if only 1 limper so far → SINGLEWAY; if ≥2 limpers → MULTIWAY
        base_flag = Flag.SINGLEWAY.value if n_limpers == 1 else Flag.MULTIWAY.value

        # Emit one row per responder after the first limper up to the raise/end
        # (Responders can fold, over‑limp (CALL), or iso‑raise (RAISE/ALL_IN))
        for hero in _seat_order_after(limper0, hand):
            hero_pos = _pos_of(hand, hero)
            if hero_pos is None:
                continue
            ev = next((e for e in pre[first_call_idx + 1:end_idx] if e["actor"] == hero), None)

            if ev is None and not infer_silent_folds:
                continue

            act_id = ev["act"] if ev else Act.FOLD.value
            amt = ev.get("amount_bb") if ev else None

            rows.append({
                "stakes_id": stakes_id,
                "street_id": street_id,
                "ctx_id": ctx_id,
                "flag_id": base_flag,  # simple v1: use base flag from n_limpers
                "hero_pos_id": hero_pos,
                "villain_pos_id": limper0_pos,  # facing the first limper
                "act_id": act_id,
                "amount_bb": amt,
                "n_limpers": int(n_limpers),  # NEW: record count of limpers
            })

        return rows  # do not also run VS_OPEN/3BET on limped pots

    # --- OPEN node info (safe) ---
    if first_raise_idx is None:
        # no open raise at all (walks / folded to BB / malformed) → nothing to emit
        return rows

    i_open = first_raise_idx
    opener_ev = pre[i_open]
    opener = opener_ev["actor"]
    open_to = _to_bb(opener_ev)  # your helper that reads the "to" size (0.0 if missing)

    opener_pos = _pos_of(hand, opener)
    if opener_pos is None:
        # actor not in position_by_player (sat out/malformed) → skip
        return rows

    # (optional, but handy if the next code needs these)
    # Find the first distinct re-raise after opener (3-bet boundary)
    i_3b, three_bettor, three_to = _next_distinct_raise(pre, i_open, opener, open_to)
    # VS_OPEN responders: from after opener until 3-bet or end of preflop
    end_idx_open = i_3b if i_3b is not None else len(pre)
    cold_call_seen = False
    for hero in _seat_order_after(opener, hand):
        hero_pos = _pos_of(hand, hero)
        if hero_pos is None:
            continue
        ev = next((e for e in pre[i_open + 1:end_idx_open] if e["actor"] == hero), None)
        if ev is None and not infer_silent_folds:
            continue
        act_id = ev["act"] if ev else Act.FOLD.value
        amt = ev.get("amount_bb") if ev else None
        flag_id = Flag.MULTIWAY.value if cold_call_seen else Flag.SINGLEWAY.value
        rows.append({
            "stakes_id": stakes_id,
            "street_id": street_id,
            "ctx_id": Ctx.VS_OPEN.value,
            "flag_id": flag_id,
            "hero_pos_id": hero_pos,
            "villain_pos_id": opener_pos,
            "act_id": act_id,
            "amount_bb": amt,
        })
        if act_id == Act.CALL.value:
            cold_call_seen = True

    # --- VS_3BET node (if exists) ---
    if i_3b is not None:
        three_pos = _pos_of(hand, three_bettor)
        if three_pos is not None:
            # find 4-bet ONCE (distinct raiser & bigger size)
            i_4b, four_bettor, four_to = _next_distinct_raise(pre, i_3b, three_bettor, three_to)
            end_idx_3b = i_4b if i_4b is not None else len(pre)

            cold_call_seen = False
            for hero in _seat_order_after(three_bettor, hand):
                hero_pos = _pos_of(hand, hero)
                if hero_pos is None:
                    continue
                ev = next((e for e in pre[i_3b + 1:end_idx_3b] if e["actor"] == hero), None)
                if ev is None and not infer_silent_folds:
                    continue
                act_id = ev["act"] if ev else Act.FOLD.value
                amt = ev.get("amount_bb") if ev else None
                flag_id = Flag.MULTIWAY.value if cold_call_seen else Flag.SINGLEWAY.value
                rows.append({
                    "stakes_id": stakes_id,
                    "street_id": street_id,
                    "ctx_id": Ctx.VS_3BET.value,
                    "flag_id": flag_id,
                    "hero_pos_id": hero_pos,
                    "villain_pos_id": three_pos,
                    "act_id": act_id,
                    "amount_bb": amt,
                })
                if act_id == Act.CALL.value:
                    cold_call_seen = True

            # --- VS_4BET node (optional) ---
            if i_4b is not None:
                four_pos = _pos_of(hand, four_bettor)
                if four_pos is not None:
                    # we stop at 5-bet; compute end bound once
                    i_5b, _, _ = _next_distinct_raise(pre, i_4b, four_bettor, four_to)
                    end_idx_4b = i_5b if i_5b is not None else len(pre)

                    cold_call_seen = False
                    for hero in _seat_order_after(four_bettor, hand):
                        hero_pos = _pos_of(hand, hero)
                        if hero_pos is None:
                            continue
                        ev = next((e for e in pre[i_4b + 1:end_idx_4b] if e["actor"] == hero), None)
                        if ev is None and not infer_silent_folds:
                            continue
                        act_id = ev["act"] if ev else Act.FOLD.value
                        amt = ev.get("amount_bb") if ev else None
                        flag_id = Flag.MULTIWAY.value if cold_call_seen else Flag.SINGLEWAY.value
                        rows.append({
                            "stakes_id": stakes_id,
                            "street_id": street_id,
                            "ctx_id": Ctx.VS_4BET.value,
                            "flag_id": flag_id,
                            "hero_pos_id": hero_pos,
                            "villain_pos_id": four_pos,
                            "act_id": act_id,
                            "amount_bb": amt,
                        })
                        if act_id == Act.CALL.value:
                            cold_call_seen = True
    # ---------- end preflop ----------

    return rows