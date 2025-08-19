# ml/etl/export_popnet_rows.py
from __future__ import annotations
import gzip, json, random, math
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.schema.pop_net_schema import PopNetSample

IN_PATH  = Path("data/parsed/hands_NL10.jsonl.gz")      # <- set per stake/profile
OUT_PATH = Path("data/population/popnet.v1.jsonl.gz")

def _expand_hand_to_decision_rows(hand: dict) -> List[dict]:
    """
    Turn one parsed PokerStars hand (your normalized schema) into decision-level rows.
    Emits [{ "x": PopNetFeatures, "y": PopNetLabel }, ...].
    """
    rows: List[dict] = []

    stake_tag: str = hand.get("stake_tag", "NLx")
    positions: Dict[str,str] = hand.get("position_by_player", {})
    btn_seat = hand.get("button_seat")
    table_name = hand.get("table_name")
    stakes = hand.get("stakes")

    # stacks in BB (approx; from Seat lines at hand start)
    bb_size = float(hand.get("min_bet", 0.0)) or 0.0
    stacks_bb: Dict[str,float] = {}
    for s in hand.get("seats", []):
        pid = s["player_id"]
        stacks_bb[pid] = (float(s.get("stack_size", 0.0)) / bb_size) if bb_size > 0 else 0.0

    # very light pot tracker by street (approx)
    street = "preflop"
    pot_bb = 0.0
    last_bet_bb = None
    min_raise_to_bb = None
    bets_this, raises_this, checks_this, calls_this = 0, 0, 0, 0

    def reset_street_counters(new_street: str):
        nonlocal street, last_bet_bb, min_raise_to_bb, bets_this, raises_this, checks_this, calls_this
        street = new_street
        last_bet_bb = None
        min_raise_to_bb = None
        bets_this = raises_this = checks_this = calls_this = 0

    # preflop blinds (coarse): if we see a raise to >= 2bb, assume blinds in pot
    seen_any_preflop_raise = False

    for ev in hand.get("actions", []):
        ev_street = ev.get("street", street)
        if ev_street != street:
            reset_street_counters(ev_street)

        actor = ev.get("actor")
        act   = ev.get("act")
        amt   = ev.get("amount_bb", None)  # your normalized field
        facing_to_call = float(last_bet_bb or 0.0)

        # Quick-and-dirty pot update
        if street == "preflop" and act in ("raise", "bet") and not seen_any_preflop_raise:
            pot_bb += 1.5
            seen_any_preflop_raise = True

        if act == "bet" and amt is not None:
            pot_bb += float(amt)
            last_bet_bb = float(amt)
            bets_this += 1
            min_raise_to_bb = (last_bet_bb * 2.0)

        elif act == "raise" and amt is not None:
            if last_bet_bb is not None:
                pot_bb += max(0.0, float(amt) - last_bet_bb)
            else:
                pot_bb += float(amt)
            raises_this += 1
            last_bet_bb = float(amt)
            min_raise_to_bb = (last_bet_bb * 2.0)

        elif act == "call" and amt is not None:
            pot_bb += float(amt)
            calls_this += 1

        elif act == "check":
            checks_this += 1

        elif act == "fold":
            pass

        elif act == "allin" and amt is not None:
            pot_bb += float(amt)
            last_bet_bb = float(amt)

        # Build features/label for THIS decision
        if actor is None or actor not in positions:
            continue  # skip malformed rows

        actor_pos = positions[actor]

        # very rough effective stack: min(actor, max(other))
        actor_stack = stacks_bb.get(actor, 0.0)
        opp_max = max([stacks_bb.get(p, 0.0) for p in stacks_bb.keys() if p != actor] or [0.0])
        eff_stack = min(actor_stack, opp_max)

        # crude amount_to_call: for call acts we stored the call "to" size; otherwise 0
        amount_to_call_bb = float(amt) if act == "call" and amt is not None else 0.0

        # Flags (quick heuristics for now; refine later if needed)
        is_first_in = (street == "preflop" and act in ("bet","raise") and bets_this == 1 and raises_this == 0)
        facing_open = (street == "preflop" and act in ("call","raise") and seen_any_preflop_raise)
        facing_3bet  = False
        facing_4bet_plus = False
        is_3bet_pot = False
        is_4bet_plus = False
        num_players = len(positions) or len(stacks_bb) or 6

        x = {
            "stake_tag": stake_tag,
            "players": num_players,
            "street": street,
            "actor": actor,
            "actor_pos": actor_pos,
            "positions": positions,
            "effective_stack_bb": max(0.0, float(eff_stack)),
            "pot_bb": max(0.0, float(pot_bb)),
            "amount_to_call_bb": max(0.0, facing_to_call),
            "is_3bet_pot": is_3bet_pot,
            "is_4bet_plus": is_4bet_plus,
            "is_first_in": is_first_in,
            "facing_open": facing_open,
            "facing_3bet": facing_3bet,
            "facing_4bet_plus": facing_4bet_plus,
            "board_cluster_id": None,
            "board_cards": hand.get("board") or None,
            "last_bet_bb": None if last_bet_bb is None else float(last_bet_bb),
            "min_raise_to_bb": None if min_raise_to_bb is None else float(min_raise_to_bb),
            "bets_this_street": int(bets_this),
            "raises_this_street": int(raises_this),
            "checks_this_street": int(checks_this),
            "calls_this_street": int(calls_this),
            "table_name": table_name,
            "stakes": stakes,
        }

        y = {
            "action": act,
            "amount_bb": None if amt is None else float(amt),
            "size_bucket": None
        }

        rows.append({"x": x, "y": y})

    return rows



def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") \
           else open(path, "rt", encoding="utf-8")

def run(in_path: Path = IN_PATH, out_path: Path = OUT_PATH):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_hands = n_kept = n_rows = n_bad = 0

    with open_text(in_path) as fin, gzip.open(out_path, "wt", encoding="utf-8") as fout:

        for ln in fin:
            if not ln.strip():
                continue
            n_hands += 1
            raw_hand = json.loads(ln)

            # Expand: hand -> iterable of decision rows (dicts with keys x,y)
            for sample in _expand_hand_to_decision_rows(raw_hand):
                # (optional) validate to catch schema mismatches early
                try:
                    PopNetSample(**sample)  # throws if incomplete
                    n_kept += 1
                except Exception:
                    n_bad += 1
                    continue

                fout.write(json.dumps(sample, separators=(",", ":")) + "\n")
                n_rows += 1

    print(f"✅ PopNet export → {out_path}")
    print(f"   hands read:     {n_hands:,}")
    print(f"   decisions ok:   {n_kept:,}")
    print(f"   decisions bad:  {n_bad:,}")
    print(f"   rows written:   {n_rows:,}")

if __name__ == "__main__":
    run()