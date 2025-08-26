import argparse
import gzip
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Optional

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.datasets.parsers.parse_hand_history.build_postflop_decisions import build_postflop_decision_rows
from ml.datasets.parsers.parse_hand_history.build_preflop_decisions import build_preflop_decision_rows
from ml.core.types import Act, Street, Pos, Stakes, POSITIONS, Ctx
from infra.storage.s3_client import S3Client
from ml.datasets.parsers.parse_hand_history.regex_patterns import FLOP_RE, TURN_RE, RIVER_RE, TABLE_RE, SEATLINE_RE, HOLE_CARDS_RE, \
    SHOWDOWN_RE, HAND_SPLIT_RE, BB_RE, ACTION_RAISE, ACTION_BET, ACTION_CALL, ACTION_CHECK, ACTION_FOLD
from ml.core.hand_schema import RawSeat, HandSchema, StakeObj

s3 = S3Client()

def parse_stakes(header_line: str) -> tuple[float, float, Optional[int]]:
    """
    Parse small blind and big blind from header line.
    Also map to Stakes enum if possible.

    Returns:
        (sb, bb, stake_enum_id) where stake_enum_id is from Stakes
        or None if not matched.
    """
    m = BB_RE.search(header_line)
    if not m:
        return (0.0, 0.0, None)

    sb = float(m.group(1).replace("$", ""))
    bb = float(m.group(2).replace("$", ""))

    # Map to enum
    stake_enum = None
    try:
        stake_enum = Stakes[f"NL{int(bb*100)}"].value
        # e.g. bb=0.10 → "NL10" → Stakes.NL10.value
    except KeyError:
        pass  # if not in enum, just leave None

    return sb, bb, stake_enum

def seat_to_position(button_seat: int, seats: list[int]) -> dict[int, int]:
    n = len(seats)
    if n < 2 or n > 6:
        # log + return empty mapping so the parser can decide
        return {}

    order = POSITIONS[n]  # list of Pos enums
    ring = sorted(seats)

    if button_seat not in ring:
        return {}

    # Rotate seat order so SB starts after BTN
    btn_idx = ring.index(button_seat)
    rotated = ring[btn_idx+1:] + ring[:btn_idx+1]

    # Map seat → position (using Pos.value for compact int representation)
    return {seat: order[i].value for i, seat in enumerate(rotated)}

def norm_amt_to_bb(s: str|float, bb: float) -> float:
    if isinstance(s, (int,float)): return round(float(s)/bb, 4) if bb>0 else 0.0
    return round(float(s.replace("$",""))/bb, 4) if bb>0 else 0.0

def extract_actions(lines: list[str], bb: float) -> list[dict]:
    street = Street.PREFLOP.value
    events: list[dict] = []
    for line in lines:
        line = line.strip()

        # street switches
        if line.startswith("*** FLOP ***"):  street = Street.FLOP.value;  continue
        if line.startswith("*** TURN ***"):  street = Street.TURN.value;  continue
        if line.startswith("*** RIVER ***"): street = Street.RIVER.value; continue
        if line.startswith("*** SHOW DOWN ***") or line.startswith("*** SUMMARY ***"):
            break

        # normalize actions
        m = ACTION_RAISE.match(line)
        if m:
            p, by_amt, to_amt = m.group(1).strip(), float(m.group(2)), float(m.group(3))
            events.append({
                "street": street, "actor": p, "act": Act.RAISE.value,
                "amount_bb": norm_amt_to_bb(to_amt, bb)  # <— use the “to” size
            })
            continue

        m = ACTION_BET.match(line)
        if m:
            p, amt = m.group(1).strip(), float(m.group(2))
            events.append({
                "street": street,
                "actor": p,
                "act": Act.BET.value,
                "amount_bb": norm_amt_to_bb(amt, bb),
            })
            continue

        m = ACTION_CALL.match(line)
        if m:
            p, amt = m.group(1).strip(), float(m.group(2))
            events.append({
                "street": street,
                "actor": p,
                "act": Act.CALL.value,
                "amount_bb": norm_amt_to_bb(amt, bb),
            })
            continue

        m = ACTION_CHECK.match(line)
        if m:
            p = m.group(1).strip()
            events.append({
                "street": street,
                "actor": p,
                "act": Act.CHECK.value,
            })
            continue

        m = ACTION_FOLD.match(line)
        if m:
            p = m.group(1).strip()
            events.append({
                "street": street,
                "actor": p,
                "act": Act.FOLD.value,
            })
            continue

    return events


def determine_street(board_cards: list[str]) -> int:
    if len(board_cards) == 5:
        return Street.RIVER.value
    if len(board_cards) == 4:
        return Street.TURN.value
    if len(board_cards) == 3:
        return Street.FLOP.value
    return Street.PREFLOP.value

def extract_board(lines: list[str]) -> list[str]:
    board: list[str] = []
    for line in lines:
        if (m := FLOP_RE.search(line)):
            board += m.group(1).split()
        if (m := TURN_RE.search(line)):
            board.append(m.group(1))
        if (m := RIVER_RE.search(line)):
            board.append(m.group(1))
    return board

# ── PARSER ─────────────────────────────────────────────────────────────────────

def parse_hand_block(text: str, stake_obj: StakeObj) -> dict:
    lines = [ln.strip("\ufeff") for ln in text.strip().splitlines()]  # strip BOMs
    hand_raw = {
        "hand_id": "",
        "table_name": None,
        "stakes": stake_obj.dict(),
        "button_seat": None,
        "seats": [],
        "position_by_player": {},          # NEW
        "street": "preflop",
        "hero": None,
        "hole_cards_by_player": {},
        "board": [],
        "actions": [],                     # structured events (dicts)
        "players": [],
        "results_by_player": {},           # NEW net in BB (signed)
        "pot_bb": 0.0,                     # NEW
        "rake_bb": 0.0,                    # NEW
        "min_bet": 0.0,
    }

    sb, bb = 0.0, 0.0
    # 1) Header: hand id, table, button seat, stakes
    for line in lines:
        if line.startswith("PokerStars Hand #"):
            hand_raw["hand_id"] = line.split("#")[1].split(":")[0]
            sb, bb, _ = parse_stakes(line)
            hand_raw["min_bet"] = bb
        m = TABLE_RE.search(line)
        if m:
            hand_raw["table_name"] = m.group(1)
            hand_raw["button_seat"] = int(m.group(2))

    # 2) Seats
    seat_nums = []
    for line in lines:
        m = SEATLINE_RE.match(line)
        if not m: continue
        num = int(m.group(1))
        player = m.group(2).strip()
        stack = float(m.group(3)) if m.group(3) else 0.0
        status: Literal['active','sitting out'] = "sitting out" if "sit out" in line.lower() else "active"
        seat_nums.append(num)
        hand_raw["seats"].append(RawSeat(seat_number=num, player_id=player, stack_size=stack, status=status).dict())

    # 3) Positions
    pos_by_seat = seat_to_position(hand_raw["button_seat"], seat_nums)
    if not pos_by_seat:
        # skip this hand — likely incomplete
        return None
    # map seat -> player_id
    seat_to_player = {s["seat_number"]: s["player_id"] for s in hand_raw["seats"]}
    hand_raw["position_by_player"] = {
        seat_to_player[s]: {
            "id": pos,
            "name": Pos(pos).name
        }
        for s, pos in pos_by_seat.items()
        if s in seat_to_player
    }
    # 4) Hole cards (hero or showdown)
    for line in lines:
        m = HOLE_CARDS_RE.search(line)
        if m:
            player, c1, c2 = m.group(1).strip(), m.group(2), m.group(3)
            hand_raw["hole_cards_by_player"][player] = [c1, c2]
            hand_raw["hero"] = player
    for line in lines:
        m = SHOWDOWN_RE.match(line)
        if m:
            player, c1, c2 = m.group(1).strip(), m.group(2), m.group(3)
            hand_raw["hole_cards_by_player"].setdefault(player, [c1, c2])

    # 5) Board & street
    hand_raw["board"] = extract_board(lines)
    hand_raw["street"] = determine_street(hand_raw["board"])

    # 6) Actions (structured)
    acts = extract_actions(lines, bb)
    valid_players = set(hand_raw["position_by_player"].keys())
    acts = [a for a in acts if a["actor"] in valid_players]
    hand_raw["actions"] = acts
    hand_raw["players"] = sorted({ev["actor"] for ev in acts})

    # 7) Pot / rake / results (from summary lines)
    pot = 0.0
    rake = 0.0

    # Try to parse from summary first
    for line in lines:
        if "Total pot" in line and "| Rake" in line:
            try:
                pot_s = line.split("Total pot")[1].split("|")[0].strip().replace("$", "")
                rake_s = line.split("| Rake")[1].strip().replace("$", "")
                pot = float(pot_s)
                rake = float(rake_s)
            except Exception:
                pass
            break  # only need first match

    # Normalize
    hand_raw["pot_bb"] = norm_amt_to_bb(pot, bb)
    hand_raw["rake_bb"] = norm_amt_to_bb(rake, bb)

    # ---- FALLBACK if missing ----
    if hand_raw["pot_bb"] == 0.0:
        if hand_raw["results_by_player"]:
            total_won = sum(v for v in hand_raw["results_by_player"].values() if v > 0)
            hand_raw["pot_bb"] = round(total_won, 2)

    # results: parse “Seat X NAME showed … and won ($Y)” lines
    for line in lines:
        if " and won ($" in line:
            try:
                # Seat 1 Aanakin57 showed [...] and won ($2.80)
                parts = line.split()
                player = parts[2].strip()
                won = float(line.split("and won (")[1].split(")")[0].replace("$",""))
                hand_raw["results_by_player"][player] = norm_amt_to_bb(won, bb)
            except Exception:
                continue

    for p in hand_raw["players"]:
        if p not in hand_raw["results_by_player"]:
            hand_raw["results_by_player"][p] = 0.0

    return HandSchema(**hand_raw).dict()


def parse_all_hands(stake: int):
    stake_enum = Stakes[f"NL{stake}"]
    stake_obj = StakeObj(id=stake_enum.value, name=stake_enum.name)

    raw_dir = Path(f"data/raw/{stake_enum.name.upper()}")

    # cleaner local filenames
    hands_jsonl = Path("hands.jsonl")
    decisions_jsonl = Path("decisions.jsonl")

    hands_gz = Path(f"{hands_jsonl}.gz")
    decisions_gz = Path(f"{decisions_jsonl}.gz")

    # S3 structure: parsed/<stake>/hands.jsonl.gz, parsed/<stake>/decisions.jsonl.gz
    stake_folder = stake_enum.name.lower()
    hands_s3_key = f"parsed/{stake_folder}/{hands_gz.name}"
    decisions_s3_key = f"parsed/{stake_folder}/{decisions_gz.name}"

    total_hands = 0
    total_rows = 0

    with hands_jsonl.open("w", encoding="utf-8") as hands_f, \
         decisions_jsonl.open("w", encoding="utf-8") as dec_f:

        for txt in raw_dir.rglob("*.txt"):
            content = txt.read_text(errors="ignore")
            for block in HAND_SPLIT_RE.split(content):
                if not block.strip():
                    continue

                hand = parse_hand_block(block, stake_obj)
                if not hand:
                    continue  # skip malformed

                # write the raw hand (hand-level JSON)
                json.dump(hand, hands_f, separators=(",", ":"))
                hands_f.write("\n")
                total_hands += 1

                # build decision rows (flattened, ML-ready)
                rows_pre = build_preflop_decision_rows(hand, infer_silent_folds=True)
                rows_post = build_postflop_decision_rows(hand)

                for r in rows_pre + rows_post:
                    if r["act_id"] == Act.ALL_IN.value:
                        r["act_id"] = Act.RAISE.value
                    json.dump(r, dec_f, separators=(",", ":"))
                    dec_f.write("\n")
                    total_rows += 1

    # Compress and upload both files
    with hands_jsonl.open("rb") as f_in, gzip.open(hands_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    with decisions_jsonl.open("rb") as f_in, gzip.open(decisions_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    s3.upload_file(hands_gz, hands_s3_key)
    s3.upload_file(decisions_gz, decisions_s3_key)

    print(f"✅ Parsed {total_hands} hands → s3://{s3.bucket}/{hands_s3_key}")
    print(f"✅ Emitted {total_rows} decision rows → s3://{s3.bucket}/{decisions_s3_key}")

    # ---- Context coverage check (pre + postflop) ----
    CTX_NAME = {
        Ctx.OPEN.value: "OPEN",
        Ctx.VS_OPEN.value: "VS_OPEN",
        Ctx.VS_3BET.value: "VS_3BET",
        Ctx.VS_4BET.value: "VS_4BET",
        Ctx.VS_CBET.value: "VS_CBET",
        Ctx.VS_DONK.value: "VS_DONK",
        Ctx.VS_CHECK_RAISE.value: "VS_CHECK_RAISE",
    }
    CTX_NAME.update({
        Ctx.LIMPED_SINGLE.value: "LIMPED_SINGLE",
        Ctx.LIMPED_MULTI.value: "LIMPED_MULTI",
    })
    STREET_NAME = {
        Street.PREFLOP.value: "PREFLOP",
        Street.FLOP.value: "FLOP",
        Street.TURN.value: "TURN",
        Street.RIVER.value: "RIVER",
    }

    counts_ctx = Counter()
    counts_ctx_street = defaultdict(Counter)

    with gzip.open(decisions_gz, "rt") as f:
        for line in f:
            r = json.loads(line)
            ctx = r.get("ctx_id")
            st = r.get("street_id")
            if ctx is None or st is None:
                continue
            counts_ctx[ctx] += 1
            counts_ctx_street[ctx][st] += 1

    print("\nContext totals:")
    for ctx_id, n in sorted(counts_ctx.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {CTX_NAME.get(ctx_id, str(ctx_id))}: {n}")

    print("\nContext × Street breakdown:")
    for ctx_id, by_st in counts_ctx_street.items():
        ctx_label = CTX_NAME.get(ctx_id, str(ctx_id))
        parts = " | ".join(f"{STREET_NAME.get(st, st)}={cnt}"
                           for st, cnt in sorted(by_st.items()))
        print(f"  {ctx_label}: {parts}")

# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stake", type=int, required=True, help="Stake in big blind cents, e.g. 10 for NL10")
    args = parser.parse_args()
    parse_all_hands(args.stake)