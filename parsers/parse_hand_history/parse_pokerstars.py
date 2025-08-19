import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Literal

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_uploader import S3Uploader
from parsers.parse_hand_history.regex_patterns import FLOP_RE, TURN_RE, RIVER_RE, TABLE_RE, SEATLINE_RE, HOLE_CARDS_RE, \
    SHOWDOWN_RE, HAND_SPLIT_RE, BB_RE, ACTION_RAISE, ACTION_BET, ACTION_CALL, ACTION_CHECK, ACTION_FOLD
from parsers.parse_hand_history.utils import RawSeat, HandSchema

s3 = S3Uploader()

def parse_stakes(header_line: str) -> tuple[float,float]:
    m = BB_RE.search(header_line)
    if not m: return (0.0, 0.0)
    sb = float(m.group(1).replace("$",""))
    bb = float(m.group(2).replace("$",""))
    return sb, bb



def seat_to_position_6max(button_seat: int, seats: list[int]) -> dict[int,str]:
    order = ["SB","BB","UTG","HJ","CO","BTN"]
    ring = sorted(seats)
    if button_seat not in ring:
        return {}
    btn_idx = ring.index(button_seat)
    rotated = ring[btn_idx+1:] + ring[:btn_idx+1]  # SB first, BTN last
    return {seat: order[i % 6] for i, seat in enumerate(rotated)}

def norm_amt_to_bb(s: str|float, bb: float) -> float:
    if isinstance(s, (int,float)): return round(float(s)/bb, 4) if bb>0 else 0.0
    return round(float(s.replace("$",""))/bb, 4) if bb>0 else 0.0

def extract_actions(lines: list[str], bb: float) -> list[dict]:
    street = "preflop"
    events: list[dict] = []
    for line in lines:
        line = line.strip()

        # street switches
        if line.startswith("*** FLOP ***"):  street = "flop";  continue
        if line.startswith("*** TURN ***"):  street = "turn";  continue
        if line.startswith("*** RIVER ***"): street = "river"; continue
        if line.startswith("*** SHOW DOWN ***") or line.startswith("*** SUMMARY ***"):
            break

        # normalize actions
        m = ACTION_RAISE.match(line)
        if m:
            p, by_amt, to_amt = m.group(1).strip(), float(m.group(2)), float(m.group(3))
            events.append({
                "street": street, "actor": p, "act": "raise",
                "amount_bb": norm_amt_to_bb(to_amt, bb)  # <— use the “to” size
            })
            continue

        m = ACTION_BET.match(line)
        if m:
            p, amt = m.group(1).strip(), float(m.group(2))
            events.append({
                "street": street,
                "actor": p,
                "act": "bet",
                "amount_bb": norm_amt_to_bb(amt, bb),
            })
            continue

        m = ACTION_CALL.match(line)
        if m:
            p, amt = m.group(1).strip(), float(m.group(2))
            events.append({
                "street": street,
                "actor": p,
                "act": "call",
                "amount_bb": norm_amt_to_bb(amt, bb),
            })
            continue

        m = ACTION_CHECK.match(line)
        if m:
            p = m.group(1).strip()
            events.append({
                "street": street,
                "actor": p,
                "act": "check",
            })
            continue

        m = ACTION_FOLD.match(line)
        if m:
            p = m.group(1).strip()
            events.append({
                "street": street,
                "actor": p,
                "act": "fold",
            })
            continue

    return events

# ── HELPERS ────────────────────────────────────────────────────────────────────
def determine_street(board_cards: list[str]) -> str:
    if len(board_cards) == 5:
        return "river"
    if len(board_cards) == 4:
        return "turn"
    if len(board_cards) == 3:
        return "flop"
    return "preflop"

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

def parse_hand_block(text: str, stake_label: str) -> dict:
    lines = [ln.strip("\ufeff") for ln in text.strip().splitlines()]  # strip BOMs
    hand_raw = {
        "hand_id": "",
        "table_name": None,
        "stake_tag": stake_label,
        "stakes": stake_label,
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
            sb, bb = parse_stakes(line)
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
    pos_by_seat = seat_to_position_6max(hand_raw["button_seat"], seat_nums)
    # map seat -> player_id
    seat_to_player = {s["seat_number"]: s["player_id"] for s in hand_raw["seats"]}
    hand_raw["position_by_player"] = {seat_to_player[s]: pos for s, pos in pos_by_seat.items() if s in seat_to_player}

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
    hand_raw["actions"] = acts
    hand_raw["players"] = sorted({ev["actor"] for ev in acts})

    # 7) Pot / rake / results (from summary lines)
    pot = 0.0; rake = 0.0
    for line in lines:
        if line.startswith("*** SUMMARY ***"): break
        # keep simple; optional: add specific regex for "Total pot $X | Rake $Y"
    for line in lines:
        if "Total pot" in line and "| Rake" in line:
            # Total pot $1.12 | Rake $0.06
            try:
                pot_s = line.split("Total pot")[1].split("|")[0].strip().replace("$","")
                rake_s = line.split("| Rake")[1].strip().replace("$","")
                pot = float(pot_s); rake = float(rake_s)
            except Exception:
                pass
    hand_raw["pot_bb"] = norm_amt_to_bb(pot, bb)
    hand_raw["rake_bb"] = norm_amt_to_bb(rake, bb)

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

# ── MAIN ───────────────────────────────────────────────────────────────────────
def parse_all_hands(stake: int):
    stake_label = f"NL{stake}"
    raw_dir = Path(f"data/raw/{stake_label}")
    tmp_jsonl = Path(f"hands_{stake_label}.jsonl")
    tmp_gz = Path(f"{tmp_jsonl}.gz")
    s3_key = f"parsed/{tmp_gz.name}"

    total = 0
    with tmp_jsonl.open("w", encoding="utf-8") as out_f:
        for txt in raw_dir.rglob("*.txt"):
            content = txt.read_text(errors="ignore")
            for block in HAND_SPLIT_RE.split(content):
                if not block.strip():
                    continue
                parsed = parse_hand_block(block, stake_label)
                json.dump(parsed, out_f, separators=(",", ":"))
                out_f.write("\n")
                total += 1

    # Compress the full JSONL file
    with tmp_jsonl.open("rb") as f_in, gzip.open(tmp_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    # Upload to S3
    s3.upload_file(tmp_gz, s3_key)

    # Cleanup
    tmp_jsonl.unlink(missing_ok=True)
    tmp_gz.unlink(missing_ok=True)

    print(f"✅ Parsed & uploaded {total} hands to s3://{s3.bucket}/{s3_key}")



# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stake", type=int, required=True, help="Stake in big blind cents, e.g. 10 for NL10")
    args = parser.parse_args()
    parse_all_hands(args.stake)