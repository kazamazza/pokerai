import argparse
import json
import tempfile
from pathlib import Path

from infra.storage import s3_uploader
from parsers.parse_hand_history.regex_patterns import FLOP_RE, TURN_RE, RIVER_RE, TABLE_RE, SEATLINE_RE, HOLE_CARDS_RE, \
    SHOWDOWN_RE, ACTION_LINE_RE, FOLD_RE, SUMMARY_FOLD_RE, HAND_SPLIT_RE
from parsers.parse_hand_history.utils import RawSeat, HandSchema


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
    lines = text.strip().splitlines()
    hand_raw = {
        "hand_id": "",
        "table_name": None,
        "stake_tag": stake_label,
        "stakes": stake_label,
        "button_seat": None,
        "seats": [],
        "street": "preflop",
        "hero": None,
        "hole_cards_by_player": {},
        "board": [],
        "actions": [],
        "players": [],
        "winnings": [],
        "min_bet": 0.10,
    }

    # 1. Hand ID, table name, and button seat
    for line in lines:
        if line.startswith("PokerStars Hand #"):
            hand_raw["hand_id"] = line.split("#")[1].split(":")[0]
        if (m := TABLE_RE.search(line)):
            hand_raw["table_name"] = m.group(1)
            hand_raw["button_seat"] = int(m.group(2))

    # 2. Seats
    for line in lines:
        if (m := SEATLINE_RE.match(line)):
            num = int(m.group(1))
            player = m.group(2)
            stack = float(m.group(3)) if m.group(3) else 0.0
            if "sit out" in line.lower():
                status = "sitting out"
            else:
                status = "active"
            seat = RawSeat(
                seat_number=num,
                player_id=player,
                stack_size=stack,
                status=status,
            )
            hand_raw["seats"].append(seat.dict())

    # 3. Hole cards and hero
    for line in lines:
        if (m := HOLE_CARDS_RE.search(line)):
            player = m.group(1).strip()
            c1, c2 = m.group(2), m.group(3)
            hand_raw["hole_cards_by_player"][player] = [c1, c2]
            hand_raw["hero"] = player

    # 4. Showdown revelations
    for line in lines:
        if (m := SHOWDOWN_RE.match(line)):
            player = m.group(1).strip()
            c1, c2 = m.group(2), m.group(3)
            hand_raw["hole_cards_by_player"].setdefault(player, [c1, c2])

    # 5. Board and street
    hand_raw["board"] = extract_board(lines)
    hand_raw["street"] = determine_street(hand_raw["board"])

    # 6. Actions
    for line in lines:
        if (m := ACTION_LINE_RE.match(line)):
            player = m.group(1).strip()
            action_desc = m.group(2).strip()
            hand_raw["actions"].append(f"{player} {action_desc}")
            if player not in hand_raw["players"]:
                hand_raw["players"].append(player)
        if (m := FOLD_RE.match(line)):
            player = m.group(1).strip()
            entry = f"{player} folds"
            hand_raw["actions"].append(entry)
            if player not in hand_raw["players"]:
                hand_raw["players"].append(player)
        if (m := SUMMARY_FOLD_RE.match(line)):
            player, st = m.group(1).strip(), m.group(2).lower()
            entry = f"{player} folds before {st}"
            hand_raw["actions"].append(entry)
            if player not in hand_raw["players"]:
                hand_raw["players"].append(player)

    # 7. Normalize and validate
    return HandSchema(**hand_raw).dict()

# ── MAIN ───────────────────────────────────────────────────────────────────────
def parse_all_hands(stake: int):
    stake_label = f"NL{stake}"
    raw_dir = Path(f"data/raw/{stake_label}")
    s3_key = f"parsed/hands_{stake_label}.jsonl"

    total = 0
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp_f:
        tmp_path = tmp_f.name
        for txt in raw_dir.rglob("*.txt"):
            content = txt.read_text(errors="ignore")
            for block in HAND_SPLIT_RE.split(content):
                if not block.strip():
                    continue
                parsed = parse_hand_block(block, stake_label)
                json.dump(parsed, tmp_f)
                tmp_f.write("\n")
                total += 1

    s3_uploader.upload(tmp_path, s3_key)
    Path(tmp_path).unlink(missing_ok=True)
    print(f"✅ Parsed & uploaded {total} hands to s3://{s3_uploader.bucket}/{s3_key}")



# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stake", type=int, required=True, help="Stake in big blind cents, e.g. 10 for NL10")
    args = parser.parse_args()
    parse_all_hands(args.stake)