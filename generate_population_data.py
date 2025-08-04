import gzip
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from features.action_mapping import normalize_action
from infra.storage.s3_uploader import S3Uploader
from ml.schema.population_net_schema import PopulationNetFeatures, PopulationNetLabel

s3 = S3Uploader()

# --- Utility: Position mapping ---
POSITION_ORDER = ["BTN", "SB", "BB", "UTG", "MP", "CO"]

def determine_action_context(actions: list[str]) -> str:
    text = " ".join(actions).lower()
    if "limp" in text:
        return "VS_LIMP"
    if "iso" in text:
        return "VS_ISO"
    if "4bet" in text:
        return "VS_4BET"
    if "3bet" in text or (text.count("raises") >= 2):
        return "VS_3BET"
    if "raises" in text:
        return "VS_OPEN"
    return "OPEN"

def get_position(seat_num: int, button_seat: int, seats: list[dict]) -> str:
    """Return position relative to button seat, supports 2-6 handed."""
    pos_order = {
        2: ["SB", "BB"],
        3: ["BTN", "SB", "BB"],
        4: ["BTN", "SB", "BB", "UTG"],
        5: ["BTN", "SB", "BB", "UTG", "CO"],
        6: ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    }
    total_players = len(seats)
    positions = pos_order.get(total_players, ["BTN"] * total_players)
    ordered_seats = sorted(seat["seat_number"] for seat in seats)
    btn_idx = ordered_seats.index(button_seat)
    clockwise = ordered_seats[btn_idx:] + ordered_seats[:btn_idx]
    if seat_num not in clockwise:
        return "BTN"
    return positions[clockwise.index(seat_num)]

def generate_population_data(stake: str):
    stake_label = f"NL{stake}" if not stake.startswith("NL") else stake
    gz_name = f"hands_{stake_label}.jsonl.gz"
    tmp_input_gz = Path(gz_name)
    tmp_input_jsonl = Path(gz_name.replace(".gz", ""))
    output_jsonl = Path(f"population_{stake_label}.jsonl")
    output_gz = output_jsonl.with_suffix(".jsonl.gz")
    s3_key_input = f"parsed/{gz_name}"
    s3_key_output = f"processed/{output_gz.name}"

    s3.download_file(s3_key_input, tmp_input_gz)
    with gzip.open(tmp_input_gz, "rb") as f_in, tmp_input_jsonl.open("wb") as f_out:
        f_out.write(f_in.read())

    grouped_data = defaultdict(lambda: defaultdict(lambda: {
        "fold": 0, "call": 0, "raise": 0, "bet_sizes": [], "count": 0
    }))

    with tmp_input_jsonl.open() as f:
        for line in tqdm(f, desc=f"📊 Parsing {stake_label} hands"):
            try:
                hand = json.loads(line)
                seats = hand.get("seats", [])
                if not seats or len(seats) < 2:
                    continue
                button = hand.get("button_seat", 1)
                player_count = len(seats)
                for seat in seats:
                    seat_num = seat["seat_number"]
                    pos = get_position(seat_num, button, seats)
                    key = (stake_label, determine_action_context(hand.get("actions", [])), pos, player_count)
                    bucket = grouped_data[hand.get("street", "preflop")][key]
                    for act in hand.get("actions", []):
                        action, amount = normalize_action(act)
                        if action not in {"fold", "call", "raise"}:
                            continue
                        bucket[action] += 1
                        if action == "raise" and amount:
                            bucket["bet_sizes"].append(amount / hand.get("min_bet", 0.1))
                        bucket["count"] += 1
            except Exception:
                continue

    with output_jsonl.open("w") as out_f:
        for street, stats_dict in grouped_data.items():
            for (stake, context, position, pc), stats in stats_dict.items():
                total = stats["count"]
                if total == 0:
                    continue
                features = PopulationNetFeatures(
                    stake_level=stake,
                    action_context=context,
                    position=position,
                    player_count=pc
                )
                label = PopulationNetLabel(
                    fold_pct=round(stats["fold"] / total, 4),
                    call_pct=round(stats["call"] / total, 4),
                    raise_pct=round(stats["raise"] / total, 4),
                    avg_bet_sizing=round(sum(stats["bet_sizes"]) / len(stats["bet_sizes"]), 4) if stats["bet_sizes"] else 0.0
                )
                json.dump({"features": features.model_dump(), "label": label.model_dump(), "street": street}, out_f)
                out_f.write("\n")

    with gzip.open(output_gz, "wb") as f_out, output_jsonl.open("rb") as f_in:
        f_out.write(f_in.read())

    s3.upload_file(output_gz, s3_key_output)
    tmp_input_gz.unlink(missing_ok=True)
    tmp_input_jsonl.unlink(missing_ok=True)
    output_jsonl.unlink(missing_ok=True)
    output_gz.unlink(missing_ok=True)
    print(f"✅ Uploaded population stats to s3://{s3.bucket}/{s3_key_output}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stake", required=True, help="e.g. NL10 or just 10")
    args = parser.parse_args()

    generate_population_data(args.stake)