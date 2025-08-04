import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from features.action_mapping import normalize_action
from features.types import ACTION_CONTEXTS
from ml.schema.population_net_schema import PopulationNetFeatures, PopulationNetLabel

# --- HELPERS ---
def extract_action_info(action_line: str):
    """
    Extracts normalized action type and amount from raw text.
    """
    action_line = action_line.lower()
    action = normalize_action(action_line)
    amount = None

    if "$" in action_line:
        try:
            amount = float(action_line.split("$")[-1])
        except Exception:
            pass

    return action, amount

def determine_action_context(actions: list[str]) -> str:
    for context in ACTION_CONTEXTS:
        if context.lower() in " ".join(actions).lower():
            return context
    return "OPEN"

# --- MAIN SCRIPT ---
def generate_population_data(stake: str):
    input_path = Path(f"data/parsed/hands_{stake}.jsonl")
    output_path = Path(f"data/processed/population_{stake}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped_data = defaultdict(lambda: {
        "fold": 0, "call": 0, "raise": 0, "bet_sizes": [], "count": 0
    })

    with input_path.open() as f:
        for line in tqdm(f, desc=f"📊 Parsing {stake} hands"):
            try:
                hand = json.loads(line)
                position = hand.get("hero_position") or "BTN"
                player_count = hand.get("player_count") or 6
                action_context = determine_action_context(hand.get("actions", []))
                min_bet = hand.get("min_bet") or 0.1

                key = (stake, action_context, position, player_count)

                for act in hand.get("actions", []):
                    action, amount = extract_action_info(act)
                    if action == "fold":
                        grouped_data[key]["fold"] += 1
                    elif action == "call":
                        grouped_data[key]["call"] += 1
                    elif action == "raise":
                        grouped_data[key]["raise"] += 1
                        if amount:
                            grouped_data[key]["bet_sizes"].append(amount / min_bet)

                grouped_data[key]["count"] += 1
            except Exception:
                continue

    with output_path.open("w") as out_f:
        for (stake, context, position, pc), stats in grouped_data.items():
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
            json.dump({"features": features.model_dump(), "label": label.model_dump()}, out_f)
            out_f.write("\n")

    print(f"✅ Done! Saved: {output_path}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stake", required=True, help="e.g. NL10, NL25")
    args = parser.parse_args()

    generate_population_data(args.stake)