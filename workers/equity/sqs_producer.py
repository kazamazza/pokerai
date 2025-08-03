import os
import json
import sys
import random
from pathlib import Path
from tqdm import tqdm
import boto3
from dotenv import load_dotenv
import eval7

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.schema.equity_net_schema import EquityNetFeatures, EquityNetLabel
from features.types import STACK_BUCKETS, POSITIONS, ACTION_CONTEXTS

# Load AWS credentials from .env
load_dotenv()
sqs = boto3.client(
    "sqs",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

QUEUE_URL = os.getenv("EQUITY_QUEUE_URL")
EXPECTED_TOTALS_PATH = os.getenv("EXPECTED_TOTALS_PATH")  # e.g. "s3://pokeraistore/expected.json"

NUM_SAMPLES = 5_000_000
EVAL_TRIALS = 5_000
INITIATIVE_OPTIONS = [True, False]

def generate_random_hand(deck):
    return deck.deal(2)

def generate_random_board(deck: eval7.Deck, street="flop") -> list[eval7.Card]:
    board_size = {"flop": 3, "turn": 4, "river": 5}[street]
    return deck.deal(board_size)

def compute_equity(hero, board, num_players, trials=100):
    hero_score = 0
    total = 0

    hero_cards = [eval7.Card(c) for c in hero]
    board_cards = [eval7.Card(c) for c in board]

    for _ in range(trials):
        deck = eval7.Deck()
        used = set(hero_cards + board_cards)
        deck.cards = [c for c in deck.cards if c not in used]
        deck.shuffle()

        villains = [[deck.deal(1)[0], deck.deal(1)[0]] for _ in range(num_players - 1)]

        sim_board = board_cards.copy()
        while len(sim_board) < 5:
            sim_board.append(deck.deal(1)[0])

        hero_eval = eval7.evaluate(hero_cards + sim_board)
        villain_evals = [eval7.evaluate(v + sim_board) for v in villains]

        if hero_eval > max(villain_evals):
            hero_score += 1
        elif hero_eval == max(villain_evals):
            hero_score += 0.5
        total += 1

    return round(hero_score / total, 4)

def compute_pot_size(action_context: str, num_players: int, include_antes: bool = False) -> int:
    sb = 0.5
    bb = 1.0
    antes = num_players * 0.1 if include_antes else 0.0

    if action_context == "OPEN":
        pot = sb + bb + 2.5
    elif action_context == "VS_LIMP":
        pot = sb + bb + 1.0 + 3.0
    elif action_context == "VS_OPEN":
        pot = sb + bb + 2.5 + 2.5
    elif action_context == "VS_ISO":
        pot = sb + bb + 1.0 + 3.0 + 3.0
    elif action_context == "VS_3BET":
        pot = sb + bb + 2.5 + 8.0 + 8.0
    elif action_context == "VS_4BET":
        pot = sb + bb + 2.5 + 8.0 + 20.0 + 20.0
    else:
        pot = sb + bb + 2.5

    return int(round(pot + antes))

def enqueue_simulations():
    total = 0
    for _ in tqdm(range(NUM_SAMPLES)):
        deck = eval7.Deck()
        deck.shuffle()

        hero_hand = generate_random_hand(deck)
        board = generate_random_board(deck, street="flop")
        num_players = random.randint(2, 6)
        stack_bb = random.choice(STACK_BUCKETS)
        position = random.choice(POSITIONS)
        has_initiative = random.choice(INITIATIVE_OPTIONS)
        pot_size = compute_pot_size(
            action_context=random.choice(ACTION_CONTEXTS),
            num_players=num_players
        )

        equity = compute_equity(
            [str(c) for c in hero_hand],
            [str(c) for c in board],
            num_players,
            EVAL_TRIALS
        )

        features = EquityNetFeatures(
            hero_hand=[str(c) for c in hero_hand],
            board=[str(c) for c in board],
            num_players=num_players,
            stack_bb=stack_bb,
            position=position,
            has_initiative=has_initiative,
            pot_size=pot_size
        )
        label = EquityNetLabel(
            raw_equity=equity,
            normalized_equity=equity
        )

        msg = json.dumps({"features": features.model_dump(), "label": label.model_dump()})
        sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=msg)

        total += 1
        if total % 100 == 0:
            print(f"📨 Enqueued {total} tasks")

    print(f"✅ Enqueued all {total} equity simulations")
    from utils.expected_counts import update_expected_count
    update_expected_count("equity_simulations", total)

if __name__ == "__main__":
    enqueue_simulations()