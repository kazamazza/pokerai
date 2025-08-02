import random
import json
from pathlib import Path
import eval7
from tqdm import tqdm

from features.types import STACK_BUCKETS, POSITIONS, ACTION_CONTEXTS
from ml.schema.equity_net_schema import EquityNetFeatures, EquityNetLabel

OUTPUT_FILE = Path("data/equitynet/simulations.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 5_000_000       # You can increase this to 1M+ for full training
EVAL_TRIALS = 5_000        # Monte Carlo samples per hand
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

        # Villain hands as lists of eval7.Card
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

def generate_simulation():
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
        num_players=num_players,
        include_antes=False
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
        normalized_equity=equity  # or apply a scaling function here if needed
    )

    return features, label


def compute_pot_size(
    action_context: str,
    num_players: int,
    include_antes: bool = False
) -> int:
    """
    Estimate preflop pot size based on action context and player count.

    Args:
        action_context (str): One of OPEN, VS_LIMP, VS_OPEN, VS_ISO, VS_3BET, VS_4BET
        num_players (int): Total players at the table (usually 2 to 6)
        include_antes (bool): Include antes in pot size calculation

    Returns:
        int: Estimated pot size in big blinds
    """
    sb = 0.5
    bb = 1.0
    antes = num_players * 0.1 if include_antes else 0.0

    if action_context == "OPEN":
        pot = sb + bb + 2.5  # Simple open raise
    elif action_context == "VS_LIMP":
        pot = sb + bb + 1.0 + 3.0  # Limp + iso
    elif action_context == "VS_OPEN":
        pot = sb + bb + 2.5 + 2.5  # Open + flat
    elif action_context == "VS_ISO":
        pot = sb + bb + 1.0 + 3.0 + 3.0  # Limp + iso + call
    elif action_context == "VS_3BET":
        pot = sb + bb + 2.5 + 8.0 + 8.0  # Open + 3bet + call
    elif action_context == "VS_4BET":
        pot = sb + bb + 2.5 + 8.0 + 20.0 + 20.0  # Open + 3bet + 4bet + call
    else:
        pot = sb + bb + 2.5  # Default to simple open raise

    total = pot + antes
    return int(round(total))

if __name__ == "__main__":
    print(f"🚀 Generating {NUM_SAMPLES} equity simulations...")
    with open(OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(NUM_SAMPLES)):
            features, label = generate_simulation()
            f.write(json.dumps({
                "features": features.model_dump(),
                "label": label.model_dump()
            }) + "\n")
    print(f"✅ Done! Saved to {OUTPUT_FILE}")