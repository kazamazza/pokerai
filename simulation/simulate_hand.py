import eval7
import uuid
from typing import Dict, Any, List

from features.board_analyzer import BoardAnalyzer
from features.board_clusterer import BoardClusterer
from features.range_extractor import RangeFeatureExtractor
from features.types import (
    Action,
    ActionType,
    Hand,
    Seat,
    Status,
    RangeFeatureRequest
)
from simulation.solver_interface import run_solver  # placeholder, to be implemented

def generate_training_sample(player_type="reg") -> Dict[str, Any]:
    """
    Generate a synthetic training sample with full input-output for the model.
    """

    # 1. Deal hole cards and board
    deck = eval7.Deck()
    deck.shuffle()

    hero_hand = [deck.deal(1)[0], deck.deal(1)[0]]
    board = [deck.deal(1)[0] for _ in range(3)]  # Flop only

    # 2. Create seats (2-player HU)
    seats = [
        Seat(seat_number=1, player_id="hero", stack_size=100.0, status=Status.active),
        Seat(seat_number=2, player_id="villain", stack_size=100.0, status=Status.active),
    ]

    # 3. Create action history (preflop only for now)
    actions = [
        Action(player_id="hero", type=ActionType.POST_SB, amount=0.5),
        Action(player_id="villain", type=ActionType.POST_BB, amount=1.0),
        Action(player_id="hero", type=ActionType.RAISE, amount=2.5),
        Action(player_id="villain", type=ActionType.CALL, amount=2.5),
    ]

    # 4. Construct Hand object
    current_hand = Hand(
        hand_id=str(uuid.uuid4()),
        seats=seats,
        button_seat=1,
        actions=actions,
        board=[str(c) for c in board],
        street="flop",
    )

    # 5. Generate session history (placeholder empty or reuse current hand)
    history = [current_hand]  # simple use for now

    # 6. Build RangeFeatureRequest
    board_clusterer = BoardClusterer()
    board_analyzer = BoardAnalyzer(board_clusterer)
    extractor = RangeFeatureExtractor(history, board_analyzer)

    req = RangeFeatureRequest(
        history=history,
        current_hand=current_hand,
        player_id="villain"
    )

    features = extractor.extract(req)
    flat_features = flatten_feature_dict(features)

    # 7. Generate opponent range (mocked for now)
    opponent_range = generate_opponent_range(player_type)

    # 8. Compute equity
    equity = compute_equity(hero_hand, board, opponent_range)

    # 9. Get solver labels
    range_label, action_probs = run_solver(hero_hand, board, "BB", 100)

    return {
        "features": flat_features,
        "equity": equity,
        "range_label": range_label,
        "action_probs": action_probs,
    }


def compute_equity(hero_cards, board, opponent_range) -> float:
    """
    Use eval7 to compute equity of hero's hand vs opponent's range.
    """
    wins = 0
    ties = 0
    total = 0

    board_cards = [c for c in board]
    for opp_hand in opponent_range:  # list of tuples of two eval7.Card
        full_board = [eval7.Card(str(c)) for c in board_cards]
        hero = hero_cards
        opp = opp_hand

        all_cards = hero + opp + full_board
        if len(set(all_cards)) < 7:
            continue  # skip overlapping cards

        hero_score = eval7.evaluate(hero + full_board)
        opp_score = eval7.evaluate(opp + full_board)

        if hero_score > opp_score:
            wins += 1
        elif hero_score == opp_score:
            ties += 1
        total += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


def flatten_feature_dict(features: Dict[str, Any]) -> List[float]:
    """
    Flatten a feature dict into a 1D list of floats (for training input).
    """
    vec = []
    for key in sorted(features.keys()):
        val = features[key]
        if isinstance(val, list):
            vec.extend(val)
        elif isinstance(val, (float, int)):
            vec.append(float(val))
        elif isinstance(val, bool):
            vec.append(1.0 if val else 0.0)
        else:
            raise TypeError(f"Unsupported feature type: {key} = {val}")
    return vec