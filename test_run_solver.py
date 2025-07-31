from simulation.combo_utils import get_hero_combo_string
from simulation.solver_interface import run_solver


def test_run_solver():
    hero_cards = ["Td", "9d"]
    board = ["Qs", "8d", "7h"]
    position = "BB"  # assume we're IP
    stack = 10.0

    print("Running solver...")
    try:
        range_label, action_probs = run_solver(hero_cards, board, position, stack)

        print("\n✅ Opponent Range Distribution (169 combos):")
        print(range_label[:10], "...")  # show first 10 values

        print("\n🎯 Action Probabilities for", get_hero_combo_string(hero_cards))
        print(action_probs)

    except Exception as e:
        print("❌ Solver test failed:", e)


if __name__ == "__main__":
    test_run_solver()