# simulation/solver_interface.py

import os
import subprocess
import tempfile
import json
from typing import List, Tuple
from features.types import SolverRequest
from simulation.combo_utils import get_hero_combo_string, get_169_combo_list

# Adjust if your solver binary is somewhere else
SOLVER_BINARY = "./external/solver/console_solver"
OUTPUT_PATH = "output_result.json"

def run_solver(req: SolverRequest) -> Tuple[List[float], List[float]]:
    """
    Run TexasSolver via CLI using a complete SolverRequest object.
    Returns:
      - 169-dim opponent range vector
      - action probabilities for hero's combo
    """
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".txt", delete=False) as f:
        input_file = f.name
        _write_solver_input(f, req)
        f.flush()

    try:
        print(f"\n🚀 Launching solver: {SOLVER_BINARY} -i {input_file}")
        with open(input_file, 'r') as debug_f:
            print("\n📝 Input file contents:")
            print(debug_f.read())

        subprocess.run([SOLVER_BINARY, "-i", input_file], check=True)

        with open(OUTPUT_PATH, "r") as f:
            data = json.load(f)

        opponent_range = _parse_opponent_range(data)
        action_probs = _parse_action_probs(data, req.hero_cards)

        return opponent_range, action_probs

    finally:
        os.remove(input_file)


def _write_solver_input(f, req: SolverRequest):
    """Write full command input file using a SolverRequest object."""

    board_str = ",".join(req.board)

    f.write(f"set_pot {req.pot_size}\n")
    f.write(f"set_effective_stack {req.stack_depth}\n")
    f.write(f"set_board {board_str}\n")
    f.write(f"set_range_ip {req.ip_range}\n")
    f.write(f"set_range_oop {req.oop_range}\n")

    # Write all bet sizes dynamically
    for (street, role, act), size in req.bet_sizes.items():
        f.write(f"set_bet_sizes {role},{street},{act},{size}\n")

    f.write("set_allin_threshold 1.0\n")
    f.write("build_tree\n")
    f.write("set_thread_num 4\n")
    f.write("set_accuracy 0.5\n")
    f.write("set_max_iteration 200\n")
    f.write("set_print_interval 10\n")
    f.write("set_use_isomorphism 1\n")
    f.write("start_solve\n")
    f.write("set_dump_rounds 2\n")
    f.write(f"dump_result {OUTPUT_PATH}\n")


def _parse_opponent_range(data: dict) -> List[float]:
    """
    Build a 169-combo opponent range vector from strategy node.
    Averages probabilities across all actions for each combo.
    """
    strategy = _find_first_strategy_node(data)
    combo_action_map = strategy.get("strategy", {})

    combo_probs = {
        combo: sum(probs) / len(probs)
        for combo, probs in combo_action_map.items()
    }

    ordered_combos = get_169_combo_list()
    return [combo_probs.get(c, 0.0) for c in ordered_combos]


def _parse_action_probs(data: dict, hero_cards: List[str]) -> List[float]:
    """
    Extract action probabilities for the hero’s combo (e.g., 'Td9d').
    Returns [fold_prob, call_prob, raise_prob, ...]
    """
    strategy = _find_first_strategy_node(data)
    combo_map = strategy.get("strategy", {})

    hero_combo = get_hero_combo_string(hero_cards)
    return combo_map.get(hero_combo, [0.0] * len(strategy.get("actions", [])))

def _find_first_strategy_node(data: dict) -> dict:
    """
    Recursively find and return the first strategy node in the solver output.
    This is usually the first action node where strategies are defined.
    """
    def recurse(node):
        if "strategy" in node:
            return node["strategy"]
        if "childrens" in node:
            for child in node["childrens"].values():
                result = recurse(child)
                if result:
                    return result
        return None

    strategy = recurse(data)
    if not strategy:
        raise ValueError("No strategy node found in solver output.")
    return strategy