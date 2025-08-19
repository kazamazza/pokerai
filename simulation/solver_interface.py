# simulation/solver_interface.py
from __future__ import annotations
import os
import subprocess
import tempfile
import json
from typing import List, Tuple
from ml.types import SolverRequest
from simulation.combo_utils import get_hero_combo_string
from utils.combos import get_169_combo_list
from utils.solver_input_sanitize import assert_console_range_ok, console_safe_range

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


def _write_solver_input(f, req: SolverRequest) -> None:
    f.write(f"set_pot {int(req.pot_size)}\n")
    f.write(f"set_effective_stack {int(req.stack_depth)}\n")

    board = req.board if isinstance(req.board, list) else [req.board[0:2], req.board[2:4], req.board[4:6]]
    f.write(f"set_board {board[0]},{board[1]},{board[2]}\n")

    def _as_str(r):
        if isinstance(r, str):   return r.strip()
        if isinstance(r, list):  return ",".join(s for s in (x.strip() for x in r) if s)
        return str(r).strip()

    ip_raw = _as_str(req.ip_range)
    oop_raw = _as_str(req.oop_range)

    # Expand ALL '+' shorthand into explicit tokens
    ip_safe = console_safe_range(ip_raw, expand_all_plus=True)
    oop_safe = console_safe_range(oop_raw, expand_all_plus=True)

    # Verify solver-safe
    assert_console_range_ok(ip_safe)
    assert_console_range_ok(oop_safe)

    f.write("set_range_ip " + ip_safe + "\n")
    f.write("set_range_oop " + oop_safe + "\n")

    # bet sizes (same as you have; keep ints)
    def emit(role, street, act, sizes):
        if act == "allin":
            f.write(f"set_bet_sizes {role},{street},allin\n")
            return
        vals = sizes if isinstance(sizes, list) else [sizes]
        ints = []
        for v in vals:
            iv = int(round(float(v)))
            if iv < 1: iv = 1
            ints.append(str(iv))
        f.write(f"set_bet_sizes {role},{street},{act}," + ",".join(ints) + "\n")

    order = [("flop","oop"),("flop","ip"),("turn","oop"),("turn","ip"),("river","oop"),("river","ip")]
    for street, role in order:
        for act in ("bet","raise"):
            k = (street, role, act)
            if k in req.bet_sizes: emit(role, street, act, req.bet_sizes[k])
        k = (street, role, "allin")
        if k in req.bet_sizes: emit(role, street, "allin", None)

    f.write(f"set_allin_threshold {req.allin_threshold}\n")
    f.write("build_tree\n")
    f.write(f"set_thread_num {int(req.threads)}\n")
    f.write(f"set_accuracy {float(req.accuracy)}\n")
    f.write(f"set_max_iteration {int(req.max_iterations)}\n")
    f.write(f"set_print_interval {int(req.print_interval)}\n")
    f.write(f"set_use_isomorphism {1 if req.use_isomorphism else 0}\n")
    f.write("start_solve\n")
    f.write(f"set_dump_rounds {int(req.dump_rounds)}\n")
    f.write(f"dump_result {req.output_path}\n")


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