# ml/range/solvers/command_text.py

from typing import Dict, List, Literal, Optional

Street = Literal["flop","turn","river"]
Role   = Literal["ip","oop"]

def build_command_text(
    *,
    pot_bb: float,
    effective_stack_bb: float,
    board: str,                 # "QsJh2h"
    range_ip: str,              # "AA,KK,QQ,..." (Monker-like)
    range_oop: str,
    # action menus per street (omit a street to skip it entirely)
    bet_sizes: Optional[Dict[Street, Dict[Role, Dict[str, List[int]]]]] = None,
    # solver knobs
    allin_threshold: float = 0.67,
    thread_num: int = 1,
    accuracy: float = 0.5,
    max_iteration: int = 200,
    print_interval: int = 10,
    use_isomorphism: int = 1,
    dump_path: str = "output_result.json",   # absolute path recommended
) -> str:
    """
    bet_sizes: {
      "flop": {
        "oop": {"bet": [50], "raise": [60], "donk": [], "allin": True},
        "ip":  {"bet": [50], "raise": [60], "allin": True}
      },
      # "turn": {...},   # omit to skip that street completely
      # "river": {...},
    }
    Percentages are integer pot % (e.g., 50 means 50% pot).
    Presence of a key enables that action; empty list means no sizes for that action.
    """
    lines: List[str] = []
    # Required state
    lines.append(f"set_pot {int(pot_bb)}")
    lines.append(f"set_effective_stack {int(effective_stack_bb)}")
    # Ensure commas between cards for this solver
    board_commas = ",".join([board[i:i+2] for i in range(0, len(board), 2)])
    lines.append(f"set_board {board_commas}")
    lines.append(f"set_range_ip {range_ip}")
    lines.append(f"set_range_oop {range_oop}")

    def emit_sizes(street: Street, role: Role, kind: str, sizes: List[int]):
        # kind in {"bet","raise","donk"} – solver expects this exact token set
        for pct in sizes:
            lines.append(f"set_bet_sizes {role},{street},{kind},{pct}")

    if bet_sizes:
        for street, per_role in bet_sizes.items():
            for role, kinds in per_role.items():
                # Optional kinds
                for kind in ("donk","bet","raise"):
                    if kind in kinds and kinds[kind]:
                        emit_sizes(street, role, kind, kinds[kind])
                # All-in switches
                if kinds.get("allin"):
                    lines.append(f"set_bet_sizes {role},{street},allin")

    # Tree + solver controls
    lines.append("build_tree")
    lines.append(f"set_thread_num {int(thread_num)}")
    lines.append(f"set_accuracy {accuracy}")
    lines.append(f"set_max_iteration {int(max_iteration)}")
    lines.append(f"set_print_interval {int(print_interval)}")
    lines.append(f"set_use_isomorphism {int(use_isomorphism)}")
    lines.append(f"set_allin_threshold {allin_threshold}")
    lines.append("start_solve")
    # Dump to an ABSOLUTE path you pass from the adapter
    lines.append("set_dump_rounds 2")
    lines.append(f"dump_result {dump_path}")
    return "\n".join(lines) + "\n"