# ml/range/solvers/command_text.py

from typing import Dict, List, Literal, Optional

Street = Literal["flop","turn","river"]
Role   = Literal["ip","oop"]

def build_command_text(
    *,
    pot_bb: float,
    effective_stack_bb: float,
    board: str,                 # e.g. "QsJh2h" → must emit "Qs,Jh,2h"
    range_ip: str,
    range_oop: str,
    bet_sizes: Optional[Dict[Literal["flop","turn","river"],
                             Dict[Literal["ip","oop"],
                                  Dict[str, List[int] | bool]]]] = None,
    allin_threshold: float = 0.67,
    thread_num: int = 1,
    accuracy: float = 0.5,
    max_iteration: int = 200,
    print_interval: int = 10,
    use_isomorphism: int = 1,
    dump_path: str = "output_result.json",
) -> str:
    lines: List[str] = []

    # Required state
    lines.append(f"set_pot {int(pot_bb)}")
    lines.append(f"set_effective_stack {int(effective_stack_bb)}")

    # BOARD: comma-separated tokens (e.g. "Qs,Jh,2h")
    board_csv = ",".join([board[i:i+2] for i in range(0, len(board), 2)])
    lines.append(f"set_board {board_csv}")

    # RANGES: pass-through
    lines.append(f"set_range_ip {range_ip}")
    lines.append(f"set_range_oop {range_oop}")

    # Bet menus (comma-separated fields: role,street,kind[,sizes...])
    # If multiple sizes exist, emit them on a single line: ... raise,60,100
    if bet_sizes:
        for street, per_role in bet_sizes.items():
            for role, kinds in per_role.items():
                # donk/bet/raise with optional size lists
                for kind in ("donk", "bet", "raise"):
                    sizes = kinds.get(kind) or []
                    if isinstance(sizes, list) and len(sizes) > 0:
                        size_csv = ",".join(str(int(s)) for s in sizes)
                        lines.append(f"set_bet_sizes {role},{street},{kind},{size_csv}")
                # all-in toggle (no size argument)
                if kinds.get("allin"):
                    lines.append(f"set_bet_sizes {role},{street},allin")

    # Follow the example’s ordering
    lines.append(f"set_allin_threshold {allin_threshold}")
    lines.append("build_tree")
    lines.append(f"set_thread_num {int(thread_num)}")
    lines.append(f"set_accuracy {accuracy}")
    lines.append(f"set_max_iteration {int(max_iteration)}")
    lines.append(f"set_print_interval {int(print_interval)}")
    lines.append(f"set_use_isomorphism {int(use_isomorphism)}")
    lines.append("start_solve")
    lines.append("set_dump_rounds 2")
    lines.append(f"dump_result {dump_path}")
    # optional but harmless
    # lines.append("quit")

    return "\n".join(lines) + "\n"