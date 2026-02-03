from typing import Dict, List, Literal, Optional, Union

Street = Literal["flop","turn","river"]
Role   = Literal["ip","oop"]

def _fmt_num(x: float) -> str:
    s = f"{float(x):.6f}"
    return s.rstrip("0").rstrip(".") if "." in s else s

def build_command_text(
    *,
    pot_bb: float,
    effective_stack_bb: float,
    board: str,
    range_ip: str,
    range_oop: str,
    bet_sizes: Optional[
        Dict[
            Literal["flop", "turn", "river"],
            Dict[
                Literal["ip", "oop"],
                Dict[str, Union[List[float], List[int], bool]]
            ]
        ]
    ] = None,
    allin_threshold: float = 0.67,
    thread_num: int = 1,
    accuracy: float = 0.5,
    max_iteration: int = 200,
    print_interval: int = 10,
    use_isomorphism: int = 1,
    dump_path: str = "output_result.json",
) -> str:
    lines: List[str] = []
    lines.append(f"set_pot {_fmt_num(pot_bb)}")
    lines.append(f"set_effective_stack {_fmt_num(effective_stack_bb)}")

    board_csv = ",".join([board[i:i+2] for i in range(0, len(board), 2)])
    lines.append(f"set_board {board_csv}")
    lines.append(f"set_range_ip {range_ip}")
    lines.append(f"set_range_oop {range_oop}")

    if bet_sizes:
        for street, per_role in bet_sizes.items():
            for role, kinds in per_role.items():
                for kind in ("donk", "bet", "raise"):
                    sizes = kinds.get(kind) or []
                    if isinstance(sizes, list) and len(sizes) > 0:
                        if kind in ("bet", "donk"):
                            size_csv = ",".join(f"{s * 100:.0f}" for s in sizes)  # fractions -> %
                        else:
                            size_csv = ",".join(_fmt_num(float(s)) for s in sizes)  # multipliers stay multipliers
                        lines.append(f"set_bet_sizes {role},{street},{kind},{size_csv}")
                # all-in toggle
                if kinds.get("allin"):
                    lines.append(f"set_bet_sizes {role},{street},allin")

    lines.append(f"set_allin_threshold {allin_threshold}")
    lines.append("build_tree")
    lines.append(f"set_thread_num {int(thread_num)}")
    lines.append(f"set_accuracy {accuracy}")
    lines.append(f"set_max_iteration {int(max_iteration)}")
    lines.append(f"set_print_interval {int(print_interval)}")
    lines.append(f"set_use_isomorphism {int(use_isomorphism)}")
    lines.append("start_solve")
    lines.append("set_dump_rounds 3")
    lines.append(f"dump_result {dump_path}")

    return "\n".join(lines) + "\n"