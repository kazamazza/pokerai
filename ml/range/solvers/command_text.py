from typing import Dict, List, Literal, Optional, Union
from ml.range.solvers.context_profiles import ContextProfile

Street = Literal["flop","turn","river"]
Role   = Literal["ip","oop"]

def _format_board_csv(board: str) -> str:
    b = (board or "").strip()
    if "," in b:
        parts = [t.strip() for t in b.split(",") if t.strip()]
    else:
        if len(b) % 2 != 0:
            raise ValueError(f"board string length must be even, got {b!r}")
        parts = [b[i:i+2] for i in range(0, len(b), 2)]
    if not all(len(t) == 2 for t in parts):
        raise ValueError(f"bad board tokens parsed from {board!r}: {parts}")
    return ",".join(parts)

def _strip_allins_in_menu(menu: dict, strip: bool) -> dict:
    if not strip:
        return menu
    out: dict = {}
    for street, per_role in menu.items():
        out[street] = {}
        for role, kinds in per_role.items():
            k2 = {k: v for k, v in kinds.items() if k != "allin"}
            out[street][role] = k2
    return out

def build_command_text(
    *,
    profile: ContextProfile,
    pot_bb: float,
    effective_stack_bb: float,
    board: str,
    range_ip: str,
    range_oop: str,
    scale_units: int = 100,
    print_interval: int = 20,
    thread_num: int = 1,
) -> str:
    spr = (float(effective_stack_bb) / float(pot_bb)) if float(pot_bb) > 0 else 0.0
    if spr < profile.min_spr:
        raise RuntimeError(f"SPR={spr:.2f} < min_spr={profile.min_spr:.2f} for {profile.name}; "
                           f"check pot/effective or choose a shallower profile")

    board_csv = _format_board_csv(board)

    # strip all-ins on flop/turn when SPR is healthy; keep river AI
    strip_ai = spr >= profile.strip_allins_when_spr_ge
    menu = _strip_allins_in_menu(profile.bet_menu, strip_ai)

    lines: List[str] = []
    lines.append(f"set_scale {int(scale_units)}")
    lines.append(f"set_pot {float(pot_bb)}")
    lines.append(f"set_effective_stack {float(effective_stack_bb)}")
    lines.append(f"set_board {board_csv}")
    lines.append(f"set_range_ip {range_ip}")
    lines.append(f"set_range_oop {range_oop}")

    # bet sizes
    for street in ("flop","turn","river"):
        per_role = menu.get(street, {})
        for role in ("oop","ip"):
            kinds = per_role.get(role, {})
            for kind in ("bet","raise"):
                sizes = kinds.get(kind) or []
                if sizes:
                    size_csv = ",".join(str(s) for s in sizes)
                    lines.append(f"set_bet_sizes {role},{street},{kind},{size_csv}")
            if kinds.get("allin"):
                lines.append(f"set_bet_sizes {role},{street},allin")

    lines.append(f"set_allin_threshold {profile.allin_threshold}")
    lines.append("build_tree")
    lines.append(f"set_thread_num {int(thread_num)}")
    lines.append(f"set_accuracy {profile.accuracy}")
    lines.append(f"set_max_iteration {int(profile.max_iter)}")
    lines.append(f"set_print_interval {int(print_interval)}")
    lines.append("set_use_isomorphism 1")
    lines.append("start_solve")
    lines.append("set_dump_rounds 2")
    return "\n".join(lines) + "\n"