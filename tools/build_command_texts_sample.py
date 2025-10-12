import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.config.bet_menus import build_contextual_bet_sizes
from ml.etl.rangenet.postflop.helpers_topology import _menu_for
from ml.range.solvers.command_text import build_command_text


SCENARIOS = [
    ("VS_OPEN", "BTN", "BB", "BTN", None, "srp_ip"),
    ("VS_OPEN", "BB", "BTN", "BTN", None, "srp_oop"),
    ("BLIND_VS_STEAL", "BTN", "BB", "BTN", None, "bvs"),
    ("VS_3BET", "BTN", "BB", "BTN", "BB", "3bet_ip"),
    ("VS_3BET", "BB", "BTN", "BTN", "BB", "3bet_oop"),
    ("VS_4BET", "BTN", "BB", "BTN", "BB", "4bet"),
    ("LIMPED_SINGLE", "SB", "BB", None, None, "limp"),
    ("LIMPED_MULTI", "BTN", "BB", None, None, "limp"),
]

def main():
    for ctx, ip, oop, opener, three_bettor, tag in SCENARIOS:
        print(f"\n=== {ctx} ({ip} vs {oop}) ===")
        menu_id, sizes = _menu_for(ctx, ip, oop, opener, three_bettor, menu_tag=tag, stake="NL10")
        print(f"menu_id={menu_id} sizes={sizes}")

        bet_sizes = build_contextual_bet_sizes(menu_id, stakes="NL10")
        cmd = build_command_text(
            pot_bb=7.5,
            effective_stack_bb=100,
            board="QsJh2h",
            range_ip="AA:1.0,KK:1.0",
            range_oop="AA:1.0,KK:1.0",
            bet_sizes=bet_sizes,
            dump_path=f"/tmp/{menu_id}.json",
        )
        print(cmd)

if __name__ == "__main__":
    main()