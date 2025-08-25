import subprocess
from pathlib import Path

def write_test_input(path: Path):
    """Write a minimal command file for solver smoke test."""
    txt = """\
set_pot 10
set_effective_stack 100
set_board Qs,Jh,2h
set_range_ip AA,KK,QQ
set_range_oop JJ,TT,99
set_bet_sizes oop,flop,bet,50
set_bet_sizes ip,flop,bet,50
set_allin_threshold 0.67
build_tree
set_thread_num 1
set_accuracy 1.0
set_max_iteration 5
set_print_interval 1
start_solve
dump_result output_test.json
"""
    path.write_text(txt)

def run_solver():
    solver_bin = Path("external/solver/console_solver")  # adjust if different
    input_file = Path("test_input.txt")
    write_test_input(input_file)

    print(f"▶️ Running solver with {input_file}...")
    result = subprocess.run(
        [str(solver_bin), "-i", str(input_file)],
        capture_output=True,
        text=True,
    )

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    if Path("output_test.json").exists():
        print("✅ Solver produced output_test.json")
    else:
        print("❌ Solver did not produce output JSON")

if __name__ == "__main__":
    run_solver()