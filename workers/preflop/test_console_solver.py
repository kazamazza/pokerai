# test_console_solver.py
import os, json, tempfile, subprocess, sys
from pathlib import Path
from typing import Dict, Tuple, Literal, Union, List



ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from simulation.solver_interface import _write_solver_input
from ml.types import SolverRequest, SizeVal

# ✅ import YOUR definitions (adjust import paths to your repo)


SOLVER_BIN = "./external/solver/console_solver"  # adjust if needed


BetKey = Tuple[
    Literal["flop","turn","river"],
    Literal["ip","oop"],
    Literal["bet","raise","donk","allin"],
]
BetVal = Union[int, float, str, List[Union[int, float, str]], None]

def build_production_smoke_request() -> SolverRequest:
    """
    Benchmark smoke request using the *exact* config that build_solver_request_from_cluster()
    would use for a typical 100bb SRP cluster.
    """
    # Match your production defaults (no raises, 1 size/street, all-in at river)
    bet_sizes: Dict[BetKey, SizeVal] = {
        ("flop",  "oop", "bet"):   33,
        ("flop",  "ip",  "bet"):   33,
        ("turn",  "oop", "bet"):   66,
        ("turn",  "ip",  "bet"):   66,
        ("river", "oop", "bet"):   100,
        ("river", "ip",  "bet"):   100,
        ("river", "oop", "allin"): None,
        ("river", "ip",  "allin"): None,
    }

    return SolverRequest(
        pot_size=6.5,                  # same preflop open/call pot size
        stack_depth=100.0,             # standard 100bb cluster depth
        board=["2c", "4d", "Qh"],      # arbitrary flop; doesn't matter for speed test
        ip_range="AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,KQs",  # just placeholder combos
        oop_range="AA,KK,QQ,JJ,TT,99,88,77,66,55,AQs,AJs,KQs,KJs,QJs", # same here
        bet_sizes=bet_sizes,

        # Match cluster builder knobs exactly
        allin_threshold=0.67,
        threads=1,
        accuracy=1.0,
        max_iterations=150,
        print_interval=0,
        use_isomorphism=True,
        dump_rounds=0,
        output_path="output_result.json",
    )

def main():
    with tempfile.TemporaryDirectory() as td:
        in_path  = os.path.join(td, "solver_input.txt")
        out_path = os.path.join(td, "result.json")

        req = build_smoke_request()               # your existing builder
        req.output_path = out_path                # <<< absolute path for dump_result
        req.dump_rounds = 1
        req.threads = 1

        with open(in_path, "w") as f:
            _write_solver_input(f, req)

        cmd = ["./external/solver/console_solver", "-i", in_path]
        print("🚀", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print("— stdout —\n" + proc.stdout)
        print("— stderr —\n" + proc.stderr)

        if proc.returncode != 0:
            raise SystemExit(f"Solver failed rc={proc.returncode}")

        # Verify and show a tiny summary
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Expected result at {out_path}")

        with open(out_path, "r") as f:
            res = json.load(f)

        # Print a couple of keys so we see it's real
        print("✅ Parsed result keys:", list(res.keys())[:8])
        # e.g. res.get("root_strategy") / res.get("exploitability") etc.

if __name__ == "__main__":
    main()