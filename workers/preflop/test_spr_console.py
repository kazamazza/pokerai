# workers/preflop/test_console_solver.py

import json, tempfile, subprocess
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.types import SolverRequest, BetKey, SizeVal
from simulation.solver_interface import _write_solver_input  # your existing helpers

def build_realistic_srp_request() -> SolverRequest:
    """
    Production-mimic SRP: single-size tree (33/66/100) + river all-in only,
    same solver knobs as build_solver_request_from_cluster.
    BTN vs BB @ 100bb; ranges kept from your demo (doesn't affect timing much).
    """
    # Pot and stack in BB (same as before)
    pot_bb = 6.5
    stack_bb = 100.0

    # Any 3-card flop is fine; keeping your example
    board = ["Qs", "Jh", "2h"]

    # Your illustrative weighted ranges (unchanged)
    ip_range = (
        "AA,KK,QQ,JJ,TT,99:0.75,88:0.75,77:0.5,66:0.25,55:0.25,"
        "AKs,AQs,AQo:0.75,AJs,AJo:0.5,ATs:0.75,A6s:0.25,A5s:0.75,A4s:0.75,A3s:0.5,A2s:0.5,"
        "KQs,KQo:0.5,KJs,KTs:0.75,K5s:0.25,K4s:0.25,"
        "QJs:0.75,QTs:0.75,Q9s:0.5,"
        "JTs:0.75,J9s:0.75,J8s:0.75,"
        "T9s:0.75,T8s:0.75,T7s:0.75,"
        "98s:0.75,97s:0.75,96s:0.5,87s:0.75,86s:0.5,85s:0.5,76s:0.75,75s:0.5,65s:0.75,64s:0.5,54s:0.75,53s:0.5,43s:0.5"
    )
    oop_range = (
        "QQ:0.5,JJ:0.75,TT,99,88,77,66,55,44,33,22,"
        "AKo:0.25,AQs,AQo:0.75,AJs,AJo:0.75,ATs,ATo:0.75,"
        "A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
        "KQ,KJ,KTs,KTo:0.5,K9s,K8s,K7s,K6s,K5s:0.5,K4s:0.5,K3s:0.5,K2s:0.5,"
        "QJ,QTs,Q9s,Q8s,Q7s,"
        "JTs,JTo:0.5,J9s,J8s,"
        "T9s,T8s,T7s,98s,97s,96s,87s,86s,76s,75s,65s,64s,54s,53s,43s"
    )

    # EXACT production bet tree: one bet size per street, river all-in only, no raises
    bet_sizes: Dict[BetKey, SizeVal] = {
        ("flop",  "oop", "bet"):   0.33,
        ("flop",  "ip",  "bet"):   0.33,
        ("turn",  "oop", "bet"):   0.66,
        ("turn",  "ip",  "bet"):   0.66,
        ("river", "oop", "bet"):   1.00,
        ("river", "ip",  "bet"):   1.00,
        ("river", "oop", "allin"): None,
        ("river", "ip",  "allin"): None,
    }

    # You can keep a temp output path for inspection; performance is identical either way
    out_dir = Path(tempfile.mkdtemp(prefix="solver_demo_"))
    out_json = out_dir / "result.json"

    return SolverRequest(
        pot_size=float(pot_bb),
        stack_depth=float(stack_bb),
        board=board,
        ip_range=ip_range,
        oop_range=oop_range,
        position="IP",              # same as prod builder sets based on hero_role
        hero_cards=[],              # unused by console input; fine to leave empty
        bet_sizes=bet_sizes,

        # Solver knobs — mirrored from build_solver_request_from_cluster
        allin_threshold=0.67,
        threads=1,
        accuracy=0.5,
        max_iterations=200,
        print_interval=10,
        use_isomorphism=True,
        dump_rounds=2,
        output_path=str(out_json),
    )

def main():
    req = build_realistic_srp_request()

    # Prepare input script file
    tmp_dir = Path(tempfile.mkdtemp(prefix="solver_input_"))
    in_path = tmp_dir / "solver_input.txt"
    with in_path.open("w") as f:
        _write_solver_input(f, req)

    print(f"🚀 ./external/solver/console_solver -i {in_path}")
    proc = subprocess.run(
        ["./external/solver/console_solver", "-i", str(in_path)],
        capture_output=True, text=True
    )
    print("— stdout —"); print(proc.stdout)
    print("— stderr —"); print(proc.stderr)

    if proc.returncode != 0:
        print(f"❌ Solver exited with code {proc.returncode}")
    else:
        out_path = Path(req.output_path)
        if out_path.exists():
            print(f"✅ Result file: {out_path}")
            # show a tiny peek so you know it's there & non-empty
            with out_path.open() as f:
                data = json.load(f)
            print(f"🔑 Top-level keys: {list(data.keys())[:10]}")
            # if you want, keep files; else uncomment to clean:
            # shutil.rmtree(out_path.parent, ignore_errors=True)
        else:
            print("⚠️ Solver succeeded but result file missing.")

if __name__ == "__main__":
    main()