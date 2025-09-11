import sys
from pathlib import Path
from typing import Optional, Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

#!/usr/bin/env python3
import argparse, json, re, subprocess, tempfile
from pathlib import Path
import numpy as np

from ml.etl.utils.monker_range_converter import to_monker, monker_to_vec169
from ml.range.solvers.command_text import build_command_text

def _parse_bet_sizes_json(s: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Expect shape:
    {
      "flop": {
        "ip":  { "bet": [33, 75], "raise": [100], "allin": true },
        "oop": { "bet": [50] }
      },
      "turn": { ... },
      "river": { ... }
    }
    """
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("bet-sizes JSON must be an object")

    def _is_list_of_ints(x):
        return isinstance(x, list) and all(isinstance(v, (int, float)) for v in x)

    for street, per_role in obj.items():
        if street not in ("flop", "turn", "river"):
            raise ValueError(f"invalid street '{street}' in bet-sizes JSON")
        if not isinstance(per_role, dict):
            raise ValueError(f"street '{street}' must map to an object")
        for role, kinds in per_role.items():
            if role not in ("ip", "oop"):
                raise ValueError(f"invalid role '{role}' in bet-sizes JSON")
            if not isinstance(kinds, dict):
                raise ValueError(f"{street}.{role} must be an object")
            for k, v in kinds.items():
                if k in ("bet", "raise", "donk"):
                    if not _is_list_of_ints(v):
                        raise ValueError(f"{street}.{role}.{k} must be a list of numbers")
                elif k == "allin":
                    if not isinstance(v, bool):
                        raise ValueError(f"{street}.{role}.allin must be boolean")
                else:
                    raise ValueError(f"unknown key '{k}' under {street}.{role}")
    return obj


def main():
    import argparse
    ap = argparse.ArgumentParser(description="One-off demo solve using local ranges")
    ap.add_argument("--ip", required=True, help="Path to IP range (Monker CSV string or 169 form)")
    ap.add_argument("--oop", required=True, help="Path to OOP range")
    ap.add_argument("--board", required=True, help="e.g. Ah4sQc")
    ap.add_argument("--pot-bb", type=float, required=True)
    ap.add_argument("--stack-bb", type=float, required=True)
    # Either raw lines or JSON:
    ap.add_argument("--bet-sizes", type=str, default=None,
                    help="Raw console lines joined by \\n (e.g. 'set_bet_sizes ip,flop,bet,33\\nset_bet_sizes oop,flop,bet,50')")
    ap.add_argument("--bet-sizes-json", type=str, default=None,
                    help='JSON dict for sizes: {"flop":{"ip":{"bet":[33]},"oop":{"bet":[50]}}}')
    ap.add_argument("--accuracy", type=float, default=0.25)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--worker-bin", default="external/worker/console_solver")
    ap.add_argument("--out", default="demo_result.json")
    ap.add_argument("--dump-cmd", default="demo_commands.txt")
    args = ap.parse_args()

    # Load ranges as strings; we rely on worker’s to_monker inside build_command_text caller in your pipeline,
    # but for this demo we can pass the files directly (Monker CSV is fine).
    ip_text  = Path(args.ip).read_text(encoding="utf-8").strip()
    oop_text = Path(args.oop).read_text(encoding="utf-8").strip()

    # Prepare bet sizes argument (either dict or raw string)
    bet_sizes_arg: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    bet_sizes_raw: Optional[str] = None
    if args.bet_sizes_json:
        bet_sizes_arg = _parse_bet_sizes_json(args.bet_sizes_json)
    elif args.bet_sizes:
        # keep raw; demo path will bypass dict emission and append raw lines
        bet_sizes_raw = args.bet_sizes

    # Build command text
    # If we have a dict, pass it in the documented `bet_sizes` param.
    # If we only have raw lines, pass None and append raw after build_command_text.
    cmd = build_command_text(
        pot_bb=args.pot_bb,
        effective_stack_bb=args.stack_bb,
        board=args.board,
        range_ip=ip_text,
        range_oop=oop_text,
        bet_sizes=bet_sizes_arg,              # dict → auto expanded inside build_command_text
        allin_threshold=0.67,
        thread_num=1,
        accuracy=args.accuracy,
        max_iteration=args.max_iter,
        print_interval=10,
        use_isomorphism=1,
        dump_path=args.out,
    )

    # If raw bet sizes provided, append them after (since build_command_text expects dict)
    if bet_sizes_raw:
        cmd = cmd.replace("\nbuild_tree\n", f"\n{bet_sizes_raw.strip()}\nbuild_tree\n")

    Path(args.dump_cmd).write_text(cmd, encoding="utf-8")
    print(f"📝 wrote command file → {args.dump_cmd}")

    # Run worker
    import subprocess, shlex
    run_cmd = f"{args.solver_bin} -i {shlex.quote(args.dump_cmd)}"
    print(f"▶️  {run_cmd}")
    proc = subprocess.run(shlex.split(run_cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"worker exit {proc.returncode}")
    print("✅ worker finished")
    # optionally print small tail of output
    print(proc.stdout.splitlines()[-5:])

if __name__ == "__main__":
    main()