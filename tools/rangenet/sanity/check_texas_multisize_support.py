import os, json, gzip, subprocess, tempfile, sys
from pathlib import Path

import dotenv

dotenv.load_dotenv()
SOLVER = os.environ.get("SOLVER_BIN", "console_solver")

CMD_HEADER = """\
set_pot 6.5
set_effective_stack 60
set_board Ah,Kd,2s
set_range_ip AA:1.0,KK:1.0,QQ:1.0,AKs:1.0
set_range_oop AA:1.0,KK:1.0,QQ:1.0,AKs:1.0
"""

CMD_FOOTER = """\
set_allin_threshold 0.67
build_tree
set_thread_num 1
set_accuracy 0.5
set_max_iteration 40
set_print_interval 10
set_use_isomorphism 1
start_solve
set_dump_rounds 1
dump_result {out}
"""

CASES = {
    # 1) one line, comma separated
    "comma_one_line": "set_bet_sizes ip,flop,bet,25,50\n",
    # 2) one line, space separated (some builds parse this)
    "space_one_line": "set_bet_sizes ip,flop,bet,25 50\n",
    # 3) two lines, hoping it appends vs overwrites
    "two_lines": "set_bet_sizes ip,flop,bet,25\nset_bet_sizes ip,flop,bet,50\n",
}

def run_case(name, body, tmpdir):
    out = Path(tmpdir)/f"{name}.json"
    cmd_text = CMD_HEADER + body + CMD_FOOTER.format(out=str(out))
    cmd_file = Path(tmpdir)/f"{name}.txt"
    cmd_file.write_text(cmd_text, encoding="utf-8")

    proc = subprocess.run([SOLVER, "-i", str(cmd_file)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[{name}] solver rc={proc.returncode}")
        print(proc.stderr)
        return name, None, "solver_failed"

    payload = None
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
    elif Path(str(out)+".gz").exists():
        with gzip.open(str(out)+".gz", "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        return name, None, "no_output_json"

    root = payload.get("root", payload)
    acts = root.get("actions") or []
    acts = [str(a) for a in acts]
    has25 = any("BET" in a and "25" in a for a in acts)
    has50 = any("BET" in a and "50" in a for a in acts)
    return name, {"actions": acts, "has25": has25, "has50": has50}, "ok"

def main():
    with tempfile.TemporaryDirectory() as d:
        results = []
        for k, body in CASES.items():
            results.append(run_case(k, body, d))

    print("\nRESULTS")
    for name, data, status in results:
        if status != "ok":
            print(f" - {name}: {status}")
            continue
        print(f" - {name}: actions={data['actions']}")
        print(f"   → found BET 25? {data['has25']} | BET 50? {data['has50']}")

    print("\nINTERPRETATION")
    print(" - If only one BET appears in every case: your CLI behaves single-size (overwrites or snaps).")
    print(" - If comma or space case shows both: multi-size supported; use that format.")
    print(" - If two-lines shows both: the solver APPENDS on repeated set_bet_sizes; emit one line per size.")

if __name__ == "__main__":
    main()