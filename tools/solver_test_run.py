import os, sys, tempfile, subprocess, json, gzip, shutil
from pathlib import Path
from typing import Dict, Any, Tuple

# --- import your project helpers ---
# Adjust these imports if your paths differ
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from ml.config.bet_menus import build_contextual_bet_sizes
from ml.etl.rangenet.postflop.helpers_topology import _menu_for
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor
from ml.range.solvers.command_text import build_command_text

# ---- CONFIG ----
CONSOLE_SOLVER = os.environ.get("SOLVER_BIN", "console_solver")  # path to texas solver binary
STAKE          = os.environ.get("STAKE", "NL10")                  # NL10 or NL25
TMP_DIR        = os.environ.get("SMOKE_TMP", "data/ts_smoke")

# one dummy, tiny range each side (keep files small & fast)
DUMMY_RANGE_IP  = "AA:1.0,KK:1.0"
DUMMY_RANGE_OOP = "AA:1.0,KK:1.0"

# tiny solve settings to keep this FAST
SOLVE_CFG = dict(
    allin_threshold=0.67,
    thread_num=1,
    accuracy=0.5,         # coarse
    max_iteration=120,    # small
    print_interval=20,
    use_isomorphism=1,
)

# boards, stacks, pots: 1 example per scenario is enough for smoke test
BOARD = "QsJh2h"
STACK_BB = 60.0
# Pot will be recomputed by your manifest builder in production; here keep simple “SRP-ish”
POT_BB = 7.5

# (ctx, ip, oop, opener, three_bettor, menu_tag)
SCENARIOS = [
    ("VS_OPEN",       "BTN", "BB",  "BTN",  None, "srp_ip"),
    ("VS_OPEN",       "IP",  "OOP", None,   None, "srp_oop"),        # OOP caller donk menu
    ("BLIND_VS_STEAL","BTN", "BB",  "BTN",  None, "bvs"),
    ("VS_3BET",       "BTN", "BB",  None,   "BTN", "3bet_ip"),
    ("VS_3BET",       "BB",  "BTN", None,   "BB",  "3bet_oop"),
    ("VS_4BET",       "BTN", "BB",  None,   "BTN", "4bet"),
    ("LIMPED_SINGLE", "SB",  "BB",  None,   None,  "limp"),
    ("LIMPED_MULTI",  "BTN", "BB",  None,   None,  "limp"),
]

# basic success criteria
ROOT_OK_TOKENS   = {"BET_25","BET_33","BET_50","BET_66","BET_75","BET_100","DONK_33","CHECK"}
FACING_OK_TOKENS = {"CALL","FOLD","RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"}

def run_solver(cmd_text: str, dump_path: Path) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_file = dump_path.with_suffix(".txt")
    cmd_file.write_text(cmd_text, encoding="utf-8")

    # run solver
    proc = subprocess.run(
        [CONSOLE_SOLVER, "-i", str(cmd_file)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solver failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    if not dump_path.exists():
        # some builds write .json then gzip externally; handle both
        gz = Path(str(dump_path) + ".gz")
        if gz.exists():  # ok
            return
        raise FileNotFoundError(f"expected dump at {dump_path} (or .gz) not found")

def load_json_any(p: Path) -> Dict[str, Any]:
    if p.suffix == ".gz" or p.name.endswith(".json.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    gz = Path(str(p) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(str(p))

def check_mixes(root_mix: Dict[str,float], facing_mix: Dict[str,float]) -> Tuple[bool,str]:
    if not root_mix:
        return False, "empty root mix"
    if not facing_mix:
        return False, "empty facing mix"
    root_ok   = any(k in ROOT_OK_TOKENS   and v > 1e-6 for k,v in root_mix.items())
    facing_ok = any(k in FACING_OK_TOKENS and v > 1e-6 for k,v in facing_mix.items())
    if not root_ok:
        return False, f"root lacks any of {sorted(ROOT_OK_TOKENS)} (got {sorted([k for k,v in root_mix.items() if v>1e-6])})"
    if not facing_ok:
        return False, f"facing lacks any of {sorted(FACING_OK_TOKENS)} (got {sorted([k for k,v in facing_mix.items() if v>1e-6])})"
    return True, "ok"

def main():
    tmp = Path(TMP_DIR)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    x = TexasSolverExtractor()
    failures = []

    for idx, (ctx, ip, oop, opener, three_bettor, tag) in enumerate(SCENARIOS, 1):
        print(f"\n=== {ctx} ({ip} vs {oop}) ===")

        # menu + sizes for this stake
        menu_id, _ = _menu_for(ctx, ip, oop, opener, three_bettor, menu_tag=tag, stake=STAKE)
        sizes = build_contextual_bet_sizes(menu_id, stakes=STAKE)

        out_name = f"{menu_id.replace('.','_')}_{idx}.json"
        out_path = tmp / out_name

        cmd = build_command_text(
            pot_bb=POT_BB,
            effective_stack_bb=STACK_BB,
            board=BOARD,
            range_ip=DUMMY_RANGE_IP,
            range_oop=DUMMY_RANGE_OOP,
            bet_sizes=sizes,
            dump_path=str(out_path),
            **SOLVE_CFG,
        )
        print(cmd)

        try:
            run_solver(cmd, out_path)
            payload = load_json_any(out_path)

            ex = x.extract(
                str(out_path),
                ctx=ctx,
                ip_pos=ip,
                oop_pos=oop,
                board=BOARD,
                pot_bb=POT_BB,
                stack_bb=STACK_BB,
                bet_sizing_id=menu_id,
            )

            if not ex.ok:
                raise RuntimeError(f"extract failed: {ex.reason} meta={ex.meta}")

            ok, why = check_mixes(ex.root_mix, ex.facing_mix)
            if ok:
                def top3(m):
                    return ", ".join([f"{k}:{v:.3f}" for k,v in sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:3]])
                print(f"  PASS | menu_id={menu_id}  root=[{top3(ex.root_mix)}]  facing=[{top3(ex.facing_mix)}]  via={ex.meta.get('facing_path')}")
            else:
                raise AssertionError(why)

        except Exception as e:
            failures.append((ctx, ip, oop, str(e)))
            print(f"  FAIL | {e}")

    print("\n--- SUMMARY ---")
    if failures:
        for ctx, ip, oop, err in failures:
            print(f"❌ {ctx} {ip}v{oop} -> {err}")
        sys.exit(1)
    else:
        print("✅ All scenarios passed the smoke test.")

if __name__ == "__main__":
    main()