import argparse
import re

import dotenv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from ml.config.bet_menus import build_contextual_bet_sizes
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor
from ml.range.solvers.command_text import build_command_text
from ml.core.types import Stakes
from ml.etl.utils.monker_range_converter import to_monker
import os, sys, gzip, json, shutil, subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from ml.config.solver import STAKE_CFG
import pandas as pd

dotenv.load_dotenv()
CONSOLE_SOLVER = os.environ.get("SOLVER_BIN", "console_solver")
DEFAULT_TMP = os.environ.get("SMOKE_TMP", "data/ts_smoke")

# ——— profiles (seconds vs minutes). Increase for deeper solves.
SOLVE_PROFILES = {
    "smoke": dict(allin_threshold=0.67, thread_num=1, accuracy=0.50, max_iteration=120, print_interval=20,
                  use_isomorphism=1),
    "prod": dict(allin_threshold=0.67, thread_num=8, accuracy=0.03, max_iteration=4000, print_interval=200,
                 use_isomorphism=1),
}

# include 25% in acceptance
ROOT_OK_TOKENS = {"BET_25", "BET_33", "BET_50", "BET_66", "BET_75", "BET_100", "DONK_25", "DONK_33", "CHECK"}
FACING_OK_TOKENS = {"CALL", "FOLD", "RAISE_150", "RAISE_200", "RAISE_250", "RAISE_300", "RAISE_400", "RAISE_500", "ALLIN"}


def stake_from_str(s: str) -> Stakes:
    s = (s or "NL10").upper()
    return {"NL25": Stakes.NL25, "NL10": Stakes.NL10, "NL5": Stakes.NL5, "NL2": Stakes.NL2}.get(s, Stakes.NL10)


def run_solver(cmd_text: str, dump_path: Path) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_file = dump_path.with_suffix(".txt")
    cmd_file.write_text(cmd_text, encoding="utf-8")
    proc = subprocess.run([CONSOLE_SOLVER, "-i", str(cmd_file)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"solver failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    if not dump_path.exists() and not Path(str(dump_path) + ".gz").exists():
        raise FileNotFoundError(f"expected dump at {dump_path} (or .gz) not found")


def load_json_any(p: Path) -> Dict[str, Any]:
    if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
    gz = Path(str(p) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f: return json.load(f)
    raise FileNotFoundError(str(p))


def check_mixes(root_mix: Dict[str, float], facing_mix: Dict[str, float]) -> Tuple[bool, str]:
    if not root_mix:  return False, "empty root mix"
    if not facing_mix: return False, "empty facing mix"
    root_ok = any(k in ROOT_OK_TOKENS and v > 1e-6 for k, v in root_mix.items())
    facing_ok = any(k in FACING_OK_TOKENS and v > 1e-6 for k, v in facing_mix.items())
    if not root_ok:   return False, f"root lacks any of {sorted(ROOT_OK_TOKENS)}"
    if not facing_ok: return False, f"facing lacks any of {sorted(FACING_OK_TOKENS)}"
    return True, "ok"


# ——— audit helpers (look for INTEGER percents in emitted command)
def _collect_bet_lines(cmd: str, prefix: str) -> List[str]:
    pref = prefix.lower()
    return [ln for ln in cmd.splitlines() if ln.strip().lower().startswith(pref)]


def _audit_set_bet_lines(cmd: str, ip_ints: List[int], oop_ints: List[int]) -> Tuple[bool, List[str]]:
    probs: List[str] = []
    ip_lines = _collect_bet_lines(cmd, "set_bet_sizes ip,flop,bet")
    oop_lines = _collect_bet_lines(cmd, "set_bet_sizes oop,flop,bet") + _collect_bet_lines(cmd,
                                                                                           "set_bet_sizes oop,flop,donk")

    def need(lines: List[str], want: List[int], label: str):
        if not want: return
        if not lines:
            probs.append(f"missing line '{label}'");
            return
        joined = ",".join(lines)
        for v in want:
            tok = f",{v}"
            if tok not in joined:
                probs.append(f"{label} missing {v}%")

    need(ip_lines, ip_ints, "ip,flop,bet")
    need(oop_lines, oop_ints, "oop,flop,(bet|donk)")
    return (len(probs) == 0, probs)


def _first_bet_pct_from_meta(meta: Dict[str, Any]) -> Optional[float]:
    path = meta.get("facing_path") or []
    if not path: return None
    m = re.search(r'bet\s+(\d+(?:\.\d+)?)', (path[-1] or "").lower())
    return float(m.group(1)) if m else None  # integer percent labeling in JSON


def main():
    ap = argparse.ArgumentParser("Scenario-selectable smoke test; stack-aware bet sizes + integer audit.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stake", default="NL10")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--per-key", type=int, default=1)
    ap.add_argument("--by", nargs="*",
                    default=["ctx", "bet_sizing_id", "effective_stack_bb", "ip_actor_flop", "oop_actor_flop"])
    ap.add_argument("--tmp-dir", default=DEFAULT_TMP)
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--profile", choices=["smoke", "prod"], default="smoke")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    stake = stake_from_str(args.stake)
    tmp = Path(args.tmp_dir)
    if tmp.exists() and not args.keep_tmp:
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest)
    if "ctx" not in df.columns: sys.exit("Manifest missing 'ctx'")

    df["ctx"] = df["ctx"].astype(str).str.upper()
    if args.scenarios:
        df = df[df["ctx"].isin({s.upper() for s in args.scenarios})]

    for c in args.by:
        if c not in df.columns:
            df[c] = ""

    # stratified sampling
    parts = []
    rnd = np.random.RandomState(42)
    for key, g in df.groupby(args.by):
        n = min(args.per_key, len(g))
        if n > 0: parts.append(g.sample(n=n, random_state=rnd))
    sample = pd.concat(parts, ignore_index=True) if parts else df.head(0)
    if sample.empty:
        print("No rows after stratification.");
        sys.exit(2)

    x = TexasSolverExtractor()
    failures = []
    rows_out = []

    SOLVE_CFG = SOLVE_PROFILES[args.profile]

    for _, r in sample.iterrows():
        ctx = str(r["ctx"])
        ip = str(r.get("ip_actor_flop") or str(r.get("positions", "BTNvBB")).split("v")[0])
        oop = str(r.get("oop_actor_flop") or str(r.get("positions", "BTNvBB")).split("v")[1])
        board = str(r.get("board") or "QsJh2h")
        pot_bb = float(r.get("pot_bb", 7.5) or 7.5)
        stack_bb = float(r.get("effective_stack_bb", 60.0) or 60.0)
        menu_id = str(r.get("bet_sizing_id") or "")

        # STACK-AWARE sizes (CRUCIAL): pass effective_stack_bb
        menu_cfg = build_contextual_bet_sizes(menu_id, stake=stake, effective_stack_bb=stack_bb)
        # fractions for extractor bucketing
        ip_fracs = list((menu_cfg.get("flop", {}).get("ip", {}) or {}).get("bet", []))
        oop_fracs = list((menu_cfg.get("flop", {}).get("oop", {}) or {}).get("bet", [])) or \
                    list((menu_cfg.get("flop", {}).get("oop", {}) or {}).get("donk", []))
        # integers for emission audit
        ip_ints = [int(round(100 * x)) for x in ip_fracs]
        oop_ints = [int(round(100 * x)) for x in oop_fracs]

        range_ip = (r.get("range_ip"))
        range_oop = (r.get("range_oop"))

        out_name = f"{ctx}_{ip}v{oop}_{menu_id.replace('.', '_')}_{pot_bb:.2f}bb.json"
        out_path = tmp / out_name

        print(f"\n=== {ctx} ({ip} vs {oop}) menu={menu_id} stack={stack_bb:.0f} pot={pot_bb:.2f}bb ===")
        print(f"legalized: ip={ip_ints}% oop={oop_ints}%")

        cmd = build_command_text(
            pot_bb=pot_bb,
            effective_stack_bb=stack_bb,
            board=board,
            range_ip=to_monker(range_ip),
            range_oop=to_monker(range_oop),
            bet_sizes=menu_cfg,  # builder must emit integer percents from these fractions
            dump_path=str(out_path),
            **SOLVE_CFG,
        )
        ok_cmd, issues = _audit_set_bet_lines(cmd, ip_ints, oop_ints)
        if not ok_cmd:
            print("❌ AUDIT |", " | ".join(issues))
        print(cmd)

        try:
            run_solver(cmd, out_path)
            _ = load_json_any(out_path)

            ex = x.extract(
                str(out_path),
                ctx=ctx, ip_pos=ip, oop_pos=oop, board=board,
                pot_bb=pot_bb, stack_bb=stack_bb, bet_sizing_id=menu_id,
                bet_sizes=ip_fracs or oop_fracs,  # extractor buckets to these
                raise_mults=STAKE_CFG[stake]["raise_mult"],
            )
            if not ex.ok:
                raise RuntimeError(f"extract failed: {ex.reason} meta={ex.meta}")

            ok, why = check_mixes(ex.root_mix, ex.facing_mix)
            first_pct = _first_bet_pct_from_meta(ex.meta)
            if first_pct is not None:
                exp_all = (ip_ints or oop_ints)
                nearest = min(exp_all, key=lambda v: abs(v - first_pct)) if exp_all else None
                if nearest is not None and abs(nearest - first_pct) > 5:
                    print(f"[WARN] solver used ~{first_pct:.1f}% vs menu {nearest}%")

            if ok:
                def top3(m):
                    return ", ".join(
                        [f"{k}:{v:.3f}" for k, v in sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:3]])

                print("  PASS |", why)
                print(f"    root:   {top3(ex.root_mix)}")
                print(f"    facing: {top3(ex.facing_mix)}  via={ex.meta.get('facing_path')}")
            else:
                raise AssertionError(why)

            rows_out.append({
                "ctx": ctx, "ip": ip, "oop": oop, "stack": stack_bb, "menu": menu_id,
                "ip_sizes_pct": ";".join(map(str, ip_ints)), "oop_sizes_pct": ";".join(map(str, oop_ints)),
                "root_top": sorted(ex.root_mix.items(), key=lambda kv: kv[1], reverse=True)[:3],
                "facing_top": sorted(ex.facing_mix.items(), key=lambda kv: kv[1], reverse=True)[:3],
                "first_bet_pct": first_pct,
            })

        except Exception as e:
            failures.append((ctx, ip, oop, str(e)))
            print(f"  FAIL | {e}")

    if args.report and rows_out:
        pd.DataFrame(rows_out).to_csv(args.report, index=False)
        print(f"\n📄 Wrote report → {args.report}")

    print("\n--- SUMMARY ---")
    if failures:
        for ctx, ip, oop, err in failures:
            print(f"❌ {ctx} {ip}v{oop} -> {err}")
        sys.exit(1)
    else:
        print("✅ All selected scenarios passed.")


if __name__ == "__main__":
    main()