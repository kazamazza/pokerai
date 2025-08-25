from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import json, hashlib, subprocess, tempfile, shutil, time
from infra.storage.s3_client import S3Client
from ml.config.types_hands import HAND_TO_ID, RANK_TO_I


def load_villain_range_from_solver(
    cfg: Dict[str, Any],
    *,
    pot_bb: float,
    effective_stack_bb: float,
    board: str,                # e.g. "QsJh2h" (no commas)
    range_ip: str,             # Monker-like compact string (AA,KK,QQ,JJ,TT,99:0.75,...)
    range_oop: str,            # same format
    actor: str,                # "ip" or "oop" – whose range we want
    node_key: str = "flop_root",  # optional path within tree; for now we take root
) -> Dict[str, float]:
    """
    Build a one-off command file, run console_solver, parse output_result.json,
    and return a normalized {hand169_code -> prob} for the requested actor at node_key.
    Cached by content hash to avoid re-solving.
    """
    solver_bin = Path(cfg["solver"]["bin"])
    work_dir   = Path(cfg["solver"].get("work_dir", "data/solver_work"))
    timeout    = int(cfg["solver"].get("timeout_sec", 900))
    upload     = bool(cfg["solver"].get("upload_s3", False))

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Command content
    cmd_text = build_command_text(
        pot_bb=pot_bb,
        effective_stack_bb=effective_stack_bb,
        board=board,
        range_ip=range_ip,
        range_oop=range_oop,
    )

    # 2) Cache key (content hash)
    key = sha1_str(cmd_text + f"|actor={actor}|node={node_key}")
    cache_dir = work_dir / "cache" / key
    out_json  = cache_dir / "output_result.json"
    if out_json.exists():
        rng = parse_solver_output(out_json, actor=actor, node_key=node_key)
        if rng:
            return rng

    # 3) Write command file into a temp run dir
    run_dir = Path(tempfile.mkdtemp(prefix="solve_", dir=work_dir))
    try:
        cmd_file = run_dir / "input.txt"
        cmd_file.write_text(cmd_text, encoding="utf-8")

        # ensure output path matches the last line in command text
        # (we always dump to output_result.json in run_dir)
        # already aligned in build_command_text()

        # 4) Run the solver
        run_console_solver(
            solver_bin=solver_bin,
            cmd_file=cmd_file,
            cwd=run_dir,
            timeout=timeout,
        )

        # 5) Parse result
        out_json_tmp = run_dir / "output_result.json"
        if not out_json_tmp.exists():
            raise RuntimeError("Solver finished but output_result.json not found")

        rng = parse_solver_output(out_json_tmp, actor=actor, node_key=node_key)

        # 6) Move to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json_tmp, out_json)

        # 7) Optional upload to S3
        if upload:
            s3 = S3Client()
            bucket = cfg["solver"]["s3_bucket"]
            prefix = cfg["solver"].get("s3_prefix", "solver/outputs")
            s3_key = f"{prefix}/{key}/output_result.json"
            s3.upload_file(str(out_json), bucket, s3_key)

        return rng

    finally:
        # keep cache, remove temp run_dir
        shutil.rmtree(run_dir, ignore_errors=True)

def build_command_text(
    *,
    pot_bb: float,
    effective_stack_bb: float,
    board: str,
    range_ip: str,
    range_oop: str,
) -> str:
    """
    Compose exactly the format your sample shows.
    You can tune bet sizes and accuracy knobs here (or pull from cfg if you prefer).
    """
    # Board needs commas for the solver: "Qs,Jh,2h"
    board_csv = ",".join([board[i:i+2] for i in range(0, len(board), 2)])

    lines = []
    lines.append(f"set_pot {int(pot_bb)}")
    lines.append(f"set_effective_stack {int(effective_stack_bb)}")
    lines.append(f"set_board {board_csv}")
    lines.append(f"set_range_ip {range_ip}")
    lines.append(f"set_range_oop {range_oop}")

    # Minimal sane sizing set (match your example)
    for who in ("oop", "ip"):
        lines.append(f"set_bet_sizes {who},flop,bet,50")
        lines.append(f"set_bet_sizes {who},flop,raise,60")
        lines.append(f"set_bet_sizes {who},flop,allin")
        lines.append(f"set_bet_sizes {who},turn,bet,50")
        lines.append(f"set_bet_sizes {who},turn,raise,60")
        lines.append(f"set_bet_sizes {who},turn,allin")
        lines.append(f"set_bet_sizes {who},river,bet,50")
        if who == "oop":
            lines.append(f"set_bet_sizes {who},river,donk,50")
        lines.append(f"set_bet_sizes {who},river,raise,60,100")
        lines.append(f"set_bet_sizes {who},river,allin")

    lines += [
        "set_allin_threshold 0.67",
        "build_tree",
        "set_thread_num 8",
        "set_accuracy 0.5",
        "set_max_iteration 200",
        "set_print_interval 10",
        "set_use_isomorphism 1",
        "start_solve",
        "set_dump_rounds 2",
        "dump_result output_result.json",
    ]
    return "\n".join(lines) + "\n"


def run_console_solver(
    *,
    solver_bin: Path,
    cmd_file: Path,
    cwd: Path,
    timeout: int,
) -> None:
    """
    Launch: console_solver -i input.txt
    """
    if not solver_bin.exists():
        raise FileNotFoundError(f"solver binary not found: {solver_bin}")
    proc = subprocess.Popen(
        [str(solver_bin), "-i", str(cmd_file)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    start = time.time()
    last = ""
    while True:
        if proc.poll() is not None:
            break
        if time.time() - start > timeout:
            proc.kill()
            raise TimeoutError(f"solver timed out after {timeout}s")
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            last = line.strip()
        else:
            time.sleep(0.05)
    if proc.returncode != 0:
        raise RuntimeError(f"solver failed rc={proc.returncode}. Last line: {last}")


def parse_solver_output(path: Path, *, actor: str, node_key: str) -> Dict[str, float]:
    """
    Expect structure like:
    {
      "node_type":"action_node",
      "childrens": {
        "BET 98.0": {
          "node_type":"action_node",
          "strategy": { "strategy": { "ThTc":[0.79,0.21], "AsAh":[6e-11,1.0], ... } },
          ...
        }
      }
    }
    We:
      1) walk childrens by `node_key` (slash-separated),
      2) read `strategy.strategy` dict,
      3) convert concrete 2-card strings -> 169 codes,
      4) sum per 169 and renormalize.
    """
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    node = _walk_to_node(data, node_key)
    if not node:
        return {}

    strat = None
    st = node.get("strategy")
    if isinstance(st, dict):
        strat = st.get("strategy", None)
    if strat is None:
        # some dumps use actor nesting: node["strategy"][actor]["strategy"]
        st_actor = node.get("strategy", {}).get(actor)
        if isinstance(st_actor, dict):
            strat = st_actor.get("strategy")

    if not isinstance(strat, dict):
        return {}

    # compress concrete combos to 169 codes
    accum: Dict[str, float] = {}
    total = 0.0
    for combo_str, action_probs in strat.items():
        # action_probs is like [p_bet, p_call, ...]. Presence weight ~ sum to ~1
        try:
            w = float(sum(action_probs))
        except Exception:
            continue
        code169 = _combo_to_169(combo_str)
        if code169 is None or code169 not in HAND_TO_ID:
            continue
        accum[code169] = accum.get(code169, 0.0) + w
        total += w

    if total <= 0:
        return {}

    # Normalize
    scale = 1.0 / total
    for k in list(accum.keys()):
        accum[k] *= scale
    return accum


def _walk_to_node(root: dict, node_key: str) -> Optional[dict]:
    """
    node_key like "root/BET 98.0/CALL" → traverse 'childrens'.
    If node_key == "root" or empty → return root.
    """
    if not node_key or node_key.lower() == "root":
        return root
    cur = root
    for name in node_key.split("/"):
        if not name or name.lower() == "root":
            continue
        ch = cur.get("childrens")
        if not isinstance(ch, dict) or name not in ch:
            return None
        cur = ch[name]
    return cur


def _combo_to_169(combo: str) -> Optional[str]:
    """
    "AsAh" -> "AA"
    "AhKh" -> "AKs"
    "AhKd" -> "AKo"
    """
    combo = combo.strip()
    if len(combo) != 4:
        return None
    r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]
    # order ranks high->low per your RANKS ordering
    if RANK_TO_I[r1] > RANK_TO_I[r2]:
        hi, lo, sh, sl = r1, r2, s1, s2
    else:
        hi, lo, sh, sl = r2, r1, s2, s1
    if hi == lo:
        return hi + lo
    suited = (sh == sl)
    return f"{hi}{lo}{'s' if suited else 'o'}"


def _normalize_and_filter_169(raw_map: Dict[str, float]) -> Dict[str, float]:
    """Keep only canonical 169 keys, renormalize to 1.0."""
    out: Dict[str, float] = {}
    s = 0.0
    for k, v in raw_map.items():
        if k in HAND_TO_ID:
            try:
                fv = float(v)
            except Exception:
                continue
            if fv > 0:
                out[k] = fv
                s += fv
    if s <= 0:
        return {}
    scale = 1.0 / s
    for k in list(out.keys()):
        out[k] *= scale
    return out


def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()