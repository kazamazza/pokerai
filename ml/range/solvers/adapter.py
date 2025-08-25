from __future__ import annotations

import gzip
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import json, hashlib, subprocess, tempfile, shutil, time

from infra.storage.s3_client import S3Client
from ml.config.types_hands import HAND_TO_ID, RANK_TO_I

# at top of adapter.py (already present)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOLVER_BIN = (PROJECT_ROOT / "external/solver/console_solver").resolve()

_DUMP_RE = re.compile(r"^\s*dump_result\b.*$", re.IGNORECASE | re.MULTILINE)

def _force_dump_path(cmd_text: str, dump_abs: Path) -> str:
    """
    Remove any existing 'dump_result ...' lines and append a single absolute one.
    """
    cleaned = re.sub(_DUMP_RE, "", cmd_text).strip()
    if not cleaned.endswith("\n"):
        cleaned += "\n"
    cleaned += f"dump_result {dump_abs}\n"
    return cleaned

def load_villain_range_from_solver(
    cfg: Dict[str, Any],
    *,
    pot_bb: float,
    effective_stack_bb: float,
    board: str,
    range_ip: str,
    range_oop: str,
    actor: str,
    node_key: str = "flop_root",
    debug: bool = True,
) -> Dict[str, float]:
    solver_bin = SOLVER_BIN
    work_dir   = Path(cfg["solver"].get("work_dir", "data/solver_work"))
    timeout    = int(cfg["solver"].get("timeout_sec", 900))
    upload     = bool(cfg["solver"].get("upload_s3", True))
    s3_prefix  = cfg["solver"].get("s3_prefix", "solver/outputs")
    s3_bucket  = cfg["solver"].get("s3_bucket")  # optional; S3Client will use env if None
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Base command content
    base_cmd = build_command_text(
        pot_bb=pot_bb,
        effective_stack_bb=effective_stack_bb,
        board=board,
        range_ip=range_ip,
        range_oop=range_oop,
    )

    # 2) Cache key based on content + actor/node
    key = sha1_str(base_cmd + f"|actor={actor}|node={node_key}")
    cache_dir = work_dir / "cache" / key
    out_json_cache = cache_dir / "output_result.json"
    if out_json_cache.exists():
        rng = parse_solver_output(out_json_cache, actor=actor, node_key=node_key)
        if rng:
            return rng

    # 3) Temp run dir and absolute file paths
    run_dir = Path(tempfile.mkdtemp(prefix="solve_", dir=work_dir))
    try:
        cmd_file      = (run_dir / "input.txt").resolve()
        out_json_tmp  = (run_dir / "output_result.json").resolve()
        out_json_gz   = out_json_tmp.with_suffix(".json.gz")

        # Ensure we FORCE the dump_result to our run_dir
        cmd_text = _force_dump_path(base_cmd, out_json_tmp)
        cmd_file.write_text(cmd_text, encoding="utf-8")

        # 4) Run solver from project root (matches your manual CLI)
        cmd = [str(solver_bin), "-i", str(cmd_file)]
        if debug:
            print(f"[solver] CWD       : {PROJECT_ROOT}")
            print(f"[solver] BIN       : {solver_bin}")
            print(f"[solver] INPUT     : {cmd_file}")
            print(f"[solver] DUMP PATH : {out_json_tmp}")
            print(f"[solver] CMD       : {' '.join(cmd)}")

        # Ensure executable
        if not os.access(solver_bin, os.X_OK):
            solver_bin.chmod(solver_bin.stat().st_mode | 0o111)

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"solver failed rc={result.returncode}\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 5) Must exist exactly where we forced it
        if not out_json_tmp.exists():
            raise RuntimeError(
                f"Solver finished but output_result.json not found at {out_json_tmp}\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # 6) Parse, then cache the raw JSON
        rng = parse_solver_output(out_json_tmp, actor=actor, node_key=node_key)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json_tmp, out_json_cache)

        # 7) Gzip + S3 upload (if enabled)
        if upload:
            # gzip
            with out_json_tmp.open("rb") as f_in, gzip.open(out_json_gz, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            # upload
            s3 = S3Client(bucket_name=s3_bucket) if s3_bucket else S3Client()
            s3_key = f"{s3_prefix}/{key}/output_result.json.gz"
            print(f"[solver] Uploading to s3://{s3.bucket}/{s3_key}")
            s3.upload_file(out_json_gz, s3_key)

        return rng

    finally:
        if debug:
            print(f"[solver] Debug mode ON, keeping run_dir: {run_dir.resolve()}")
            print(f"[solver] You can inspect: {run_dir.resolve()}/input.txt and output_result.json")
        else:
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
        "set_thread_num 1",
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
    print(f"[solver] Running {solver_bin} with input {cmd_file} (cwd={cwd})")

    result = subprocess.run(
        [str(solver_bin), "-i", str(cmd_file)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    print("=== Solver stdout/stderr ===")
    print(result.stdout)
    print("============================")

    if result.returncode != 0:
        raise RuntimeError(
            f"Solver failed rc={result.returncode}. "
            f"stdout={result.stdout[-500:]}, stderr={result.stderr[-500:]}"
        )

    print(f"[solver] Completed OK. Last 5 lines:\n" + "\n".join(result.stdout.splitlines()[-5:]))


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