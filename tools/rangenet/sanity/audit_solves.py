import argparse, json, gzip, sys, tempfile, re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import boto3

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.utils.solver_parse import actions_and_mix, root_node, get_children

SPR_RAISE_REQUIREMENT_THRESHOLD = 1.4

def parse_meta_from_key(key: str) -> dict:
    """
    Pull {context, stack, pot, spr} from an S3 key like:
    solver/outputs/v1/street=1/pos=BTNvBB/stack=25/pot=20/.../sizes=3bet_hu.Aggressor_OOP/...
    """
    out = {"context": "unknown", "stack": None, "pot": None, "spr": None}
    m_ctx = re.search(r"sizes=([^/]+)/", key)
    if m_ctx: out["context"] = m_ctx.group(1)
    m_stack = re.search(r"/stack=(\d+(?:\.\d+)?)/", key)
    m_pot   = re.search(r"/pot=(\d+(?:\.\d+)?)/", key)
    if m_stack: out["stack"] = float(m_stack.group(1))
    if m_pot:   out["pot"]   = float(m_pot.group(1))
    if out["stack"] and out["pot"] and out["pot"] > 0:
        out["spr"] = out["stack"] / out["pot"]
    return out

def parse_context_from_key(key: str) -> str:
    m = re.search(r"sizes=([^/]+)/", key)
    return m.group(1) if m else "unknown"

def load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def _iter_bet_children_with_weight(node: Dict[str, Any]):
    acts, mix = actions_and_mix(node)
    ch = get_children(node)
    have_mix = bool(mix) and len(mix) == len(acts)

    # from actions[] indices (preferred)
    yielded = set()
    for i, lab in enumerate(acts):
        up = str(lab).upper()
        if up.startswith("BET"):
            child = ch.get(lab) or ch.get(str(lab))
            if isinstance(child, dict):
                w = float(mix[i]) if have_mix else 0.0
                yielded.add(str(lab))
                yield str(lab), child, w

    # fallback: scan dict keys for orphaned BET nodes
    for k, v in ch.items():
        up = str(k).upper()
        if up.startswith("BET") and isinstance(v, dict) and str(k) not in yielded:
            yield str(k), v, 0.0

def _has_raise_somewhere(node: Dict[str, Any], cap: int = 5000) -> bool:
    stack = [node]; seen = 0
    while stack and seen < cap:
        cur = stack.pop(); seen += 1
        acts, _ = actions_and_mix(cur)
        for a in acts:
            up = str(a).upper()
            if up.startswith("RAISE") or up == "ALLIN":
                return True
        for v in get_children(cur).values():
            if isinstance(v, dict):
                stack.append(v)
    return False

# ---------- core audit ----------

def audit_solve(payload: dict, context: str, spr: float | None) -> List[str]:
    """
    Returns a list of problems found. Empty list → ok.
    SPR is used to relax raise requirements in very low-SPR 3/4-bet pots.
    """
    problems: List[str] = []

    root = root_node(payload)
    if not isinstance(root, dict):
        return ["root is not an object"]

    root_actions, root_mix = actions_and_mix(root)
    ch = get_children(root)

    if not root_actions:
        problems.append("missing actions at root")

    if root_mix:
        s = sum(root_mix)
        if abs(s - 1.0) > 1e-3:
            problems.append(f"root mix sums to {s:.3f} (≠1)")

    up_root = [str(a).upper() for a in root_actions]
    has_check = any(a.startswith("CHECK") for a in up_root)
    has_bet   = any(a.startswith("BET")   for a in up_root)
    if not (has_check or has_bet):
        problems.append("no CHECK or BET at root")

    ctx_lower = (context or "").lower()
    heavy_ctx = ("3bet" in ctx_lower) or ("4bet" in ctx_lower)

    if has_bet:
        any_bet_child = False
        any_child_has_call_fold = False
        any_child_has_raise = False

        for _, child, _w in _iter_bet_children_with_weight(root):
            any_bet_child = True
            child_actions, _ = actions_and_mix(child)
            cup = [str(a).upper() for a in child_actions]
            if any(a in ("CALL", "FOLD") for a in cup):
                any_child_has_call_fold = True
            if any(a.startswith("RAISE") or a == "ALLIN" for a in cup):
                any_child_has_raise = True

        if not any_bet_child:
            problems.append("BET at root but no BET child node found")
        else:
            if not any_child_has_call_fold:
                problems.append("facing-bet children missing CALL/FOLD")

            # Require a raise child only if (a) 3/4bet context AND (b) SPR not very low
            require_raise_child = heavy_ctx
            if spr is not None and spr < SPR_RAISE_REQUIREMENT_THRESHOLD:
                require_raise_child = False
            if require_raise_child and not any_child_has_raise:
                problems.append("facing-bet children have no RAISE in 3bet/4bet context")

    # Also require *some* raise anywhere only if heavy context & SPR not very low
    require_raise_anywhere = heavy_ctx
    if spr is not None and spr < SPR_RAISE_REQUIREMENT_THRESHOLD:
        require_raise_anywhere = False
    if require_raise_anywhere and not _has_raise_somewhere(root):
        problems.append("no RAISE found anywhere in 3bet/4bet solve")

    if ch and not root_actions:
        problems.append("children present but no root actions listed")

    return problems

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="S3 bucket name")
    ap.add_argument("--prefix", required=True, help="S3 prefix (e.g. solver/outputs/v1)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of solves")
    args = ap.parse_args()

    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    it = paginator.paginate(Bucket=args.bucket, Prefix=args.prefix)

    results = defaultdict(lambda: {"ok": 0, "fail": 0, "examples": []})
    total = 0
    tmpdir = Path(tempfile.gettempdir())

    for page in it:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("output_result.json.gz"):
                continue

            # parse meta (context/stack/pot/spr) from the key
            meta = parse_meta_from_key(key)
            context = meta["context"]
            spr = meta["spr"]

            local_path = tmpdir / key.replace("/", "_")
            try:
                s3.download_file(args.bucket, key, str(local_path))
            except Exception as e:
                results[context]["fail"] += 1
                if len(results[context]["examples"]) < 3:
                    results[context]["examples"].append((key, [f"s3 download error: {e}"]))
                continue

            try:
                payload = load_json_gz(local_path)
                problems = audit_solve(payload, context, spr)
                if problems:
                    results[context]["fail"] += 1
                    if len(results[context]["examples"]) < 3:
                        results[context]["examples"].append((key, problems))
                else:
                    results[context]["ok"] += 1
            except Exception as e:
                results[context]["fail"] += 1
                if len(results[context]["examples"]) < 3:
                    results[context]["examples"].append((key, [f"exception: {e}"]))
            finally:
                try:
                    local_path.unlink()
                except Exception:
                    pass

            total += 1
            if args.limit and total >= args.limit:
                break
        if args.limit and total >= args.limit:
            break

    # --- summary ---
    print("\n=== AUDIT SUMMARY ===")
    overall_fail = False
    for ctx, stats in results.items():
        print(f"{ctx:25} ok={stats['ok']:4d}  fail={stats['fail']:3d}")
        if stats["fail"]:
            overall_fail = True
            for key, probs in stats["examples"]:
                print(f"   example {key}:")
                for p in probs:
                    print(f"     - {p}")

    if overall_fail:
        sys.exit(1)
    else:
        print("✅ all solves passed sanity checks")

if __name__ == "__main__":
    main()