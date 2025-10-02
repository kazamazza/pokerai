import json, gzip, io
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

# Import your actual parser + vocab
from ml.etl.rangenet.postflop.solver_policy_parser import SolverPolicyParser, ACTION_VOCAB

def load_json_maybe_gz(p: Path):
    b = p.read_bytes()
    if b[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
            return json.loads(gz.read().decode("utf-8"))
    return json.loads(b.decode("utf-8"))

def main():
    import argparse
    ap = argparse.ArgumentParser("Check debug samples with solve map (no guessing)")
    ap.add_argument("--maps", required=True)    # data/artifacts/solve_maps.json
    ap.add_argument("--inputs", required=True)  # data/artifacts/solve_map_inputs.json
    ap.add_argument("--pot", type=float, default=None)
    ap.add_argument("--stack", type=float, default=100.0)
    args = ap.parse_args()

    solve_map = json.loads(Path(args.maps).read_text())
    cfg = json.loads(Path(args.inputs).read_text())
    pot_hint = float(cfg.get("pot_hint_bb") or args.pot or 1.0)
    stack_hint = float(cfg.get("stack_hint_bb") or args.stack)
    inputs = cfg.get("inputs", {})

    parser = SolverPolicyParser(ACTION_VOCAB)

    passed = 0; total = 0
    for menu_id, path in inputs.items():
        total += 1
        p = Path(path)
        if not p.is_file():
            print(f"[{menu_id}] FAIL  (missing file: {p})")
            continue

        payload = load_json_maybe_gz(p)
        vec, dbg, ok = parser.parse_using_map(
            payload, solve_map,
            bet_sizing_id=menu_id,
            pot_bb=pot_hint,
            stack_bb=stack_hint,
            action_vocab=ACTION_VOCAB,
        )

        raise_mass = 0.0
        for k in ("RAISE_150","RAISE_200","RAISE_300","ALLIN"):
            # guard in case vocab differs
            try:
                idx = ACTION_VOCAB.index(k)
                raise_mass += float(vec[idx])
            except ValueError:
                pass

        status = "PASS" if ok and (raise_mass > 0 or not any(e.get("captures_raise") for e in (solve_map.get(menu_id, {}).get("entries") or []))) else "FAIL"
        if status == "PASS": passed += 1
        print(f"[{menu_id}] {status}  any_raise={raise_mass>1e-9}  dbg={dbg}")

    print(f"\nSummary: {passed}/{total} passed")

if __name__ == "__main__":
    main()