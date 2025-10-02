#!/usr/bin/env python3
import json, gzip, io, sys
from pathlib import Path
from typing import Mapping, Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

# --- import your parser + helpers from your codebase ---
# Adjust the import path if needed
from ml.etl.rangenet.postflop.solver_policy_parser import (
    SolverPolicyParser, ACTION_VOCAB,
    root_node, get_children, actions_and_mix, resolve_child,
    bucket_bet_label, parse_root_bet_size_bb, parse_raise_to_bb, collect_oop_actions_recursive
)

def load_json_maybe_gz(p: Path) -> Mapping[str, Any]:
    b = p.read_bytes()
    if len(b) >= 2 and b[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
            return json.loads(gz.read().decode("utf-8"))
    return json.loads(b.decode("utf-8"))

def parse_using_map(payload: Mapping[str, Any], solve_map: Mapping[str, Any], *,
                    bet_sizing_id: str, pot_bb: float, stack_bb: float,
                    action_vocab: list[str]) -> tuple[list[float], dict, bool]:
    """Uses the pre-built map (entries/patterns) to aggregate OOP responses reliably."""
    idx = {a: i for i, a in enumerate(action_vocab)}
    vec = [0.0] * len(action_vocab)

    root = root_node(payload)
    acts, mix = actions_and_mix(root)
    if not acts or not mix:
        return vec, {"reason": "zero_mass_root"}, False

    m = (solve_map or {}).get(bet_sizing_id)
    if not m:
        return vec, {"reason": "no_map_for_menu"}, False

    entries = m.get("entries") or []
    children_root = get_children(root) or {}

    any_mass = False
    any_raise = False

    def _add(k, w):
        if k in idx:
            vec[idx[k]] += float(w)

    for ent in entries:
        pid = (ent.get("pattern_id") or "").upper()

        if pid == "ROOT_OOP_DONK":
            # OOP BETs at root (bucket as BET_%)
            for a, p in zip(acts, mix):
                if p <= 0: continue
                u = str(a).upper()
                if u.startswith("BET"):
                    b = bucket_bet_label(u, pot_bb=pot_bb)
                    _add(b, p); any_mass = True

        elif pid == "ROOT_IP_BETS":
            # Root IP BET → OOP one-ply responses
            max_bb = parse_root_bet_size_bb(acts, pot_bb) or 1.0
            for a, p in zip(acts, mix):
                if p <= 0: continue
                u = str(a).upper()
                if not u.startswith("BET"): continue
                node_bet = resolve_child(children_root, a)
                if not node_bet: continue
                facing = parse_raise_to_bb(u, pot_bb=pot_bb, bet_size_bb=max_bb) or max_bb
                masses = _collect_one_ply(node_bet, float(p), pot_bb=pot_bb, facing_bet_bb=facing, stack_bb=stack_bb)
                if masses:
                    any_mass = True
                    for k, w in masses.items():
                        _add(k, w)
                        if k.startswith("RAISE") or k == "ALLIN": any_raise = True

        elif pid == "ROOT_IP_BETS_DEEP":
            # Root IP BET → OOP responses + deeper re-raises
            max_bb = parse_root_bet_size_bb(acts, pot_bb) or 1.0
            for a, p in zip(acts, mix):
                if p <= 0: continue
                u = str(a).upper()
                if not u.startswith("BET"): continue
                node_bet = resolve_child(children_root, a)
                if not node_bet: continue
                facing = parse_raise_to_bb(u, pot_bb=pot_bb, bet_size_bb=max_bb) or max_bb
                masses = _collect_deep_raises(node_bet, float(p), pot_bb=pot_bb, facing_bet_bb=facing, stack_bb=stack_bb)
                if masses:
                    any_mass = True
                    for k, w in masses.items():
                        _add(k, w)
                        if k.startswith("RAISE") or k == "ALLIN": any_raise = True

        elif pid == "OOP_CHECK_THEN_IP_BET":
            # Root CHECK → child (IP node) → IP BET → OOP one-ply
            for a, p in zip(acts, mix):
                if p <= 0: continue
                u = str(a).upper()
                if not u.startswith("CHECK"): continue
                ip_node = resolve_child(children_root, a)
                if not ip_node: continue
                ip_acts, ip_mix = actions_and_mix(ip_node) or ([], [])
                ch_ip = get_children(ip_node) or {}
                for lab, q in zip(ip_acts, ip_mix):
                    if q <= 0: continue
                    uu = str(lab).upper()
                    if not uu.startswith("BET"): continue
                    node_ip_bet = resolve_child(ch_ip, lab)
                    if not node_ip_bet: continue
                    facing = parse_raise_to_bb(uu, pot_bb=pot_bb, bet_size_bb=1.0) or 1.0
                    masses = _collect_one_ply(node_ip_bet, float(p*q), pot_bb=pot_bb, facing_bet_bb=facing, stack_bb=stack_bb)
                    if masses:
                        any_mass = True
                        for k, w in masses.items():
                            _add(k, w)
                            if k.startswith("RAISE") or k == "ALLIN": any_raise = True

        elif pid == "OOP_CHECK_THEN_IP_BET_DEEP":
            # Same as above, but allow deep re-raises
            for a, p in zip(acts, mix):
                if p <= 0: continue
                u = str(a).upper()
                if not u.startswith("CHECK"): continue
                ip_node = resolve_child(children_root, a)
                if not ip_node: continue
                ip_acts, ip_mix = actions_and_mix(ip_node) or ([], [])
                ch_ip = get_children(ip_node) or {}
                for lab, q in zip(ip_acts, ip_mix):
                    if q <= 0: continue
                    uu = str(lab).upper()
                    if not uu.startswith("BET"): continue
                    node_ip_bet = resolve_child(ch_ip, lab)
                    if not node_ip_bet: continue
                    facing = parse_raise_to_bb(uu, pot_bb=pot_bb, bet_size_bb=1.0) or 1.0
                    masses = _collect_deep_raises(node_ip_bet, float(p*q), pot_bb=pot_bb, facing_bet_bb=facing, stack_bb=stack_bb)
                    if masses:
                        any_mass = True
                        for k, w in masses.items():
                            _add(k, w)
                            if k.startswith("RAISE") or k == "ALLIN": any_raise = True

    s = float(sum(vec))
    if s > 0: vec = [x / s for x in vec]
    return vec, {"map_used": True, "any_raise": any_raise}, any_mass

def _collect_one_ply(
    node: Mapping[str, Any],
    weight: float,
    *,
    pot_bb: float,
    facing_bet_bb: float,
    stack_bb: float | None,
) -> Dict[str, float]:
    """
    Collect OOP's immediate options vs a single IP bet child (one ply only):
    CALL / FOLD / ALLIN / bucketed RAISE_*.
    Uses uniform fallback if the node has actions but no mix.
    """
    out: Dict[str, float] = {}
    acts2, mix2 = actions_and_mix(node)
    if not acts2:
        return out
    # ✅ uniform fallback if mix missing/zero
    if (not mix2) or sum(mix2) <= 0:
        mix2 = [1.0 / len(acts2)] * len(acts2)

    from ml.etl.rangenet.postflop.solver_policy_parser import bucket_raise_label

    for lab2, p2 in zip(acts2, mix2):
        w = float(weight) * float(p2)
        if w <= 0:
            continue
        u2 = str(lab2).upper()
        if u2.startswith("CALL"):
            out["CALL"] = out.get("CALL", 0.0) + w
        elif u2.startswith("FOLD"):
            out["FOLD"] = out.get("FOLD", 0.0) + w
        elif ("ALLIN" in u2) or ("ALL-IN" in u2) or ("JAM" in u2):
            out["ALLIN"] = out.get("ALLIN", 0.0) + w
        elif u2.startswith("RAISE") or any(tok in u2 for tok in ("RE-RAISE", "RERAISE", "MINRAISE", "MIN-RAISE")):
            b = bucket_raise_label(
                u2,
                pot_bb=pot_bb,
                facing_bet_bb=facing_bet_bb,
                stack_bb=stack_bb,
            )
            out[b] = out.get(b, 0.0) + w
    return out

def _collect_deep_raises(
    node: Mapping[str, Any],
    weight: float,
    *,
    pot_bb: float,
    facing_bet_bb: float,
    stack_bb: float | None,
) -> Dict[str, float]:
    """
    Like one-ply, but if OOP raises, descend into the *raise child* and keep
    collecting (updating facing_bet to the raise-to size each time).
    Uses uniform fallback if the node has actions but no mix.
    """
    out: Dict[str, float] = {}
    acts2, mix2 = actions_and_mix(node)
    if not acts2:
        return out
    # ✅ uniform fallback if mix missing/zero
    if (not mix2) or sum(mix2) <= 0:
        mix2 = [1.0 / len(acts2)] * len(acts2)

    ch2 = get_children(node) or {}
    from ml.etl.rangenet.postflop.solver_policy_parser import bucket_raise_label, parse_raise_to_bb

    for lab2, p2 in zip(acts2, mix2):
        w = float(weight) * float(p2)
        if w <= 0:
            continue
        u2 = str(lab2).upper()

        if u2.startswith("CALL"):
            out["CALL"] = out.get("CALL", 0.0) + w
        elif u2.startswith("FOLD"):
            out["FOLD"] = out.get("FOLD", 0.0) + w
        elif ("ALLIN" in u2) or ("ALL-IN" in u2) or ("JAM" in u2):
            out["ALLIN"] = out.get("ALLIN", 0.0) + w
        elif u2.startswith("RAISE") or any(tok in u2 for tok in ("RE-RAISE", "RERAISE", "MINRAISE", "MIN-RAISE")):
            b = bucket_raise_label(
                u2,
                pot_bb=pot_bb,
                facing_bet_bb=facing_bet_bb,
                stack_bb=stack_bb,
            )
            out[b] = out.get(b, 0.0) + w

            # Recurse under the raise child: update facing_bet to the *raise-to* size
            ch = resolve_child(ch2, lab2)
            if isinstance(ch, dict) and ch:
                raise_to = parse_raise_to_bb(u2, pot_bb=pot_bb, bet_size_bb=facing_bet_bb) or facing_bet_bb
                deeper = _collect_deep_raises(
                    ch,
                    w,
                    pot_bb=pot_bb,
                    facing_bet_bb=raise_to,
                    stack_bb=stack_bb,
                )
                for k, v in deeper.items():
                    out[k] = out.get(k, 0.0) + v

    return out

def main():
    import argparse
    ap = argparse.ArgumentParser("Sanity: loop debug samples and report any_raise using solve_maps.json")
    ap.add_argument("--maps", required=True, help="data/artifacts/solve_maps.json")
    ap.add_argument("--inputs", required=True, help="data/artifacts/solve_map_inputs.json")
    ap.add_argument("--pot", type=float, default=20.0)
    ap.add_argument("--stack", type=float, default=100.0)
    args = ap.parse_args()

    maps = json.loads(Path(args.maps).read_text())
    cfg = json.loads(Path(args.inputs).read_text())
    inputs = cfg.get("inputs") or {}

    parser = SolverPolicyParser(ACTION_VOCAB)

    passed = 0; total = 0
    with_raise = []
    without_raise = []

    for menu_id, path in inputs.items():
        total += 1
        try:
            payload = load_json_maybe_gz(Path(path))
            vec, dbg, ok = parse_using_map(
                payload, maps,
                bet_sizing_id=menu_id,
                pot_bb=args.pot,
                stack_bb=args.stack,
                action_vocab=ACTION_VOCAB
            )
            any_raise = any(vec[ACTION_VOCAB.index(k)] > 1e-9 for k in ("RAISE_150","RAISE_200","RAISE_300","ALLIN"))
            if ok:
                passed += 1
                print(f"[{menu_id}] PASS  any_raise={any_raise}  dbg={dbg}")
                (with_raise if any_raise else without_raise).append(menu_id)
            else:
                print(f"[{menu_id}] FAIL  dbg={dbg}")
        except Exception as e:
            print(f"[{menu_id}] ERROR: {e}")

    print(f"\nSummary: {passed}/{total} passed")
    if with_raise:   print("→ Menus with raises captured:", ", ".join(with_raise))
    if without_raise:print("→ Menus without raises captured:", ", ".join(without_raise))

if __name__ == "__main__":
    main()