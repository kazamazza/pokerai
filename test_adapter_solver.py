# test_adapter_solver.py
import time
from pathlib import Path

# import your adapter
from ml.range.solvers.adapter import load_villain_range_from_solver  # adjust import to your tree

def main():
    # --- Adapter config (what your adapter expects) ---
    cfg = {
        "solver": {
            # Path to your binary
            "bin": "external/solver/console_solver",
            # Where to place temporary command files and solver outputs
            "work_dir": "data/solver_runs",
            # Optional: solver knobs (the adapter should map these to the text commands)
            "threads": 1,           # keep 1 if each worker has 1 vCPU
            "accuracy": 0.5,        # rough, faster; lower is more accurate but slower
            "max_iterations": 60,   # a bit more than the super-quick smoke test
            "print_interval": 10,   # sparse logging
            # Optional: default bet/raise grids if your adapter uses them
            "bet_sizes": {
                # percentages of pot as ints, adapter should translate appropriately
                "oop_flop_bet": [33, 50],
                "oop_flop_raise": [60],
                "ip_flop_bet": [33, 50],
                "ip_flop_raise": [60],
                # (Turn/River can be added if your adapter supports multi-street)
            },
            # Optional: all-in threshold if your solver uses it
            "allin_threshold": 0.67,
        }
    }

    Path(cfg["solver"]["work_dir"]).mkdir(parents=True, exist_ok=True)

    # --- Scenario (moderately more complex than the tiny smoke test) ---
    pot_bb = 20.0
    effective_stack_bb = 200.0
    board = "QsJh2h"  # flop

    # Some decent-sized ranges (Monker-ish compact format).
    # Keep them small-ish for speed, but varied enough to exercise the parser.
    range_ip = (
        "AA,KK,QQ,JJ,TT,99:0.75,88:0.5,"
        "AKs,AQs,AJs,ATs:0.5,A5s:0.5,"
        "KQs,KJs:0.5,QJs,QTs:0.5,JTs:0.75,T9s:0.75,98s:0.5"
    )
    range_oop = (
        "QQ:0.5,JJ:0.75,TT,99,88,77,66,55,44,33,22,"
        "AKo:0.25,AQs:0.75,AQo:0.5,AJs:0.5,AJo:0.5,ATs,ATo:0.5,"
        "KQ,KJs,KTs:0.5,QJ,QTs,JTs,JTo:0.5,T9s,98s"
    )

    # --- Run (IP actor) ---
    print("▶️  Solving (actor=ip)...")
    t0 = time.time()
    ip_result = load_villain_range_from_solver(
        cfg=cfg,
        pot_bb=pot_bb,
        effective_stack_bb=effective_stack_bb,
        board=board,
        range_ip=range_ip,
        range_oop=range_oop,
        actor="ip",               # we want IP's strategy distribution at node_key
        node_key="flop_root",     # or whatever your adapter expects
    )
    t1 = time.time()
    print(f"✅ actor=ip done in {t1 - t0:.2f}s, hands={len(ip_result)}")

    # Peek a few entries
    for i, (hand, p) in enumerate(list(ip_result.items())[:10]):
        print(f"   {hand}: {p:.3f}")
    if not ip_result:
        print("   ⚠️ Empty result for actor=ip — check adapter/solver logs.")

    # --- Run (OOP actor) ---
    print("\n▶️  Solving (actor=oop)...")
    t2 = time.time()
    oop_result = load_villain_range_from_solver(
        cfg=cfg,
        pot_bb=pot_bb,
        effective_stack_bb=effective_stack_bb,
        board=board,
        range_ip=range_ip,
        range_oop=range_oop,
        actor="oop",
        node_key="flop_root",
    )
    t3 = time.time()
    print(f"✅ actor=oop done in {t3 - t2:.2f}s, hands={len(oop_result)}")

    for i, (hand, p) in enumerate(list(oop_result.items())[:10]):
        print(f"   {hand}: {p:.3f}")
    if not oop_result:
        print("   ⚠️ Empty result for actor=oop — check adapter/solver logs.")

    # --- Quick sanity: distributions roughly normalize (sum ~1 across actions per hand) ---
    # Note: Depending on your adapter, ip_result/oop_result is either:
    #  - a map {hand_code -> probability} (already marginalized per action at node)
    #  - or a single-action density. If it’s multi-action, adapt this check.
    # Here we just print the top-5 hands by prob for each actor:
    def top5(d):
        return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:5]

    print("\nTop-5 ip hands:", top5(ip_result))
    print("Top-5 oop hands:", top5(oop_result))

if __name__ == "__main__":
    main()