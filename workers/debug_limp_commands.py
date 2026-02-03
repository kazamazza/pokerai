import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from workers.rangenet_postflop_solver_worker import _build_solver_command_text_for_job
from ml.etl.rangenet.postflop.build_rangenet_postflop_manifest import load_yaml
from workers.stake_params import load_stake_params  # only for raise_mult + allin_gate_spr

def _solver_profile_for(stake: Dict[str, Any], bet_sizing_id: str) -> Dict[str, Any]:
    profiles = stake.get("solver_profiles") or {}
    if not isinstance(profiles, dict):
        return {}
    default = profiles.get("default") or {}
    specific = profiles.get(bet_sizing_id) or {}
    # merge default -> specific
    out = {}
    if isinstance(default, dict):
        out.update(default)
    if isinstance(specific, dict):
        out.update(specific)
    return out

def main() -> None:
    ap = argparse.ArgumentParser("Print command texts for limp configs")
    ap.add_argument("--solver-yaml", required=True)
    ap.add_argument("--stake-key", default="Stakes.NL10")
    ap.add_argument("--street", type=int, default=1)
    ap.add_argument("--size-pct", type=int, default=33)
    ap.add_argument("--board", default="2h7dTh")
    ap.add_argument("--pot-bb", type=float, default=1.5)
    ap.add_argument("--eff-bb", type=float, default=100.0)
    ap.add_argument("--bet-sizing-id", default="limped_single.BB_IP")
    ap.add_argument("--range-ip", required=True)
    ap.add_argument("--range-oop", required=True)
    args = ap.parse_args()

    solver_yaml = load_yaml(args.solver_yaml)
    stake = solver_yaml.get(args.stake_key)
    if not isinstance(stake, dict):
        raise SystemExit(f"❌ stake key not found or invalid: {args.stake_key}")

    # minimal stake params used by command builder (raise mult + allin gate)
    stake_params = load_stake_params(solver_yaml, args.stake_key)

    prof = _solver_profile_for(stake, args.bet_sizing_id)
    accuracy = float(prof.get("accuracy", 0.02))
    max_iter = int(prof.get("max_iter", 3000))
    allin_threshold = float(prof.get("allin_threshold", 0.67))

    params: Dict[str, Any] = {
        "street": args.street,
        "pot_bb": args.pot_bb,
        "effective_stack_bb": args.eff_bb,
        "board": args.board,
        "range_ip": args.range_ip,
        "range_oop": args.range_oop,
        "bet_sizing_id": args.bet_sizing_id,
        "size_pct": args.size_pct,
        "accuracy": accuracy,
        "max_iter": max_iter,
        "allin_threshold": allin_threshold,
    }

    cmd = _build_solver_command_text_for_job(
        params=params,
        stake_params=stake_params,
        dump_path=Path("output_result.json"),
    )

    print("\n===== COMMANDS.TXT =====\n")
    print(cmd)

    # Assert expectations for limp BB_IP:
    # IP should have bet size; OOP should have no donk/bet sizes (only raises/allin config is ok)
    sn = {1: "flop", 2: "turn", 3: "river"}.get(args.street, "flop")

    bad_oop_donk = f"set_bet_sizes oop,{sn},donk,"
    bad_oop_bet  = f"set_bet_sizes oop,{sn},bet,"
    if bad_oop_donk in cmd or bad_oop_bet in cmd:
        raise SystemExit(f"❌ Limp cmd sets OOP bet/donk sizes on {sn} (should be IP-led).")

    must_ip_bet = f"set_bet_sizes ip,{sn},bet,"
    if must_ip_bet not in cmd:
        raise SystemExit(f"❌ Limp cmd missing IP bet menu on {sn}.")

    print("✅ Command looks structurally OK for limp BB_IP (IP has bet menu; OOP has no donk/bet menu).")

if __name__ == "__main__":
    main()