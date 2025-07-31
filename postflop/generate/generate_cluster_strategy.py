import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from postflop.schema.cluster_strategy_schema import ClusterStrategy, StrategyNode, ActionBranch

with open("data/flop/flop_cluster_map.json", "r") as f:
    FLOP_CLUSTER_MAP = json.load(f)

# === Example Preflop Combo Fetcher ===
def load_range(position: str, stack_bb: int, context: str) -> list[str]:
    # TEMP: replace with actual loader
    return ["AhKs", "AdQs", "KcQc", "JhTs"]

def generate_cluster_strategy(
    cluster_id: int,
    stack_bb: int,
    ip_position: str,
    oop_position: str,
    villain_profile: str,
    exploit_setting: str,
    multiway_context: str,
    population_type: str,
    action_context: str
) -> ClusterStrategy:
    # 1. Load example board from cluster
    example_board = next(
        board for board, cid in FLOP_CLUSTER_MAP.items() if cid == cluster_id
    )

    # 2. Define path for preflop strategy
    base_path = Path("preflop/ranges") / \
                f"profile={villain_profile.upper()}" / \
                f"exploit={exploit_setting.upper()}" / \
                f"multiway={multiway_context.upper()}" / \
                f"pop={population_type.upper()}" / \
                f"action={action_context.upper()}"

    filename = f"{ip_position}_vs_{oop_position}_{stack_bb}bb.json"
    file_path = base_path / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Preflop range not found: {file_path}")

    with open(file_path, "r") as f:
        preflop_data = json.load(f)

    # 3. Extract combos by role
    ip_range = preflop_data.get("3bet", []) + preflop_data.get("flat", []) + preflop_data.get("open", [])
    oop_range = preflop_data.get("call", []) + preflop_data.get("4bet", []) + preflop_data.get("check", []) + preflop_data.get("defend", [])

    if not ip_range or not oop_range:
        raise ValueError("Missing combos in preflop chart")

    # 4. Dummy strategies for now
    ip_strategy = StrategyNode(
        combos=ip_range,
        actions=[ActionBranch(action="BET", size=0.5, frequency=1.0)]
    )

    oop_strategy = StrategyNode(
        combos=oop_range,
        actions=[ActionBranch(action="CHECK", size=None, frequency=1.0)]
    )

    # 5. Return strategy
    return ClusterStrategy(
        cluster_id=cluster_id,
        board=example_board,
        ip_range=ip_range,
        oop_range=oop_range,
        ip_strategy=ip_strategy,
        oop_strategy=oop_strategy
    )


if __name__ == "__main__":
    cluster_id = 23  # Example
    strategy = generate_cluster_strategy(
        cluster_id=cluster_id,
        stack_bb=40,
        ip_position="BTN",
        oop_position="BB",
        villain_profile="GTO",
        exploit_setting="GTO",
        multiway_context="HU",
        population_type="REGULAR",
        action_context="VS_OPEN"
    )

    out_path = f"data/strategy/cluster_{cluster_id}.json"
    with open(out_path, "w") as f:
        json.dump(strategy.dict(), f, indent=2)

    print(f"✅ Saved cluster strategy to: {out_path}")