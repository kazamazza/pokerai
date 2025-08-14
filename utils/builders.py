from pathlib import Path
from typing import Dict

from features.types import SolverRequest
from postflop.schema.cluster_strategy_schema import ClusterStrategy, StrategyNode, ActionBranch
from utils.solver import parse_board_string, combos_to_range_str
from workers.postflop.sqs_producer import VILLAIN_PROFILE, EXPLOIT_SETTING, MULTIWAY_CONTEXT, POPULATION_TYPE, \
    ACTION_CONTEXT

# ===== Pipeline toggles =====
UPLOAD_CLUSTER_TEMPLATES = False  # set True if you still want the templates saved
SOLVE_FROM_BOTH_SIDES = True      # IP and OOP calls (leave True for symmetry)

# ===== Fixed axes for clustering/solver stage =====
CLUSTER_VILLAIN_PROFILE   = "GTO"
CLUSTER_EXPLOIT_SETTING   = "GTO"
CLUSTER_MULTIWAY_CONTEXT  = "HU"
CLUSTER_POPULATION_TYPE   = "REGULAR"
CLUSTER_ACTION_CONTEXT    = "OPEN"

# ===== S3 prefixes =====
CLUSTER_TEMPLATES_PREFIX = "postflop/strategy_templates"
POSTFLOP_SOLVED_PREFIX   = "postflop/solved"

# Where cluster templates are written locally before upload
OUTPUT_DIR = Path("postflop/strategy_templates")

def build_solver_request_from_cluster(
    strategy: ClusterStrategy,
    stack_bb: int,
    pot_size_bb: float,
    hero_role: str = "IP",
    bet_sizes: dict | None = None,
    thread_count: int = 4,
) -> SolverRequest:
    """
    hero_role: "IP" or "OOP"
    pot_size_bb: pot in BB units (your solver call below just writes the float)
    bet_sizes: {(street, role, act): size} e.g. {("FLOP","IP","BET"): 0.33, ...}
    """
    board_cards = parse_board_string(strategy.board)

    if bet_sizes is None:
        # reasonable defaults for a single-line SRP tree; adjust as you like
        bet_sizes = {
            ("FLOP", "IP",  "BET"): 0.33,
            ("FLOP", "OOP", "BET"): 0.33,
            ("TURN", "IP",  "BET"): 0.66,
            ("TURN", "OOP", "BET"): 0.66,
            ("RIVER","IP",  "BET"): 1.00,
            ("RIVER","OOP", "BET"): 1.00,
        }

    req = SolverRequest(
        board=board_cards,
        hero_cards=[],  # fill per-combo when you iterate a hero combo; empty is ok if you only want opp range
        ip_range=combos_to_range_str(strategy.ip_range),
        oop_range=combos_to_range_str(strategy.oop_range),
        position=hero_role,               # "IP" or "OOP"
        stack_depth=float(stack_bb),      # BBs
        pot_size=float(pot_size_bb),
        bet_sizes=bet_sizes,
    )
    return req

def build_cluster_strategy_object(*, cluster_id: int, board: str, ip_range: List[str], oop_range: List[str]) -> ClusterStrategy:
    # Stubbed root nodes; solver will expand later
    ip_node = StrategyNode(
        combos=ip_range,
        actions=[ActionBranch(action="CHECK", size=None, frequency=1.0, next=None)]
    )
    oop_node = StrategyNode(
        combos=oop_range,
        actions=[ActionBranch(action="CHECK", size=None, frequency=1.0, next=None)]
    )
    return ClusterStrategy(
        cluster_id=cluster_id,
        board=board,
        ip_range=ip_range,
        oop_range=oop_range,
        ip_strategy=ip_node,
        oop_strategy=oop_node
    )

def build_cluster_template_s3_key(
    *, ip: str, oop: str, stack_bb: int, cluster_id: int
) -> str:
    file_out = f"{ip}_vs_{oop}_{stack_bb}bb_cluster_{cluster_id}.json.gz"
    return (
        f"{CLUSTER_TEMPLATES_PREFIX}/"
        f"profile={CLUSTER_VILLAIN_PROFILE}/exploit={CLUSTER_EXPLOIT_SETTING}/"
        f"multiway={CLUSTER_MULTIWAY_CONTEXT}/pop={CLUSTER_POPULATION_TYPE}/"
        f"action={CLUSTER_ACTION_CONTEXT}/{file_out}"
    )


def build_postflop_solved_s3_key(
    *, ip: str, oop: str, stack_bb: int, cluster_id: int
) -> str:
    """
    Final artifact your RangeNet dataset can read.
    """
    file_out = f"{ip}_vs_{oop}_{stack_bb}bb_cluster_{cluster_id}.json.gz"
    return (
        f"{POSTFLOP_SOLVED_PREFIX}/"
        f"profile={CLUSTER_VILLAIN_PROFILE}/exploit={CLUSTER_EXPLOIT_SETTING}/"
        f"multiway={CLUSTER_MULTIWAY_CONTEXT}/pop={CLUSTER_POPULATION_TYPE}/"
        f"action={CLUSTER_ACTION_CONTEXT}/{file_out}"
    )


def build_preflop_filename(ip: str, oop: str, stack_bb: int) -> str:
    # Canonical: IP_vs_OOP_<stack>bb.json.gz
    return f"{ip}_vs_{oop}_{int(stack_bb)}bb.json.gz"


def build_preflop_s3_key_components(
    *, ip: str, oop: str, stack_bb: int,
    villain_profile: str, exploit_setting: str,
    multiway_context: str, population_type: str, action_context: str,
) -> str:
    filename = build_preflop_filename(ip, oop, stack_bb)
    return (
        "preflop/ranges/"
        f"profile={villain_profile}/exploit={exploit_setting}/multiway={multiway_context}/"
        f"pop={population_type}/action={action_context}/{filename}"
    )

def build_preflop_s3_key(config: Dict) -> str:
    """
    Wrapper to keep backward compatibility with existing producer/worker payloads.
    Expects:
      ip_position, oop_position, stack_bb,
      villain_profile, exploit_setting, multiway_context,
      population_type, action_context
    """
    return build_preflop_s3_key_components(
        ip=config["ip_position"],
        oop=config["oop_position"],
        stack_bb=config["stack_bb"],
        villain_profile=config["villain_profile"],
        exploit_setting=config["exploit_setting"],
        multiway_context=config["multiway_context"],
        population_type=config["population_type"],
        action_context=config["action_context"],
    )

def build_postflop_key(cluster_id: int, ip: str, oop: str, stack_bb: int) -> str:
    filename = f"{ip}_vs_{oop}_{stack_bb}bb_cluster_{cluster_id}.json"
    return (
        "postflop/strategy_templates/"
        f"profile={VILLAIN_PROFILE}/exploit={EXPLOIT_SETTING}/multiway={MULTIWAY_CONTEXT}/"
        f"pop={POPULATION_TYPE}/action={ACTION_CONTEXT}/"
        f"{filename}"
    )