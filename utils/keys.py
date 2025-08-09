from typing import Dict


def build_preflop_filename(ip: str, oop: str, stack_bb: int) -> str:
    return f"{ip}_vs_{oop}_{stack_bb}bb.json.gz"

def build_preflop_s3_key(config: Dict) -> str:
    """
    Expects the producer/worker config payload fields:
      ip_position, oop_position, stack_bb,
      villain_profile, exploit_setting, multiway_context,
      population_type, action_context
    """
    ip   = config["ip_position"]
    oop  = config["oop_position"]
    sb   = int(config["stack_bb"])
    prof = config["villain_profile"]
    expl = config["exploit_setting"]
    mw   = config["multiway_context"]
    pop  = config["population_type"]
    act  = config["action_context"]

    filename = build_preflop_filename(ip, oop, sb)
    return (
        "preflop/ranges/"
        f"profile={prof}/exploit={expl}/multiway={mw}/"
        f"pop={pop}/action={act}/"
        f"{filename}"
    )