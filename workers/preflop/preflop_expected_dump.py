# tools/preflop_expected_dump.py
import itertools, sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# import the SAME symbols the producer uses
from ml.types import VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS, STACK_BUCKETS
from preflop.matchups import MATCHUPS
from utils.keys import build_preflop_s3_key


def build_all_cfgs():
    for profile, exploit, multiway, pop, action in itertools.product(
        VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, ACTION_CONTEXTS
    ):
        for ip, oop in MATCHUPS:
            for stack in STACK_BUCKETS:
                yield {
                    "ip_position": ip,
                    "oop_position": oop,
                    "stack_bb": int(stack),
                    "villain_profile": profile,
                    "exploit_setting": exploit,
                    "multiway_context": multiway,
                    "population_type": pop,
                    "action_context": action,
                }

def main():
    keys = [build_preflop_s3_key(cfg) for cfg in build_all_cfgs()]
    print(f"Expected total: {len(keys)}")
    Path("expected_keys.txt").write_text("\n".join(sorted(keys)) + "\n")
    print("Wrote expected_keys.txt")

if __name__ == "__main__":
    main()