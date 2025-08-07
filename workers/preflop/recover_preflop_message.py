import json
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from preflop.generate_ranges import generate_single_range
# Paste your message body (JSON) from DLQ logs here:
RAW_BODY = """
{
  "ip_position": "BB",
  "oop_position": "SB",
  "stack_bb": 10,
  "villain_profile": "GTO",
  "exploit_setting": "GTO",
  "multiway_context": "3WAY",
  "population_type": "RECREATIONAL",
  "action_context": "OPEN"
}
"""


def main():
    try:
        config = json.loads(RAW_BODY)
        generate_single_range(config)
        print("✅ Recovery successful")
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()