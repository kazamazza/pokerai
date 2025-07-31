import os
import json
import csv


VALID_COMBOS = {
    f"{r1}{r2}" if r1 == r2 else f"{r1}{r2}s" if i < j else f"{r1}{r2}o"
    for i, r1 in enumerate("AKQJT98765432")
    for j, r2 in enumerate("AKQJT98765432")
}

def validate_preflop_file(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)

    summary = {
        'file': os.path.basename(path),
        'valid': True,
        'unknown_actions': [],
        'invalid_hands': [],
        'duplicate_hands': [],
        'empty_buckets': [],
        'total_hands': 0
    }

    hand_to_bucket = {}
    seen_hands = set()

    for action, hands in data.items():
        if action not in VALID_ACTIONS:
            summary['unknown_actions'].append(action)
            summary['valid'] = False
            continue

        if not hands:
            summary['empty_buckets'].append(action)

        for hand in hands:
            summary['total_hands'] += 1
            if hand not in VALID_COMBOS:
                summary['invalid_hands'].append(hand)
                summary['valid'] = False
            elif hand in seen_hands:
                summary['duplicate_hands'].append(hand)
                summary['valid'] = False
            else:
                seen_hands.add(hand)
                hand_to_bucket[hand] = action

    return summary

def validate_all_preflop_ranges(folder='preflop/ranges', csv_output=None):
    report = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                path = os.path.join(root, file)
                result = validate_preflop_file(path)
                report.append(result)

    # Print Summary
    print("\n🔍 Validation Summary:\n")
    for r in report:
        print(f"✅ {r['file']}" if r['valid'] else f"❌ {r['file']}")
        if r['invalid_hands']:
            print("   - Invalid hands:", r['invalid_hands'])
        if r['duplicate_hands']:
            print("   - Duplicate hands:", r['duplicate_hands'])
        if r['unknown_actions']:
            print("   - Unknown actions:", r['unknown_actions'])

    # Optional CSV output
    if csv_output:
        with open(csv_output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=report[0].keys())
            writer.writeheader()
            writer.writerows(report)
        print(f"\n📄 CSV report written to: {csv_output}")

if __name__ == "__main__":
    validate_all_preflop_ranges(csv_output="reports/preflop_validation_report.csv")