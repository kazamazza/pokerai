import re
from pathlib import Path

EXPECTED = {
    "SRP": ["ip_open.txt", "oop_call.txt", "oop_raise_s1.txt", "oop_raise_s2.txt"],
    "LIMP_SINGLE": ["ip_open.txt", "oop_call.txt", "oop_raise_s1.txt", "oop_raise_s2.txt"],
    "LIMP_SINGLE": ["ip_open.txt", "ip_call.txt", "oop_call.txt", "oop_raise_s1.txt", "oop_raise_s2.txt"],
}

def parse_range_file(path: Path) -> bool:
    """Try to parse ranges into 169 floats. Return True if valid, else False."""
    txt = path.read_text(encoding="utf-8").strip()
    # Collect numbers from either [xx]..[/xx] or simple comma-separated
    nums = [float(x) for x in re.findall(r"[\d\.]+", txt)]
    return len(nums) == 169

def sanity_check(base="data/vendor/sph"):
    base = Path(base)
    issues = []
    for ctx, expected_files in EXPECTED.items():
        for stack_dir in (base/ctx).iterdir():
            if not stack_dir.is_dir():
                continue
            for pair_dir in stack_dir.iterdir():
                if not pair_dir.is_dir():
                    continue
                present = {f.name for f in pair_dir.iterdir() if f.is_file()}
                for need in expected_files:
                    if need not in present:
                        issues.append(f"❌ Missing {ctx}/{stack_dir.name}/{pair_dir.name}/{need}")
                    else:
                        fpath = pair_dir / need
                        try:
                            if not parse_range_file(fpath):
                                issues.append(f"⚠️ Invalid length in {fpath}")
                        except Exception as e:
                            issues.append(f"⚠️ Error parsing {fpath}: {e}")
    if not issues:
        print("✅ All expected ranges found and valid (169 values each).")
    else:
        print("\n".join(issues))

if __name__ == "__main__":
    sanity_check()