import sys, glob, subprocess, os
from pathlib import Path

ROOT = Path("data/vendor/sph")
OUTC = Path("data/vendor_cache/sph")

def main():
    ctxs = ["SRP","LIMP_SINGLE","LIMP_MULTI"]
    stacks = ["25","60","100","150"]

    for ctx in ctxs:
        for s in stacks:
            base = ROOT / ctx / s
            if not base.exists():
                continue
            for pair_dir in sorted(base.glob("*_*")):
                ip, oop = pair_dir.name.split("_", 1)
                # Merge OOP defend if not present
                defend_csv = pair_dir / "oop_defend.csv"
                if not defend_csv.exists():
                    call = pair_dir / "oop_call.txt"
                    r1   = pair_dir / "oop_raise_s1.txt"
                    r2   = pair_dir / "oop_raise_s2.txt"
                    if call.exists() and r1.exists() and r2.exists():
                        # merge
                        cmd = [
                            sys.executable, "tools/rangenet/sph_merge_defend.py",
                            "--pair-dir", str(pair_dir),
                            "--out", str(defend_csv),
                        ]
                        print("MERGE:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
                    else:
                        print(f"⚠️ missing OOP files in {pair_dir}")
                        continue

                # Pack to canonical
                out_json = OUTC / ctx / s / f"{ip}_{oop}.json"
                out_json.parent.mkdir(parents=True, exist_ok=True)
                ip_open = pair_dir / "ip_open.txt"
                ip_call = pair_dir / "ip_call.txt"  # present only for LIMP_MULTI
                args = [
                    sys.executable, "tools/rangenet/sph_pack_pair.py",
                    "--stack", s, "--ip", ip, "--oop", oop, "--ctx", ctx,
                    "--ip-open", str(ip_open if ip_open.exists() else ip_call),
                    "--out", str(out_json),
                ]
                # OOP via defend CSV
                args += ["--oop-call", str(pair_dir/"oop_call.txt"),
                         "--oop-raise", str(pair_dir/"oop_raise_s1.txt"), str(pair_dir/"oop_raise_s2.txt")]
                print("PACK :", " ".join(args))
                subprocess.run(args, check=True)

    print("\nNow re-scan SPH and re-run coverage.")
    # subprocess.run([sys.executable, "tools/rangenet/scan_sph.py", "--out", "data/artifacts/sph_manifest.parquet"], check=True)

if __name__ == "__main__":
    main()