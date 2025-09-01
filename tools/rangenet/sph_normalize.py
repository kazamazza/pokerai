# tools/rangenet/sph_normalize.py
from __future__ import annotations
from pathlib import Path
import json, argparse

HAND_COUNT = 169

def parse_raw_sph(path: Path) -> dict:
    """
    TODO: adapt this to actual SPH export.
    Must return:
      { "ip_dist": [169 floats], "oop_dist": [169 floats],
        "meta": {...} }
    """
    raw = json.loads(path.read_text())
    # --- EXAMPLE ONLY: replace with your actual parsing ---
    ip = raw["ip"]["range169"]      # <- adapt
    oop = raw["oop"]["range169"]    # <- adapt
    assert len(ip) == HAND_COUNT and len(oop) == HAND_COUNT
    return {"ip_dist": ip, "oop_dist": oop, "meta": raw.get("meta", {})}

def write_norm(out_path: Path, stack: int, ip: str, oop: str, ctx: str, parsed: dict):
    out = {
        "stack_bb": int(stack),
        "ip_pos": ip, "oop_pos": oop, "ctx": ctx,
        "source": "sph",
        "ranges": {"ip": parsed["ip_dist"], "oop": parsed["oop_dist"]},
        "meta": parsed.get("meta", {})
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="folder with raw SPH exports")
    ap.add_argument("--out-root", required=True, help="normalized output root (data/vendor/sph_norm)")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    # Expect structure like raw_root/CTX/STACKbb/IPvOOP/result.json
    for ctx_dir in ["SRP", "LIMPED_SINGLE", "LIMPED_MULTI"]:
        for stack_dir in (raw_root / ctx_dir).glob("*bb"):
            try:
                stack = int(stack_dir.name.replace("bb",""))
            except:
                continue
            for pair_dir in stack_dir.iterdir():
                if not pair_dir.is_dir() or "v" not in pair_dir.name:
                    continue
                ip, oop = pair_dir.name.split("v", 1)
                # pick first json under pair_dir
                candidates = list(pair_dir.rglob("*.json"))
                if not candidates:
                    continue
                parsed = parse_raw_sph(candidates[0])
                out_path = out_root / ctx_dir / f"{stack}bb" / f"{ip}v{oop}.json"
                write_norm(out_path, stack, ip, oop, ctx_dir, parsed)

if __name__ == "__main__":
    main()