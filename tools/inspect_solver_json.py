import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

#!/usr/bin/env python3
import os, sys, gzip, io, json, re
from typing import Tuple, Optional

CHUNK = 256 * 1024  # 256KB streaming chunks

TOKENS = {
    "actions": re.compile(r'"actions"\s*:\s*(\[|{)', re.IGNORECASE),
    "childrens": re.compile(r'"childrens"\s*:\s*(\[|{)', re.IGNORECASE),
    # various names solvers might use for mix vectors
    "strategy": re.compile(r'"strategy"\s*:\s*(\[|{)', re.IGNORECASE),
    "probabilities": re.compile(r'"probabilities"\s*:\s*(\[|{)', re.IGNORECASE),
    "frequencies": re.compile(r'"frequencies"\s*:\s*(\[|{)', re.IGNORECASE),
    "mix": re.compile(r'"mix"\s*:\s*(\[|{)', re.IGNORECASE),
}

SNIPPET_AROUND = 200  # print small window around first match

def _open(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")

def scan_tokens(path: str) -> Tuple[dict, Optional[str], Optional[int]]:
    """
    Stream-scan the file for token presence; also return a tiny snippet
    around the first strategy-like token we see.
    """
    counts = {k: 0 for k in TOKENS}
    first_strategy_snippet = None
    first_strategy_pos = None

    with _open(path) as f:
        carry = ""
        pos = 0
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            buf = carry + chunk
            # count all tokens in this buffer
            for name, rx in TOKENS.items():
                for m in rx.finditer(buf):
                    counts[name] += 1
                    if first_strategy_snippet is None and name in ("strategy", "probabilities", "frequencies", "mix"):
                        start = max(0, m.start() - SNIPPET_AROUND)
                        end = min(len(buf), m.end() + SNIPPET_AROUND)
                        snippet = buf[start:end]
                        # sanitize newlines to keep print short
                        first_strategy_snippet = snippet.replace("\n", " ")
                        first_strategy_pos = pos + m.start()
                        # don’t break; keep counting
            # keep a small tail to catch split tokens across boundaries
            carry_len = 64
            carry = buf[-carry_len:] if len(buf) > carry_len else buf
            pos += len(chunk)
    return counts, first_strategy_snippet, first_strategy_pos

def try_root_keys(path: str) -> dict:
    """
    If the file is small enough, try to load JSON and summarize top-level keys
    and the immediate root node shape (no deep traversal).
    """
    try:
        # If compressed file is too large, skip to avoid RAM blowups
        if path.endswith(".gz") and os.path.getsize(path) > 80 * 1024 * 1024:
            return {"skip": "compressed file >80MB; skipping json.load"}
        with _open(path) as f:
            data = json.load(f)
        obj = data.get("root", data) if isinstance(data, dict) else {}
        out = {
            "top_level_keys": sorted(list(data.keys())) if isinstance(data, dict) else "not_obj",
            "root_keys": sorted(list(obj.keys())) if isinstance(obj, dict) else "not_obj",
            "root_has_actions": isinstance(obj, dict) and "actions" in obj,
            "root_has_childrens": isinstance(obj, dict) and "childrens" in obj,
            "root_has_strategy": isinstance(obj, dict) and "strategy" in obj,
        }
        # if strategy is a list/dict at root, include lengths only
        if out["root_has_strategy"]:
            st = obj["strategy"]
            if isinstance(st, list):
                out["root_strategy_type"] = "list"
                out["root_strategy_len"] = len(st)
            elif isinstance(st, dict):
                out["root_strategy_type"] = "dict"
                out["root_strategy_len"] = len(st)
            else:
                out["root_strategy_type"] = type(st).__name__
        return out
    except MemoryError:
        return {"error": "MemoryError on json.load"}
    except Exception as e:
        return {"error": str(e)}

def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/inspect_solver_json.py <path-to-json-or-json.gz>")
        sys.exit(2)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"❌ no such file: {path}")
        sys.exit(1)

    size = os.path.getsize(path)
    print(f"\nInspecting: {path}")
    print(f"Compressed size: {size/1024/1024:.2f} MB")

    counts, snippet, spos = scan_tokens(path)
    print("\n--- TOKEN COUNTS (streamed) ---")
    for k in ("actions","childrens","strategy","probabilities","frequencies","mix"):
        print(f"{k:14s}: {counts[k]}")

    has_any_strategy = any(counts[k] > 0 for k in ("strategy","probabilities","frequencies","mix"))
    if has_any_strategy:
        print("\n✅ strategy-like field detected.")
        if snippet:
            print("…snippet around first occurrence…")
            print(snippet[:400] + ("…" if len(snippet) > 400 else ""))
    else:
        print("\n⚠️  No strategy-like field detected in a streamed scan.")
        print("    This looks like a tree-only dump (actions/childrens without mixes).")

    print("\n--- SHALLOW ROOT SUMMARY (best-effort) ---")
    summary = try_root_keys(path)
    print(summary)

if __name__ == "__main__":
    main()