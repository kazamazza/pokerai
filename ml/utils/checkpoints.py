from pathlib import Path
import re
from typing import Optional

def pick_best_ckpt(ckpt_dir: str | Path, fallback_last: bool = True) -> Optional[str]:
    """
    Pick the best checkpoint in a directory based on the filename pattern:
      popnet-<epoch>-<val_loss>.ckpt
    Returns path as string (or None if nothing found and fallback_last=False).
    """
    p = Path(ckpt_dir)
    if not p.exists():
        return None
    best = None
    best_loss = float("inf")
    rx = re.compile(r"popnet-\d{2}-(?P<loss>[\d.]+)\.ckpt$")
    for f in p.glob("popnet-*.ckpt"):
        m = rx.match(f.name)
        if not m:
            continue
        loss = float(m.group("loss"))
        if loss < best_loss:
            best_loss, best = loss, f
    if best:
        return str(best)
    if fallback_last and (p / "last.ckpt").exists():
        return str(p / "last.ckpt")
    return None