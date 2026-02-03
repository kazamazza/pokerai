from typing import Literal

ExtractorFailureMode = Literal["fail", "sentinel", "skip"]


class ExtractorInvariantError(RuntimeError):
    """Hard failure when extractor invariants are violated."""


def validate_extractor_output(
    ex,
    *,
    mode: ExtractorFailureMode = "fail",
) -> bool:
    def _fail(msg: str) -> bool:
        if mode == "fail":
            raise ExtractorInvariantError(msg)
        return False

    if ex is None:
        return _fail("extractor returned None")

    if not getattr(ex, "ok", False):
        return _fail(f"extractor not ok: {getattr(ex, 'reason', 'unknown')}")

    meta = getattr(ex, "meta", None)
    if not isinstance(meta, dict):
        return _fail("extractor meta missing or invalid")

    # root_only is allowed ONLY if extractor explicitly sets it
    root_only = bool(meta.get("root_only", False))

    # -----------------------
    # Root mix invariants
    # -----------------------
    root_mix = getattr(ex, "root_mix", None)
    if not isinstance(root_mix, dict):
        return _fail("root_mix is not a dict")
    if len(root_mix) == 0:
        return _fail("root_mix is empty")
    root_mass = sum(float(v) for v in root_mix.values())
    if root_mass <= 0:
        return _fail("root_mix has zero mass")

    # -----------------------
    # Facing mix invariants
    # -----------------------
    facing_mix = getattr(ex, "facing_mix", None)

    if root_only:
        # For root-only outputs, facing may be None or empty dict.
        if facing_mix is None or facing_mix == {}:
            pass
        elif not isinstance(facing_mix, dict):
            return _fail("root_only but facing_mix not dict/None")
        else:
            facing_mass = sum(float(v) for v in facing_mix.values()) if facing_mix else 0.0
            if facing_mass > 0:
                return _fail("root_only but facing_mix has mass")
    else:
        # Non-root-only MUST have a facing mix with mass.
        if facing_mix is None:
            return _fail("facing_mix missing (None) for non-root-only solve")
        if not isinstance(facing_mix, dict):
            return _fail("facing_mix is not a dict")
        if len(facing_mix) == 0:
            return _fail("facing_mix is empty")
        facing_mass = sum(float(v) for v in facing_mix.values())
        if facing_mass <= 0:
            return _fail("facing_mix has zero mass")

    # -----------------------
    # Metadata invariants
    # -----------------------
    if "facing_path" not in meta:
        return _fail("extractor meta missing facing_path")

    fp = meta.get("facing_path")
    if fp is not None and not isinstance(fp, list):
        return _fail("meta.facing_path must be list or None")

    return True