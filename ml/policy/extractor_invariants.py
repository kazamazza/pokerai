from typing import Literal


ExtractorFailureMode = Literal["fail", "sentinel", "skip"]


class ExtractorInvariantError(RuntimeError):
    """Hard failure when extractor invariants are violated."""


def validate_extractor_output(
    ex,
    *,
    mode: ExtractorFailureMode = "fail",
) -> bool:
    """
    Enforce non-negotiable invariants on TexasSolverExtractor output.

    Returns:
        True  -> extractor output is valid
        False -> invalid but caller chose skip/sentinel

    Raises:
        ExtractorInvariantError if mode == "fail"
    """

    def _fail(msg: str) -> bool:
        if mode == "fail":
            raise ExtractorInvariantError(msg)
        return False

    # -----------------------
    # Basic sanity
    # -----------------------
    if ex is None:
        return _fail("extractor returned None")

    if not getattr(ex, "ok", False):
        return _fail(f"extractor not ok: {getattr(ex, 'reason', 'unknown')}")

    # -----------------------
    # Root mix invariants
    # -----------------------
    root_mix = getattr(ex, "root_mix", None)
    if root_mix is not None:
        if not isinstance(root_mix, dict):
            return _fail("root_mix is not a dict")
        if len(root_mix) == 0:
            return _fail("root_mix is empty")
        s = sum(float(v) for v in root_mix.values())
        if s <= 0:
            return _fail("root_mix has zero mass")

    # -----------------------
    # Facing mix invariants
    # -----------------------
    facing_mix = getattr(ex, "facing_mix", None)
    if facing_mix is not None:
        if not isinstance(facing_mix, dict):
            return _fail("facing_mix is not a dict")
        if len(facing_mix) == 0:
            return _fail("facing_mix is empty")
        s = sum(float(v) for v in facing_mix.values())
        if s <= 0:
            return _fail("facing_mix has zero mass")

    # -----------------------
    # Metadata invariants
    # -----------------------
    meta = getattr(ex, "meta", None)
    if meta is None or not isinstance(meta, dict):
        return _fail("extractor meta missing or invalid")

    if "facing_path" not in meta:
        return _fail("extractor meta missing facing_path")

    fp = meta.get("facing_path")
    if fp is not None and not isinstance(fp, list):
        return _fail("meta.facing_path must be list or None")

    return True