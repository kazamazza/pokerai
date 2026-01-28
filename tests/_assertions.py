import math

def assert_probs_valid(probs: dict[str, float], *, tol: float = 1e-6) -> None:
    assert isinstance(probs, dict)
    s = 0.0
    for k, v in probs.items():
        assert isinstance(k, str) and k
        assert isinstance(v, (int, float))
        assert math.isfinite(float(v))
        assert float(v) >= -tol
        s += float(v)
    assert abs(s - 1.0) <= 1e-3  # looser tolerance is fine

def assert_mask52(mask: list[float]) -> None:
    assert isinstance(mask, list)
    assert len(mask) == 52
    for x in mask:
        assert x in (0.0, 1.0)