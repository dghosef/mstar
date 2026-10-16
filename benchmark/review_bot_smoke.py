"""Smoke-test file for the PR review bot (fork-only test PR; never merged)."""


def p90(latencies: list[float]) -> float:
    """The 90th-percentile latency (nearest rank)."""
    lat = sorted(latencies)
    return lat[int(len(lat) * 0.9) - 1]
