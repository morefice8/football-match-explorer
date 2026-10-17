"""Small, dependency-free statistics for share-of-events comparisons.

Football event samples are tiny, so every share we quote should carry its
uncertainty and every "this player is different" claim needs a test. Two tools:

* ``wilson_interval``: 95% confidence interval for a proportion k/n (stable for
  small n and for 0% / 100%).
* ``fisher_exact_two_sided``: exact two-sided p-value for a 2x2 table, the
  right test for "group shares vs the rest" on counts this small.
"""

from __future__ import annotations

from math import comb, sqrt

Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``successes / trials`` as fractions in [0, 1]."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided exact p-value for the table ``[[a, b], [c, d]]``."""
    a, b, c, d = int(a), int(b), int(c), int(d)
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c
    low, high = max(0, col1 - (n - row1)), min(row1, col1)
    total = comb(n, col1)

    def probability(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / total

    observed = probability(a)
    return float(min(1.0, sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed * (1.0 + 1e-9))))


def group_vs_rest(group_hits: int, group_n: int, rest_hits: int, rest_n: int) -> float:
    """Fisher p-value for "group share differs from the rest" on hit counts."""
    return fisher_exact_two_sided(group_hits, group_n - group_hits, rest_hits, rest_n - rest_hits)
