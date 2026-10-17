"""Small, dependency-free statistics for judge validation."""
from __future__ import annotations
import math
import random
from collections import Counter
from typing import Sequence


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test_p(wins_a: int, wins_b: int) -> float:
    """Exact two-sided sign test on non-tied pairs (H0: P(A beats B) = 0.5)."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]):
    """Returns None when kappa is undefined (empty input, or chance agreement == 1,
    e.g. every label identical). The old code returned 1.0 there, which hides a degenerate test set."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1.0 else (po - pe) / (1 - pe)


def kappa_bootstrap_ci(a: Sequence[bool], b: Sequence[bool], n_boot=2000, seed=0):
    rng = random.Random(seed)
    idx = range(len(a))
    ks = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        k = cohens_kappa([a[i] for i in s], [b[i] for i in s])
        if k is not None:
            ks.append(k)
    if len(ks) < 20:
        return (None, None)
    ks.sort()
    return (ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks)) - 1])


def paired_bootstrap_diff(x: Sequence[float], y: Sequence[float], n_boot=2000, seed=0):
    """Mean of (y - x) with a 95% percentile interval, resampling pairs."""
    d = [b - a for a, b in zip(x, y)]
    if not d:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    ms = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_boot))
    return (sum(d) / len(d), ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot) - 1])


def score_spread(scores: Sequence[float], lo: float = 1.0, hi: float = 5.0) -> dict:
    """Quantifies score clustering: std dev, share at the scale ends, distinct values,
    and normalised entropy of the rounded-score histogram (1.0 = evenly used scale)."""
    s = list(scores)
    if not s:
        return {"n": 0}
    m = sum(s) / len(s)
    sd = math.sqrt(sum((x - m) ** 2 for x in s) / len(s))
    buckets = Counter(round(x) for x in s)
    nb = int(hi - lo + 1)
    ent = -sum((c / len(s)) * math.log(c / len(s)) for c in buckets.values())
    return {"n": len(s), "mean": round(m, 3), "std": round(sd, 3),
            "frac_at_extremes": round(sum(x <= lo + 1e-9 or x >= hi - 1e-9 for x in s) / len(s), 3),
            "distinct_values": len(set(s)),
            "entropy_norm": round(ent / math.log(nb), 3) if nb > 1 else 0.0,
            "histogram": {int(k): v for k, v in sorted(buckets.items())}}
