"""Retrieval metrics with CHUNK-level relevance.

A chunk is relevant to a question if it contains at least one of the question's
`evidence` quotes (verbatim text copied from the source document, whitespace/case-insensitive).
Because relevance is defined by text, not chunk id, the same labels stay valid when you change
chunk size/overlap. Questions with no answer in the corpus (out_of_domain) are NOT scored here.
"""
from __future__ import annotations
import math
import random
import re
from typing import Sequence


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def is_relevant(chunk_text: str, evidence: Sequence[str]) -> bool:
    t = norm(chunk_text)
    return any(norm(e) in t for e in evidence if e.strip())


def relevance_flags(retrieved_texts: Sequence[str], evidence: Sequence[str]) -> list[int]:
    return [int(is_relevant(t, evidence)) for t in retrieved_texts]


def count_relevant_in_corpus(all_chunk_texts: Sequence[str], evidence: Sequence[str]) -> int:
    return sum(is_relevant(t, evidence) for t in all_chunk_texts)


def hit_at_k(flags: Sequence[int], k: int) -> float:
    return float(any(flags[:k]))


def reciprocal_rank(flags: Sequence[int], k: int | None = None) -> float:
    for i, f in enumerate(flags[:k], start=1):
        if f:
            return 1.0 / i
    return 0.0


def evidence_recall(retrieved_texts: Sequence[str], evidence: Sequence[str], k: int) -> float:
    """Fraction of evidence quotes found in at least one of the top-k chunks.
    (Counting chunks instead would punish you for overlap: one quote can sit in 2 chunks.)"""
    ev = [e for e in evidence if e.strip()]
    if not ev:
        raise ValueError("question has no evidence quotes")
    top = [norm(t) for t in retrieved_texts[:k]]
    return sum(any(norm(e) in t for t in top) for e in ev) / len(ev)


def ndcg_at_k(flags: Sequence[int], n_relevant_in_corpus: int, k: int) -> float:
    """Binary-gain nDCG. The ideal ranking puts min(n_relevant, k) relevant chunks first,
    so the score is bounded by 1.0 (the old code assumed exactly one relevant item)."""
    dcg = sum(f / math.log2(i + 1) for i, f in enumerate(flags[:k], start=1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(n_relevant_in_corpus, k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def context_precision(flags: Sequence[int], k: int | None = None) -> float:
    """Share of the chunks actually given to the LLM that are relevant."""
    f = list(flags[:k])
    return sum(f) / len(f) if f else 0.0


def context_precision_ap(flags: Sequence[int], k: int | None = None) -> float:
    """Rank-aware variant (average of precision@i at each relevant position)."""
    f, seen, acc = list(flags[:k]), 0, 0.0
    for i, x in enumerate(f, start=1):
        if x:
            seen += 1
            acc += seen / i
    return acc / seen if seen else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """(mean, lo, hi) percentile bootstrap. With ~20 questions the interval is wide: report it."""
    v = list(values)
    if not v:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(v, k=len(v))) / len(v) for _ in range(n_boot))
    return (sum(v) / len(v), means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1])


def percentile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
