import math
import pytest
from eval.metrics import *


def test_ndcg_hand_computed():
    # relevant at ranks 1 and 3, two relevant chunks exist, k=5
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k([1, 0, 1, 0, 0], 2, 5) == pytest.approx(dcg / idcg)


def test_ndcg_bounded_and_perfect():
    assert ndcg_at_k([1, 1, 1, 0, 0], 3, 5) == pytest.approx(1.0)       # old code gave 2.13 here
    assert ndcg_at_k([1, 1, 1, 1, 1], 9, 5) == pytest.approx(1.0)       # more relevant than k
    assert 0 <= ndcg_at_k([0, 1, 0, 0, 1], 2, 5) < 1


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k([0, 0, 0], 2, 3) == 0.0


def test_rr_hit_precision():
    assert reciprocal_rank([0, 0, 1, 1]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0, 0]) == 0.0
    assert hit_at_k([0, 0, 1], 2) == 0.0 and hit_at_k([0, 0, 1], 3) == 1.0
    assert context_precision([1, 0, 1, 0, 0]) == pytest.approx(0.4)
    assert context_precision_ap([1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert context_precision([]) == 0.0


def test_relevance_is_text_based_and_whitespace_insensitive():
    ev = ["IVF partitions the vector   space"]
    assert is_relevant("... ivf PARTITIONS the vector space into cells", ev)
    assert not is_relevant("unrelated text", ev)


def test_evidence_recall_counts_quotes_not_chunks():
    chunks = ["alpha beta gamma", "gamma delta", "zzz"]
    assert evidence_recall(chunks, ["gamma", "delta", "missing quote"], k=2) == pytest.approx(2 / 3)
    assert evidence_recall(chunks, ["zzz"], k=2) == 0.0
    with pytest.raises(ValueError):
        evidence_recall(chunks, [""], k=2)


def test_bootstrap_ci_contains_mean_and_widens_with_noise():
    m, lo, hi = bootstrap_ci([1, 1, 1, 1], seed=1)
    assert (m, lo, hi) == (1, 1, 1)
    m, lo, hi = bootstrap_ci([0, 1] * 10, seed=1)
    assert lo < m < hi and hi - lo > 0.2


def test_percentile_matches_numpy():
    np = pytest.importorskip("numpy")
    v = [3, 9, 1, 14, 7, 2, 25, 4]
    for q in (50, 95):
        assert percentile(v, q) == pytest.approx(float(np.percentile(v, q)))
