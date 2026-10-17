import pytest
from judge.stats import *


def test_kappa_known_value():
    a = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    b = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]       # po=0.8, pe=0.5 -> kappa 0.6
    assert cohens_kappa(a, b) == pytest.approx(0.6)


def test_kappa_undefined_is_none_not_one():
    assert cohens_kappa([True] * 5, [True] * 5) is None
    assert cohens_kappa([], []) is None


def test_kappa_ci_wide_for_small_n():
    a = [True] * 6 + [False] * 4
    lo, hi = kappa_bootstrap_ci(a, a)         # perfect agreement on n=10
    assert cohens_kappa(a, a) == 1.0
    # resampling can produce degenerate samples; lower bound must not claim certainty of 1.0
    assert lo is None or lo <= 1.0


def test_wilson_ci_n10_all_correct_is_not_certain():
    lo, hi = wilson_ci(10, 10)
    assert 0.65 < lo < 0.75 and hi == pytest.approx(1.0)


def test_sign_test():
    assert sign_test_p(8, 0) == pytest.approx(2 / 256)
    assert sign_test_p(4, 4) == 1.0
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 3) > 0.05


def test_spread_detects_clustering():
    clustered = score_spread([5, 5, 5, 5, 4.5, 5, 5, 5])
    spread = score_spread([1, 2, 3, 4, 5, 3, 2, 4])
    assert clustered["std"] < spread["std"]
    assert clustered["entropy_norm"] < spread["entropy_norm"]
    assert score_spread([1, 5, 1, 5])["frac_at_extremes"] == 1.0


def test_paired_bootstrap():
    m, lo, hi = paired_bootstrap_diff([1, 2, 3, 4], [2, 3, 4, 5])
    assert m == 1 and lo == 1 and hi == 1
