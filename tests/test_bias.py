"""Tests the harness logic with a scripted stand-in judge. Real judge/models.py + pipeline.py are used if present."""
import sys, types
from dataclasses import dataclass
from types import SimpleNamespace as NS
import pytest

try:
    import judge.models, judge.pipeline  # noqa: F401
except ImportError:                         # sandbox without the full repo: minimal stand-ins
    m = types.ModuleType("judge.models")

    @dataclass
    class SymmetricalPairwiseResult:
        case_id: str; forward_winner: str; reverse_winner: str; consensus_winner: str; flipped: bool
        mean_score_a: float; mean_score_b: float; forward_rationale: str; reverse_rationale: str

    @dataclass
    class PositionBiasReport:
        total_pairs: int; agreement_count: int; flip_count: int; flip_rate: float
        a_first_win_rate: float; b_first_win_rate: float; tie_rate: float

    @dataclass
    class TestCase:
        __test__ = False
        id: str; input: str; model_output: str; expected_output: str | None = None; reference_context: str | None = None

    m.SymmetricalPairwiseResult, m.PositionBiasReport, m.TestCase, m.PairwiseVerdict = (
        SymmetricalPairwiseResult, PositionBiasReport, TestCase, object)
    p = types.ModuleType("judge.pipeline"); p.JudgePipeline = object
    sys.modules["judge.models"], sys.modules["judge.pipeline"] = m, p

from judge.bias import BiasMitigationHarness
from judge.validation import JudgeValidator


class FakePipe:
    """pairwise: `pick(order, a, b)` returns slot-relative winner 'first'|'second'|'tie' ; labels are un-swapped here."""
    def __init__(self, pick=None, point=None):
        self.pick, self.point, self.calls = pick, point, 0

    def evaluate_pairwise(self, case_id, user_input, output_a, output_b, order, expected_output=None, reference_context=None):
        self.calls += 1
        first, second = (("A", output_a), ("B", output_b)) if order == "A_then_B" else (("B", output_b), ("A", output_a))
        w = self.pick(first, second)                # returns 'first' | 'second' | 'tie'
        winner = {"first": first[0], "second": second[0], "tie": "TIE"}[w]
        sa = 5.0 if winner == "A" else 1.0 if winner == "B" else 3.0
        sb = 5.0 if winner == "B" else 1.0 if winner == "A" else 3.0
        return NS(winner=winner, score_a=sa, score_b=sb, rationale=f"{order}:{w}")

    def evaluate_pointwise(self, tc, pass_threshold=3.5):
        self.calls += 1
        s = self.point(tc)
        return NS(overall_score=s, passed=s >= pass_threshold, overall_rationale="r")


first_biased = lambda f, s: "first"
content_judge = lambda f, s: "first" if len(f[1]) >= len(s[1]) else "second"     # longer text wins (content-based)


def test_content_based_judge_has_no_flips():
    h = BiasMitigationHarness(FakePipe(content_judge))
    pairs = [{"id": "p1", "input": "q", "output_a": "short", "output_b": "much longer answer", "gold_winner": "B"}]
    rep, res = h.measure_position_bias(pairs)
    assert rep.flip_count == 0 and res[0].consensus_winner == "B"
    d = h.position_bias_details(res, pairs)
    assert d["accuracy_vs_gold"]["consensus"] == 1.0


def test_position_biased_judge_flips_and_consensus_is_tie():
    h = BiasMitigationHarness(FakePipe(first_biased))
    pairs = [{"id": "p1", "input": "q", "output_a": "x", "output_b": "y", "gold_winner": "B"}]
    rep, res = h.measure_position_bias(pairs)
    assert rep.flip_rate == 1.0 and res[0].consensus_winner == "TIE"
    d = h.position_bias_details(res, pairs)
    assert d["hard_flips"] == 1
    assert d["accuracy_vs_gold"] == {"n": 1, "forward_only": 0.0, "reverse_only": 1.0, "consensus": 0.0}


def test_soft_flip_counted():
    seq = iter(["first", "tie"])
    h = BiasMitigationHarness(FakePipe(lambda f, s: next(seq)))
    rep, res = h.measure_position_bias([{"id": "p", "input": "q", "output_a": "x", "output_b": "y"}])
    assert res[0].flipped and h.position_bias_details(res, [{"id": "p"}])["soft_flips"] == 1


def test_identical_pair_probe_exposes_slot_preference():
    cases = [{"id": f"c{i}", "input": "q", "output": "same text"} for i in range(6)]
    biased = BiasMitigationHarness(FakePipe(first_biased)).measure_identical_pair_bias(cases)
    assert biased["first_slot_win_rate_among_decisive"] == 1.0 and biased["sign_test_p"] < 0.05
    fair = BiasMitigationHarness(FakePipe(lambda f, s: "tie")).measure_identical_pair_bias(cases)
    assert fair["ties"] == 12 and fair["first_slot_win_rate_among_decisive"] is None


def test_verbosity_probe_detects_length_reward():
    cases = [{"id": f"v{i}", "input": "q", "terse_output": "t", "verbose_output": "t" + " fluff" * 20} for i in range(8)]
    rewards_len = FakePipe(content_judge, point=lambda tc: 3.0 + (1.0 if len(tc.model_output) > 20 else 0.0))
    out = BiasMitigationHarness(rewards_len).measure_verbosity_bias(cases)
    assert out["after_mitigation"]["verdict"] == "favours verbose"
    neutral = FakePipe(content_judge, point=lambda tc: 4.0)
    out = BiasMitigationHarness(neutral).measure_verbosity_bias(cases, baseline_pipeline=rewards_len)
    assert out["after_mitigation"]["verdict"] == "no detectable effect at this n"
    assert out["before_mitigation"]["verdict"] == "favours verbose"


def test_sycophancy_counts_ties_separately_and_hides_reference():
    seen = []
    class P(FakePipe):
        def evaluate_pairwise(self, **kw):
            seen.append(kw["expected_output"]); return super().evaluate_pairwise(**kw)
    cases = [{"id": "s1", "input": "q", "correct_output": "right", "confidently_wrong_output": "wrong", "expected_output": "GOLD"}]
    out = BiasMitigationHarness(P(lambda f, s: "first" if f[0] == "A" else "second")).measure_sycophancy_and_style_bias(cases, use_reference=False)
    assert out["rejected_count"] == 1 and set(seen) == {None}
    fooled = BiasMitigationHarness(FakePipe(lambda f, s: "first" if f[0] == "B" else "second")).measure_sycophancy_and_style_bias(cases)
    assert fooled["fooled_count"] == 1
    tie = BiasMitigationHarness(FakePipe(first_biased)).measure_sycophancy_and_style_bias(cases)
    assert tie["tied_count"] == 1 and tie["fooled_count"] == 0


def test_clustering_before_after():
    cases = [{"id": f"k{i}", "input": "q", "model_output": str(i)} for i in range(8)]
    anchored = FakePipe(point=lambda tc: 1.0 + int(tc.id[1:]) % 5)
    clustered = FakePipe(point=lambda tc: 5.0 if int(tc.id[1:]) % 4 else 4.0)
    out = BiasMitigationHarness(anchored).measure_score_clustering(cases, baseline_pipeline=clustered)
    assert out["after_mitigation"]["std"] > out["before_mitigation"]["std"]


def test_self_preference():
    cases = [{"id": f"e{i}", "input": "q", "output_judge_family": "own", "output_other_family": "other"} for i in range(8)]
    out = BiasMitigationHarness(FakePipe(lambda f, s: "first" if f[1] == "own" else "second")).measure_self_preference(cases)
    assert out["own_family_wins"] == 8 and out["sign_test_p"] < 0.01


def test_ab_inconclusive_when_not_significant_and_winner_when_significant():
    v = lambda pick: JudgeValidator(FakePipe(pick), BiasMitigationHarness(FakePipe(pick)))
    split = [{"id": f"a{i}", "input": "q", "output_a": "A" if i % 2 else "a", "output_b": "B"} for i in range(6)]
    pick_alt = lambda f, s: "first" if (f[1] == "A") else "second"
    r = v(pick_alt).run_ab_comparison(split, "v1", "v2")
    assert r["declared_winner"] in ("INCONCLUSIVE", "v1", "v2") and r["sign_test_p"] <= 1.0
    sweep = [{"id": f"b{i}", "input": "q", "output_a": "bad", "output_b": "good"} for i in range(8)]
    r = v(lambda f, s: "first" if f[1] == "good" else "second").run_ab_comparison(sweep, "v1", "v2")
    assert r["declared_winner"] == "v2" and r["wins_b"] == 8 and r["sign_test_p"] == pytest.approx(0.0078, abs=1e-4)
    close = [{"id": f"c{i}", "input": "q", "output_a": "x", "output_b": "y"} for i in range(3)]
    assert v(lambda f, s: "first" if f[1] == "y" else "second").run_ab_comparison(close)["declared_winner"] == "INCONCLUSIVE"


def test_retest_flags_cache_and_counts_real_calls():
    class Cached(FakePipe):
        def __init__(self): super().__init__(point=lambda tc: 4.0); self.memo = {}
        def evaluate_pointwise(self, tc, pass_threshold=3.5):
            if tc.id not in self.memo: self.memo[tc.id] = super().evaluate_pointwise(tc, pass_threshold)
            return self.memo[tc.id]
    cases = [sys.modules["judge.models"].TestCase(id=f"t{i}", input="q", model_output="o") for i in range(3)]
    r = JudgeValidator(Cached(), None).validate_test_retest_consistency(cases, runs=3)
    assert r["cache_suspected"] is True and r["judge_calls_made"] == 3 and r["judgements_requested"] == 9
    r = JudgeValidator(FakePipe(point=lambda tc: 4.0), None).validate_test_retest_consistency(cases, runs=3)
    assert r["cache_suspected"] is False


def test_human_agreement_reports_label_source_and_undefined_kappa():
    cases = [{"id": f"h{i}", "input": "q", "model_output": "o", "gold_pass": True, "label_source": "author"} for i in range(4)]
    r = JudgeValidator(FakePipe(point=lambda tc: 5.0), None).validate_human_agreement(cases)
    assert r["agreement_rate"] == 1.0 and r["cohens_kappa"] is None and r["label_sources"] == {"author": 4}
    assert r["agreement_ci95_wilson"][0] < 0.6                       # n=4 cannot certify anything
