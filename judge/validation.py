"""Judge validation + A/B runner (Problem 2). Statistics live in judge/stats.py."""
from __future__ import annotations
from collections import Counter
from typing import Any
import numpy as np

from judge.models import TestCase
from judge.pipeline import JudgePipeline
from judge.bias import BiasMitigationHarness
from judge.stats import (cohens_kappa, kappa_bootstrap_ci, paired_bootstrap_diff,
                         sign_test_p, wilson_ci)


def compute_cohens_kappa(a: list[bool], b: list[bool]):
    """Kept for backwards compatibility; None means undefined (see stats.cohens_kappa)."""
    return cohens_kappa(a, b)


def _call_count(pipeline) -> int | None:
    """Best-effort: find a wrapped LLM client that counts its calls (LoggedLLM.calls)."""
    for v in (pipeline, *vars(pipeline).values()):
        c = getattr(v, "calls", None)
        if isinstance(c, int):
            return c
    return None


class JudgeValidator:
    def __init__(self, pipeline: JudgePipeline, bias_harness: BiasMitigationHarness):
        self.pipeline, self.harness = pipeline, bias_harness

    def validate_human_agreement(self, labeled_cases, pass_threshold: float = 3.5) -> dict[str, Any]:
        """Each case needs gold_pass. Add `label_source` ('human' | 'llm' | 'author') - it is echoed in the
        report so the README cannot claim human agreement for labels that were not human."""
        gold, judge, deltas, details = [], [], [], []
        for c in labeled_cases:
            tc = TestCase(id=c["id"], input=c["input"], model_output=c["model_output"],
                          expected_output=c.get("expected_output"), reference_context=c.get("reference_context"))
            v = self.pipeline.evaluate_pointwise(tc, pass_threshold=pass_threshold)
            g = bool(c["gold_pass"])
            gold.append(g)
            judge.append(v.passed)
            if "gold_score" in c:
                deltas.append(abs(v.overall_score - float(c["gold_score"])))
            details.append({"case_id": c["id"], "gold_pass": g, "judge_pass": v.passed, "agreed": g == v.passed,
                            "judge_score": v.overall_score, "gold_score": c.get("gold_score"),
                            "label_source": c.get("label_source", "unspecified"), "rationale": v.overall_rationale})
        n = len(labeled_cases)
        agree = sum(g == j for g, j in zip(gold, judge))
        k = cohens_kappa(gold, judge)
        klo, khi = kappa_bootstrap_ci(gold, judge)
        cm = Counter((g, j) for g, j in zip(gold, judge))
        return {
            "total_cases": n, "pass_threshold": pass_threshold,
            "label_sources": dict(Counter(d["label_source"] for d in details)),
            "agreement_rate": round(agree / n, 4) if n else 0.0,
            "agreement_ci95_wilson": [round(x, 3) for x in wilson_ci(agree, n)],
            "cohens_kappa": None if k is None else round(k, 4),
            "kappa_ci95_bootstrap": [None if x is None else round(x, 3) for x in (klo, khi)],
            "mean_absolute_score_error": round(float(np.mean(deltas)), 3) if deltas else None,
            "confusion_matrix": {"true_positive": cm[(True, True)], "true_negative": cm[(False, False)],
                                 "false_positive": cm[(False, True)], "false_negative": cm[(True, False)]},
            "case_details": details}

    def validate_test_retest_consistency(self, test_cases: list[TestCase], runs: int = 3) -> dict[str, Any]:
        """Judge each case `runs` times. Records how many LLM calls were really made: if that is
        fewer than cases*runs, a cache is serving repeats and the consistency numbers are void."""
        before = _call_count(self.pipeline)
        details, flips, deltas = [], 0, []
        for tc in test_cases:
            vs = [self.pipeline.evaluate_pointwise(tc) for _ in range(runs)]
            sc = [v.overall_score for v in vs]
            passed = [v.passed for v in vs]
            flip = len(set(passed)) > 1
            flips += flip
            deltas.append(max(sc) - min(sc))
            details.append({"case_id": tc.id, "scores": sc, "range": round(max(sc) - min(sc), 3),
                            "verdicts": passed, "consistent": not flip})
        after = _call_count(self.pipeline)
        n = len(test_cases)
        made = None if before is None or after is None else after - before
        # a pointwise judgement may use >1 call (parse retry), so only FEWER calls than judgements is suspicious
        cache = None if made is None else made < n * runs
        return {"total_cases": n, "runs_per_case": runs, "judge_calls_made": made, "judgements_requested": n * runs,
                "cache_suspected": cache, "flip_count": flips,
                "verdict_consistency_rate": round((n - flips) / n, 4) if n else None,
                "consistency_ci95_wilson": [round(x, 3) for x in wilson_ci(n - flips, n)],
                "mean_score_range": round(float(np.mean(deltas)), 4) if deltas else None,
                "max_score_range": round(float(np.max(deltas)), 4) if deltas else None,
                "case_details": details}

    def run_ab_comparison(self, comparison_cases, label_a="Config A", label_b="Config B", alpha: float = 0.05):
        """Declares a winner only if the sign test on non-tied cases is significant at `alpha`;
        otherwise reports INCONCLUSIVE. Cases: {id, input, output_a, output_b, expected_output?}."""
        rows, wa, wb, ties, sa, sb = [], 0, 0, 0, [], []
        for c in comparison_cases:
            r = self.harness.evaluate_symmetrical_pairwise(c["id"], c["input"], c["output_a"], c["output_b"],
                                                           c.get("expected_output"), c.get("reference_context"))
            wa += r.consensus_winner == "A"
            wb += r.consensus_winner == "B"
            ties += r.consensus_winner == "TIE"
            sa.append(r.mean_score_a)
            sb.append(r.mean_score_b)
            rows.append({"case_id": c["id"], "input": c["input"], "winner": r.consensus_winner,
                         "score_a": r.mean_score_a, "score_b": r.mean_score_b, "flipped": r.flipped,
                         "forward_rationale": r.forward_rationale, "reverse_rationale": r.reverse_rationale})
        n = len(comparison_cases)
        p = sign_test_p(wa, wb)
        md, lo, hi = paired_bootstrap_diff(sa, sb)
        if wa == wb or p >= alpha:
            winner = "INCONCLUSIVE"
        else:
            winner = label_b if wb > wa else label_a
        return {"label_a": label_a, "label_b": label_b, "total_cases": n, "wins_a": wa, "wins_b": wb, "ties": ties,
                "win_rate_a": round(wa / n, 4) if n else 0.0, "win_rate_b": round(wb / n, 4) if n else 0.0,
                "tie_rate": round(ties / n, 4) if n else 0.0, "sign_test_p": round(p, 4), "alpha": alpha,
                "declared_winner": winner, "margin_wins": abs(wb - wa),
                "mean_score_a": round(float(np.mean(sa)), 2) if sa else 0.0,
                "mean_score_b": round(float(np.mean(sb)), 2) if sb else 0.0,
                "mean_score_diff_b_minus_a": round(md, 3), "score_diff_ci95": [round(lo, 3), round(hi, 3)],
                "case_results": rows}
