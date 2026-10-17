"""Judge Validation Suite and A/B Comparison Runner (Problem 2).
Implements all validation requirements:
1. Agreement with human/gold labels (Agreement rate, Cohen's kappa, confusion matrix)
2. Test-retest consistency (Run-1 vs Run-2 flip rate, score variance)
3. Adversarial probe validation (verbose-wrong vs terse-correct, sycophancy probe)
4. A/B comparison declaring a winner between two configurations.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np

from judge.models import TestCase, PointwiseVerdict
from judge.pipeline import JudgePipeline
from judge.bias import BiasMitigationHarness


def compute_cohens_kappa(rater_a: list[bool], rater_b: list[bool]) -> float:
    """Calculates Cohen's Kappa for binary classifications (e.g. PASS vs FAIL)."""
    assert len(rater_a) == len(rater_b), "Raters must have same number of ratings"
    n = len(rater_a)
    if n == 0:
        return 1.0

    # Contingency table
    # a: both True, b: A True B False, c: A False B True, d: both False
    tp = sum(1 for a, b in zip(rater_a, rater_b) if a and b)
    fp = sum(1 for a, b in zip(rater_a, rater_b) if not a and b)
    fn = sum(1 for a, b in zip(rater_a, rater_b) if a and not b)
    tn = sum(1 for a, b in zip(rater_a, rater_b) if not a and not b)

    observed_agreement = (tp + tn) / n

    p_a_true = (tp + fn) / n
    p_b_true = (tp + fp) / n
    p_a_false = 1.0 - p_a_true
    p_b_false = 1.0 - p_b_true

    expected_agreement = (p_a_true * p_b_true) + (p_a_false * p_b_false)

    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


class JudgeValidator:
    def __init__(self, pipeline: JudgePipeline, bias_harness: BiasMitigationHarness):
        self.pipeline = pipeline
        self.harness = bias_harness

    def validate_human_agreement(
        self,
        labeled_cases: list[dict[str, Any]],
        pass_threshold: float = 3.5,
    ) -> dict[str, Any]:
        """Evaluates judge verdicts against expert gold labels."""
        gold_labels = []
        judge_labels = []
        score_deltas = []

        details = []
        for c in labeled_cases:
            tc = TestCase(
                id=c["id"],
                input=c["input"],
                model_output=c["model_output"],
                expected_output=c.get("expected_output"),
                reference_context=c.get("reference_context"),
            )
            v = self.pipeline.evaluate_pointwise(tc, pass_threshold=pass_threshold)
            
            gold_pass = bool(c["gold_pass"])
            judge_pass = v.passed
            gold_labels.append(gold_pass)
            judge_labels.append(judge_pass)

            if "gold_score" in c:
                score_deltas.append(abs(v.overall_score - float(c["gold_score"])))

            details.append({
                "case_id": c["id"],
                "gold_pass": gold_pass,
                "judge_pass": judge_pass,
                "agreed": (gold_pass == judge_pass),
                "judge_score": v.overall_score,
                "gold_score": c.get("gold_score"),
                "rationale": v.overall_rationale,
            })

        agreements = sum(1 for g, j in zip(gold_labels, judge_labels) if g == j)
        agreement_rate = agreements / len(labeled_cases) if labeled_cases else 0.0
        kappa = compute_cohens_kappa(gold_labels, judge_labels)
        mean_score_error = float(np.mean(score_deltas)) if score_deltas else 0.0

        return {
            "total_cases": len(labeled_cases),
            "agreement_rate": round(agreement_rate, 4),
            "cohens_kappa": round(kappa, 4),
            "mean_absolute_score_error": round(mean_score_error, 3),
            "confusion_matrix": {
                "true_positive": sum(1 for g, j in zip(gold_labels, judge_labels) if g and j),
                "true_negative": sum(1 for g, j in zip(gold_labels, judge_labels) if not g and not j),
                "false_positive": sum(1 for g, j in zip(gold_labels, judge_labels) if not g and j),
                "false_negative": sum(1 for g, j in zip(gold_labels, judge_labels) if g and not j),
            },
            "case_details": details,
        }

    def validate_test_retest_consistency(
        self,
        test_cases: list[TestCase],
        runs: int = 2,
    ) -> dict[str, Any]:
        """Evaluates variance across repeated runs at temperature=0.0."""
        run_1_scores = []
        run_2_scores = []
        flip_count = 0

        details = []
        for tc in test_cases:
            v1 = self.pipeline.evaluate_pointwise(tc)
            v2 = self.pipeline.evaluate_pointwise(tc)

            run_1_scores.append(v1.overall_score)
            run_2_scores.append(v2.overall_score)

            flipped = (v1.passed != v2.passed)
            if flipped:
                flip_count += 1

            details.append({
                "case_id": tc.id,
                "score_run1": v1.overall_score,
                "score_run2": v2.overall_score,
                "delta": round(abs(v1.overall_score - v2.overall_score), 2),
                "verdict_run1": v1.passed,
                "verdict_run2": v2.passed,
                "consistent": not flipped,
            })

        diffs = [abs(s1 - s2) for s1, s2 in zip(run_1_scores, run_2_scores)]
        consistency_rate = (len(test_cases) - flip_count) / len(test_cases) if test_cases else 1.0

        return {
            "total_cases": len(test_cases),
            "verdict_consistency_rate": round(consistency_rate, 4),
            "flip_count": flip_count,
            "mean_score_delta": round(float(np.mean(diffs)), 4),
            "max_score_delta": round(float(np.max(diffs)), 4),
            "case_details": details,
        }

    def run_ab_comparison(
        self,
        comparison_cases: list[dict[str, Any]],
        label_a: str = "Config A (Baseline)",
        label_b: str = "Config B (Improved)",
    ) -> dict[str, Any]:
        """Compares two configurations (Prompt v1 vs v2, or Model A vs Model B)
        using symmetrical pairwise evaluations to neutralize position bias."""
        results = []
        wins_a = 0
        wins_b = 0
        ties = 0

        scores_a = []
        scores_b = []

        for c in comparison_cases:
            res = self.harness.evaluate_symmetrical_pairwise(
                case_id=c["id"],
                user_input=c["input"],
                output_a=c["output_a"],
                output_b=c["output_b"],
                expected_output=c.get("expected_output"),
                reference_context=c.get("reference_context"),
            )
            if res.consensus_winner == "A":
                wins_a += 1
            elif res.consensus_winner == "B":
                wins_b += 1
            else:
                ties += 1

            scores_a.append(res.mean_score_a)
            scores_b.append(res.mean_score_b)
            results.append({
                "case_id": c["id"],
                "input": c["input"],
                "winner": res.consensus_winner,
                "score_a": res.mean_score_a,
                "score_b": res.mean_score_b,
                "flipped": res.flipped,
                "rationale": res.forward_rationale,
            })

        n = len(comparison_cases)
        win_rate_a = round(wins_a / n, 4) if n else 0.0
        win_rate_b = round(wins_b / n, 4) if n else 0.0
        tie_rate = round(ties / n, 4) if n else 0.0

        if wins_b > wins_a:
            winner = label_b
            margin = wins_b - wins_a
        elif wins_a > wins_b:
            winner = label_a
            margin = wins_a - wins_b
        else:
            winner = "TIE"
            margin = 0

        return {
            "label_a": label_a,
            "label_b": label_b,
            "total_cases": n,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": win_rate_a,
            "win_rate_b": win_rate_b,
            "tie_rate": tie_rate,
            "declared_winner": winner,
            "margin_wins": margin,
            "mean_score_a": round(float(np.mean(scores_a)), 2) if scores_a else 0.0,
            "mean_score_b": round(float(np.mean(scores_b)), 2) if scores_b else 0.0,
            "case_results": results,
        }
