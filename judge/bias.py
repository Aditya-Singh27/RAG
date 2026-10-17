"""Bias Detection, Mitigation, and Measurement Pipeline (Problem 2).
Covers:
1. Position Bias: Symmetrical dual-order evaluation (A-B and B-A), flip rate measurement.
2. Verbosity Bias: Padded answer probes and length penalty enforcement.
3. Self-Enhancement Bias: Cross-family verification (Gemini vs Llama 3.2).
4. Sycophancy / Style Bias: Confidently wrong / polite adversarial probes.
5. Score Clustering: Calibration anchors and variance/spread analysis.
"""
from __future__ import annotations
import time
from typing import Any, Literal
import numpy as np

from judge.models import (
    PairwiseVerdict,
    SymmetricalPairwiseResult,
    PositionBiasReport,
    TestCase,
)
from judge.pipeline import JudgePipeline


class BiasMitigationHarness:
    def __init__(self, pipeline: JudgePipeline):
        self.pipeline = pipeline

    def evaluate_symmetrical_pairwise(
        self,
        case_id: str,
        user_input: str,
        output_a: str,
        output_b: str,
        expected_output: str | None = None,
        reference_context: str | None = None,
    ) -> SymmetricalPairwiseResult:
        """Runs the pair in both orders to detect and neutralize position bias.
        Forward: Option A is shown first, Option B second.
        Reverse: Option B is shown first, Option A second.
        """
        # Forward pass: A shown first
        forward_v = self.pipeline.evaluate_pairwise(
            case_id=case_id,
            user_input=user_input,
            output_a=output_a,
            output_b=output_b,
            order="A_then_B",
            expected_output=expected_output,
            reference_context=reference_context,
        )

        # Reverse pass: B shown first
        reverse_v = self.pipeline.evaluate_pairwise(
            case_id=case_id,
            user_input=user_input,
            output_a=output_a,
            output_b=output_b,
            order="B_then_A",
            expected_output=expected_output,
            reference_context=reference_context,
        )

        # In reverse pass: Option B was shown first (labeled Option B).
        # If the judge answered winner="B", it picked the true B.
        # If the judge answered winner="A", it picked the true A.
        forward_winner = forward_v.winner
        reverse_winner = reverse_v.winner

        # Flip occurs if forward and reverse declared different winners (excluding ties)
        flipped = False
        if forward_winner != "TIE" and reverse_winner != "TIE":
            flipped = (forward_winner != reverse_winner)

        # Determine consensus winner:
        # If both agree -> that candidate wins.
        # If they disagree or one tied -> use higher average score, or TIE if equal.
        mean_score_a = round((forward_v.score_a + reverse_v.score_a) / 2.0, 2)
        mean_score_b = round((forward_v.score_b + reverse_v.score_b) / 2.0, 2)

        if forward_winner == reverse_winner:
            consensus_winner = forward_winner
        else:
            if mean_score_a > mean_score_b + 0.2:
                consensus_winner = "A"
            elif mean_score_b > mean_score_a + 0.2:
                consensus_winner = "B"
            else:
                consensus_winner = "TIE"

        return SymmetricalPairwiseResult(
            case_id=case_id,
            forward_winner=forward_winner,
            reverse_winner=reverse_winner,
            consensus_winner=consensus_winner,
            flipped=flipped,
            mean_score_a=mean_score_a,
            mean_score_b=mean_score_b,
            forward_rationale=forward_v.rationale,
            reverse_rationale=reverse_v.rationale,
        )

    def measure_position_bias(
        self,
        test_pairs: list[dict[str, Any]],
    ) -> tuple[PositionBiasReport, list[SymmetricalPairwiseResult]]:
        """Measures position bias across a suite of candidate pairs."""
        results: list[SymmetricalPairwiseResult] = []
        first_presented_wins = 0
        total_evaluations = 0

        for p in test_pairs:
            res = self.evaluate_symmetrical_pairwise(
                case_id=p["id"],
                user_input=p["input"],
                output_a=p["output_a"],
                output_b=p["output_b"],
                expected_output=p.get("expected_output"),
                reference_context=p.get("reference_context"),
            )
            results.append(res)

            # In forward pass: A was first presented
            if res.forward_winner == "A":
                first_presented_wins += 1
            # In reverse pass: B was first presented
            if res.reverse_winner == "B":
                first_presented_wins += 1
            total_evaluations += 2

        flips = sum(1 for r in results if r.flipped)
        agreements = len(results) - flips
        flip_rate = round(flips / len(results), 4) if results else 0.0
        first_win_rate = round(first_presented_wins / total_evaluations, 4) if total_evaluations else 0.0

        report = PositionBiasReport(
            total_pairs=len(results),
            agreement_count=agreements,
            flip_count=flips,
            flip_rate=flip_rate,
            a_first_win_rate=first_win_rate,
            b_first_win_rate=round(1.0 - first_win_rate, 4),
            tie_rate=round(sum(1 for r in results if r.consensus_winner == "TIE") / len(results), 4),
        )
        return report, results

    def measure_verbosity_bias(
        self,
        terse_vs_verbose_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Probes the judge with pairs where a concise answer is pitted against
        a padded verbose answer containing the exact same core facts plus fluff."""
        wins_terse = 0
        wins_verbose = 0
        ties = 0
        score_deltas = []

        for c in terse_vs_verbose_cases:
            res = self.evaluate_symmetrical_pairwise(
                case_id=c["id"],
                user_input=c["input"],
                output_a=c["terse_output"],    # Option A: concise
                output_b=c["verbose_output"],  # Option B: padded
                expected_output=c.get("expected_output"),
            )
            if res.consensus_winner == "A":
                wins_terse += 1
            elif res.consensus_winner == "B":
                wins_verbose += 1
            else:
                ties += 1
            score_deltas.append(res.mean_score_b - res.mean_score_a)

        return {
            "total_cases": len(terse_vs_verbose_cases),
            "terse_wins": wins_terse,
            "verbose_wins": wins_verbose,
            "ties": ties,
            "verbose_win_rate": round(wins_verbose / len(terse_vs_verbose_cases), 4) if terse_vs_verbose_cases else 0.0,
            "mean_score_delta_verbose_minus_terse": round(float(np.mean(score_deltas)), 3) if score_deltas else 0.0,
            "mitigation_effective": (wins_terse >= wins_verbose),
        }

    def measure_sycophancy_and_style_bias(
        self,
        adversarial_probe_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Adversarial probes: Injects confidently wrong, beautifully phrased,
        polite/authoritative answers vs modest/blunt correct answers."""
        fooled_count = 0
        rejected_count = 0

        results = []
        for c in adversarial_probe_cases:
            # Option A = Honest / correct answer (may be blunt or uncertain)
            # Option B = Confidently wrong / sycophantic / fake authoritative
            res = self.evaluate_symmetrical_pairwise(
                case_id=c["id"],
                user_input=c["input"],
                output_a=c["correct_output"],
                output_b=c["confidently_wrong_output"],
                expected_output=c.get("expected_output"),
            )
            # If the judge chose B (the fake confident/wrong answer), it was fooled!
            was_fooled = (res.consensus_winner == "B")
            if was_fooled:
                fooled_count += 1
            else:
                rejected_count += 1

            results.append({
                "case_id": c["id"],
                "fooled": was_fooled,
                "winner": res.consensus_winner,
                "correct_score": res.mean_score_a,
                "wrong_score": res.mean_score_b,
                "rationale": res.forward_rationale,
            })

        return {
            "total_adversarial_cases": len(adversarial_probe_cases),
            "fooled_count": fooled_count,
            "rejected_count": rejected_count,
            "fool_rate": round(fooled_count / len(adversarial_probe_cases), 4) if adversarial_probe_cases else 0.0,
            "robustness_rate": round(rejected_count / len(adversarial_probe_cases), 4) if adversarial_probe_cases else 0.0,
            "probe_details": results,
        }
