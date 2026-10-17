"""Full Evaluation Pipeline Runner for Problem 2 (LLM-as-a-Judge).
Executes:
1. Pointwise suite evaluation and Human Gold Label Agreement (Cohen's Kappa)
2. Symmetrical Pairwise Position Bias Check & Flip Rate Measurement
3. Verbosity Bias & Length Penalty Evaluation
4. Sycophancy / Adversarial Probe Evaluation (Fool Rate)
5. Test-Retest Consistency Check (Repeat Runs at temp=0.0)
6. A/B Comparison declaring a winner between configurations
Saves complete auditable report to results/judge_evaluation_report.json.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import JudgeConfig
from common.llm import make_llm
from judge.models import TestCase
from judge.pipeline import JudgePipeline
from judge.bias import BiasMitigationHarness
from judge.validation import JudgeValidator

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUITE_DIR = Path("data/judge_suites")


def run_full_judge_pipeline():
    print("=" * 65, flush=True)
    print("Starting Comprehensive LLM-as-a-Judge Evaluation Pipeline (Problem 2)", flush=True)
    print("=" * 65, flush=True)

    jcfg = JudgeConfig.from_env()
    print(f"Judge Model Family:     {jcfg.judge_provider} ({jcfg.judge_model})", flush=True)
    print(f"Generator Model Family: {jcfg.generator_provider} ({jcfg.generator_model})", flush=True)
    print("Self-Enhancement Bias Check: DECOUPLED across different model families!\n", flush=True)


    judge_client = make_llm(jcfg.judge_provider, jcfg.judge_model)
    pipeline = JudgePipeline(judge_client=judge_client, log_path=jcfg.log_path)
    bias_harness = BiasMitigationHarness(pipeline)
    validator = JudgeValidator(pipeline, bias_harness)

    # 1. Human / Gold Label Agreement (Cohen's Kappa)
    print("--- 1. Evaluating Agreement with Human/Gold Labels ---")
    hl_path = SUITE_DIR / "human_labeled_cases.json"
    with open(hl_path, "r", encoding="utf-8") as f:
        human_labeled_cases = json.load(f)

    agreement_res = validator.validate_human_agreement(human_labeled_cases, pass_threshold=3.5)
    print(f"  Total Cases:               {agreement_res['total_cases']}", flush=True)
    print(f"  Exact Agreement Rate:      {agreement_res['agreement_rate'] * 100:.1f}%", flush=True)
    print(f"  Cohen's Kappa (kappa):     {agreement_res['cohens_kappa']:.4f}", flush=True)
    print(f"  Mean Score Error:          {agreement_res['mean_absolute_score_error']:.3f}", flush=True)
    print(f"  Confusion Matrix:          {agreement_res['confusion_matrix']}", flush=True)


    # 2. Position Bias Check (Symmetrical Pairwise A/B Swap)
    print("\n--- 2. Evaluating Position Bias (Dual-Order Symmetrical Pairwise) ---", flush=True)
    ab_path = SUITE_DIR / "ab_comparison_cases.json"
    with open(ab_path, "r", encoding="utf-8") as f:
        ab_cases = json.load(f)

    pos_report, pos_details = bias_harness.measure_position_bias(ab_cases)
    print(f"  Total Pairs Evaluated:     {pos_report.total_pairs}", flush=True)
    print(f"  Dual-Order Agreement:      {pos_report.agreement_count} / {pos_report.total_pairs}", flush=True)
    print(f"  Position Flip Count:       {pos_report.flip_count}", flush=True)
    print(f"  Position Flip Rate:        {pos_report.flip_rate * 100:.1f}%", flush=True)
    print(f"  Candidate Presented 1st Win Rate: {pos_report.a_first_win_rate * 100:.1f}%", flush=True)
    print(f"  Mitigation in Code:        Dual-order consensus & score averaging active.", flush=True)

    # 3. Verbosity Bias Probe (Padded vs Terse)
    print("\n--- 3. Evaluating Verbosity Bias (Padded vs Concise Probes) ---", flush=True)
    adv_path = SUITE_DIR / "adversarial_probes.json"
    with open(adv_path, "r", encoding="utf-8") as f:
        adv_probes = json.load(f)

    verbosity_res = bias_harness.measure_verbosity_bias(adv_probes)
    print(f"  Total Verbosity Probes:    {verbosity_res['total_cases']}", flush=True)
    print(f"  Concise Answer Wins:       {verbosity_res['terse_wins']}", flush=True)
    print(f"  Padded Verbose Wins:       {verbosity_res['verbose_wins']}", flush=True)
    print(f"  Ties:                      {verbosity_res['ties']}", flush=True)
    print(f"  Mean Score Delta (Pad-Terse): {verbosity_res['mean_score_delta_verbose_minus_terse']}", flush=True)
    print(f"  Mitigation Effective:      {verbosity_res['mitigation_effective']}", flush=True)

    # 4. Sycophancy / Adversarial Probes (Confidently Wrong Answers)
    print("\n--- 4. Evaluating Sycophancy & Style Bias (Confidently Wrong Probes) ---", flush=True)
    sycophancy_res = bias_harness.measure_sycophancy_and_style_bias(adv_probes)
    print(f"  Total Adversarial Cases:   {sycophancy_res['total_adversarial_cases']}", flush=True)
    print(f"  Judge Fooled Count:        {sycophancy_res['fooled_count']}", flush=True)
    print(f"  Judge Rejected Wrong Count:{sycophancy_res['rejected_count']}", flush=True)
    print(f"  Fool Rate:                 {sycophancy_res['fool_rate'] * 100:.1f}%", flush=True)
    print(f"  Robustness Rate:           {sycophancy_res['robustness_rate'] * 100:.1f}%", flush=True)

    # 5. Test-Retest Consistency Check (Repeat Runs at temp=0.0)
    print("\n--- 5. Evaluating Test-Retest Consistency (Repeatability) ---", flush=True)
    sample_cases = [
        TestCase(
            id=c["id"],
            input=c["input"],
            model_output=c["model_output"],
            expected_output=c.get("expected_output"),
        )
        for c in human_labeled_cases[:5]
    ]
    consistency_res = validator.validate_test_retest_consistency(sample_cases, runs=2)
    print(f"  Sample Cases Evaluated:    {consistency_res['total_cases']}", flush=True)
    print(f"  Verdict Consistency Rate:  {consistency_res['verdict_consistency_rate'] * 100:.1f}%", flush=True)
    print(f"  Flip Count on Re-run:      {consistency_res['flip_count']}", flush=True)
    print(f"  Mean Absolute Score Delta: {consistency_res['mean_score_delta']:.4f}", flush=True)

    # 6. A/B Comparison: Config A (Baseline Naive) vs Config B (Grounded RAG with Citations)
    print("\n--- 6. Running A/B Comparison & Declaring Winner ---", flush=True)
    ab_comparison_report = validator.run_ab_comparison(
        comparison_cases=ab_cases,
        label_a="Config A (Naive Uncited RAG)",
        label_b="Config B (Grounded RAG + Citations)",
    )
    print(f"  Config A Wins:             {ab_comparison_report['wins_a']}", flush=True)
    print(f"  Config B Wins:             {ab_comparison_report['wins_b']}", flush=True)
    print(f"  Ties:                      {ab_comparison_report['ties']}", flush=True)
    print(f"  Config A Mean Score:       {ab_comparison_report['mean_score_a']} / 5.0", flush=True)
    print(f"  Config B Mean Score:       {ab_comparison_report['mean_score_b']} / 5.0", flush=True)
    print(f"  DECLARED WINNER:           {ab_comparison_report['declared_winner']}", flush=True)
    print(f"  Win Margin:                +{ab_comparison_report['margin_wins']} wins", flush=True)


    # Combine into comprehensive report artifact
    full_report = {
        "timestamp": time.time(),
        "pipeline_configuration": {
            "judge_provider": jcfg.judge_provider,
            "judge_model": jcfg.judge_model,
            "generator_provider": jcfg.generator_provider,
            "generator_model": jcfg.generator_model,
            "generator_base_url": jcfg.generator_base_url,
            "audit_log_path": str(pipeline.log_path),
        },
        "human_agreement_validation": agreement_res,
        "position_bias_evaluation": {
            "summary": pos_report.model_dump(),
            "details": [r.model_dump() for r in pos_details],
        },
        "verbosity_bias_evaluation": verbosity_res,
        "sycophancy_adversarial_evaluation": sycophancy_res,
        "test_retest_consistency": consistency_res,
        "ab_comparison_verdict": ab_comparison_report,
    }

    out_file = RESULTS_DIR / "judge_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Problem 2 Evaluation Complete! Full report written to: {out_file}")
    print("=" * 65)
    return full_report


if __name__ == "__main__":
    run_full_judge_pipeline()
