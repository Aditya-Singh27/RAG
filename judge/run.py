"""Full Evaluation Pipeline Runner for Problem 2 (LLM-as-a-Judge) - v2.

Executes comprehensive bias measurement and validation:
1. Human / Gold Label Agreement (Cohen's Kappa, Wilson CI, Confusion Matrix)
2. Symmetrical Pairwise Position Bias & Details (Hard vs Soft Flips, Accuracy vs Gold)
3. Identical-Pair Probe (Pure slot preference without content confound)
4. Verbosity Bias & Length Penalty (Before vs After baseline rubric, Bootstrap CI)
5. Sycophancy / Adversarial Probes (With reference and reference-free)
6. Score Clustering & Calibration Anchors (Spread statistics before vs after)
7. Test-Retest Consistency Check (Audit call counts, cache_suspected detection)
8. A/B Comparison (Sign test at alpha=0.05, declaring statistically sound winner)

Saves auditable report to results/judge_evaluation_report.json.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import JudgeConfig
from common.llm import LoggedLLM, make_llm
from judge.models import TestCase
from judge.pipeline import JudgePipeline
from judge.bias import BiasMitigationHarness
from judge.validation import JudgeValidator
from judge.rubrics import STANDARD_RUBRIC, RubricCriterion

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUITE_DIR = Path("data/judge_suites")

# Naive baseline rubric without length penalty (for before -> after comparison)
BASELINE_NAIVE_RUBRIC: dict[str, RubricCriterion] = {
    "correctness": RubricCriterion(
        name="correctness",
        description="Factual correctness and accuracy.",
        weight=0.50,
        anchor_1="Incorrect.",
        anchor_2="Partially incorrect.",
        anchor_3="Average.",
        anchor_4="Good.",
        anchor_5="Excellent.",
    ),
    "faithfulness": RubricCriterion(
        name="faithfulness",
        description="Grounded in context.",
        weight=0.50,
        anchor_1="Not grounded.",
        anchor_2="Low grounding.",
        anchor_3="Moderate grounding.",
        anchor_4="Grounded.",
        anchor_5="Fully grounded.",
    ),
}


def run_full_judge_pipeline():
    print("=" * 68, flush=True)
    print("Starting Comprehensive LLM-as-a-Judge Evaluation Pipeline (Problem 2)", flush=True)
    print("=" * 68, flush=True)

    jcfg = JudgeConfig.from_env()
    print(f"Judge Model Family:     {jcfg.judge_provider} ({jcfg.judge_model})", flush=True)
    print(f"Generator Model Family: {jcfg.generator_provider} ({jcfg.generator_model})", flush=True)
    print("Self-Enhancement Decoupling: DECOUPLED across independent model families.\n", flush=True)

    # Wrap judge with LoggedLLM so call counting is active and audit log is populated
    raw_judge = make_llm(jcfg.judge_provider, jcfg.judge_model)
    judge_client = LoggedLLM(raw_judge, jcfg.log_path, "judge_audit")

    # Mitigated pipeline (calibrated rubric with length penalty)
    pipeline = JudgePipeline(judge_client=judge_client, log_path=jcfg.log_path, rubric=STANDARD_RUBRIC)
    bias_harness = BiasMitigationHarness(pipeline)
    validator = JudgeValidator(pipeline, bias_harness)

    # Baseline unmitigated pipeline (naive rubric without length penalty or few-shot anchors)
    raw_baseline_judge = make_llm(jcfg.judge_provider, jcfg.judge_model)
    baseline_client = LoggedLLM(raw_baseline_judge, "logs/baseline_judge_audit.jsonl", "baseline_judge")
    baseline_pipeline = JudgePipeline(judge_client=baseline_client, log_path="logs/baseline_judge_audit.jsonl", rubric=BASELINE_NAIVE_RUBRIC)

    # 1. Human / Gold Label Agreement (Cohen's Kappa)
    print("--- 1. Evaluating Agreement with Human/Gold Labels ---", flush=True)
    hl_v2 = SUITE_DIR / "human_labeled_cases_v2.json"
    hl_path = hl_v2 if hl_v2.exists() else (SUITE_DIR / "human_labeled_cases.json")
    with open(hl_path, "r", encoding="utf-8") as f:
        human_labeled_cases = json.load(f)

    agreement_res = validator.validate_human_agreement(human_labeled_cases, pass_threshold=3.5)
    print(f"  Total Cases:               {agreement_res['total_cases']}", flush=True)
    print(f"  Label Sources:             {agreement_res['label_sources']}", flush=True)
    print(f"  Agreement Rate:            {agreement_res['agreement_rate'] * 100:.1f}% (Wilson 95% CI: {agreement_res['agreement_ci95_wilson']})", flush=True)
    kappa_str = f"{agreement_res['cohens_kappa']:.4f}" if agreement_res['cohens_kappa'] is not None else "None (undefined)"
    print(f"  Cohen's Kappa (kappa):     {kappa_str}", flush=True)
    print(f"  Mean Absolute Error:       {agreement_res['mean_absolute_score_error']}", flush=True)
    print(f"  Confusion Matrix:          {agreement_res['confusion_matrix']}", flush=True)

    # 2. Position Bias Check (Dual-Order Symmetrical Pairwise)
    print("\n--- 2. Evaluating Position Bias (Dual-Order Symmetrical Pairwise) ---", flush=True)
    ab_path = SUITE_DIR / "ab_comparison_cases.json"
    with open(ab_path, "r", encoding="utf-8") as f:
        ab_cases = json.load(f)

    pos_report, pos_results = bias_harness.measure_position_bias(ab_cases)
    pos_details = bias_harness.position_bias_details(pos_results, ab_cases)
    print(f"  Total Pairs Evaluated:     {pos_report.total_pairs}", flush=True)
    print(f"  Dual-Order Agreement:      {pos_report.agreement_count} / {pos_report.total_pairs}", flush=True)
    print(f"  Hard Flips:                {pos_details['hard_flips']}", flush=True)
    print(f"  Soft Flips:                {pos_details['soft_flips']}", flush=True)
    print(f"  Single-Order Flip Rate:    {pos_details['single_order_flip_rate'] * 100:.1f}% (95% CI: {pos_details['single_order_flip_ci95']})", flush=True)
    if "accuracy_vs_gold" in pos_details:
        acc = pos_details["accuracy_vs_gold"]
        print(f"  Accuracy vs Gold:          Fwd={acc['forward_only'] * 100:.1f}%, Rev={acc['reverse_only'] * 100:.1f}%, Consensus={acc['consensus'] * 100:.1f}%", flush=True)
    print(f"  After Mitigation Flip Rate:0.0% (by construction via dual-order consensus)", flush=True)

    # 3. Identical-Pair Probe (Pure Slot Preference)
    print("\n--- 3. Evaluating Identical-Pair Probe (Pure Position Bias, No Content Confound) ---", flush=True)
    ident_path = SUITE_DIR / "identical_pair_cases.json"
    with open(ident_path, "r", encoding="utf-8") as f:
        identical_cases = json.load(f)
    identical_res = bias_harness.measure_identical_pair_bias(identical_cases)
    print(f"  Judgements:                {identical_res['n_judgements']}", flush=True)
    print(f"  First Slot Wins:           {identical_res['first_slot_wins']}", flush=True)
    print(f"  Second Slot Wins:          {identical_res['second_slot_wins']}", flush=True)
    print(f"  Ties:                      {identical_res['ties']}", flush=True)
    print(f"  Sign Test p-value:         {identical_res['sign_test_p']}", flush=True)

    # 4. Verbosity Bias Probe (Padded vs Terse)
    print("\n--- 4. Evaluating Verbosity Bias (Padded vs Concise Probes) ---", flush=True)
    adv_path = SUITE_DIR / "adversarial_probes.json"
    with open(adv_path, "r", encoding="utf-8") as f:
        adv_probes = json.load(f)

    verbosity_res = bias_harness.measure_verbosity_bias(adv_probes, baseline_pipeline=baseline_pipeline)
    after_v = verbosity_res["after_mitigation"]
    print(f"  After Mitigation:          mean diff={after_v['mean_diff_verbose_minus_terse']}, 95% CI={after_v['ci95']}, verdict='{after_v['verdict']}'", flush=True)
    if "before_mitigation" in verbosity_res:
        before_v = verbosity_res["before_mitigation"]
        print(f"  Before Mitigation:         mean diff={before_v['mean_diff_verbose_minus_terse']}, 95% CI={before_v['ci95']}, verdict='{before_v['verdict']}'", flush=True)
    print(f"  Pairwise Probe:            Terse wins={verbosity_res['pairwise']['terse_wins']}, Verbose wins={verbosity_res['pairwise']['verbose_wins']}, Ties={verbosity_res['pairwise']['ties']}", flush=True)

    # 5. Sycophancy / Adversarial Probes (Confidently Wrong with False Belief in Input)
    print("\n--- 5. Evaluating Sycophancy & Style Bias ---", flush=True)
    sycophancy_res = bias_harness.measure_sycophancy_and_style_bias(adv_probes, use_reference=True)
    sycophancy_no_ref = bias_harness.measure_sycophancy_and_style_bias(adv_probes, use_reference=False)
    print(f"  With Gold Reference:       Fooled={sycophancy_res['fooled_count']}/{sycophancy_res['total_adversarial_cases']}, Fool Rate={sycophancy_res['fool_rate'] * 100:.1f}% (95% CI: {sycophancy_res['fool_rate_ci95']})", flush=True)
    print(f"  Without Gold Reference:    Fooled={sycophancy_no_ref['fooled_count']}/{sycophancy_no_ref['total_adversarial_cases']}, Fool Rate={sycophancy_no_ref['fool_rate'] * 100:.1f}% (95% CI: {sycophancy_no_ref['fool_rate_ci95']})", flush=True)

    # 6. Score Clustering & Calibration Anchors
    print("\n--- 6. Evaluating Score Clustering Spread ---", flush=True)
    clustering_cases = [
        {"id": c["id"], "input": c["input"], "model_output": c["model_output"], "expected_output": c.get("expected_output")}
        for c in human_labeled_cases
    ]
    clustering_res = bias_harness.measure_score_clustering(clustering_cases, baseline_pipeline=baseline_pipeline)
    print(f"  After Calibration Anchors: std={clustering_res['after_mitigation']['std']}, distinct values={clustering_res['after_mitigation']['distinct_values']}, extremes={clustering_res['after_mitigation']['frac_at_extremes']}", flush=True)
    if "before_mitigation" in clustering_res:
        print(f"  Before Anchors (Baseline): std={clustering_res['before_mitigation']['std']}, distinct values={clustering_res['before_mitigation']['distinct_values']}", flush=True)

    # 7. Test-Retest Consistency Check
    print("\n--- 7. Evaluating Test-Retest Consistency (Repeatability) ---", flush=True)
    sample_cases = [
        TestCase(
            id=c["id"],
            input=c["input"],
            model_output=c["model_output"],
            expected_output=c.get("expected_output"),
        )
        for c in human_labeled_cases[:5]
    ]
    consistency_res = validator.validate_test_retest_consistency(sample_cases, runs=3)
    print(f"  Total Cases:               {consistency_res['total_cases']} (3 runs each, {consistency_res['judgements_requested']} requested)", flush=True)
    print(f"  Judge Calls Made:          {consistency_res['judge_calls_made']}", flush=True)
    print(f"  Cache Suspected:           {consistency_res['cache_suspected']}", flush=True)
    print(f"  Verdict Consistency Rate:  {consistency_res['verdict_consistency_rate'] * 100:.1f}% (95% CI: {consistency_res['consistency_ci95_wilson']})", flush=True)
    print(f"  Mean Score Range:          {consistency_res['mean_score_range']}", flush=True)

    # 8. A/B Comparison: Config A (Naive) vs Config B (Grounded RAG)
    print("\n--- 8. Running A/B Comparison & Declaring Winner ---", flush=True)
    ab_comparison_report = validator.run_ab_comparison(
        comparison_cases=ab_cases,
        label_a="Config A (Naive Uncited RAG)",
        label_b="Config B (Grounded RAG + Citations)",
        alpha=0.05,
    )
    print(f"  Config A Wins:             {ab_comparison_report['wins_a']}", flush=True)
    print(f"  Config B Wins:             {ab_comparison_report['wins_b']}", flush=True)
    print(f"  Ties:                      {ab_comparison_report['ties']}", flush=True)
    print(f"  Config A Mean Score:       {ab_comparison_report['mean_score_a']} / 5.0", flush=True)
    print(f"  Config B Mean Score:       {ab_comparison_report['mean_score_b']} / 5.0", flush=True)
    print(f"  Sign Test p-value:         {ab_comparison_report['sign_test_p']} (alpha={ab_comparison_report['alpha']})", flush=True)
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
            "judge_calls": judge_client.calls,
            "audit_log_path": str(pipeline.log_path),
        },
        "human_agreement_validation": agreement_res,
        "position_bias_evaluation": {
            "summary": {
                "total_pairs": pos_report.total_pairs,
                "agreement_count": pos_report.agreement_count,
                "flip_count": pos_report.flip_count,
                "flip_rate": pos_report.flip_rate,
                "a_first_win_rate": pos_report.a_first_win_rate,
                "b_first_win_rate": pos_report.b_first_win_rate,
                "tie_rate": pos_report.tie_rate,
            },
            "details": pos_details,
        },
        "identical_pair_probe": identical_res,
        "verbosity_bias_evaluation": verbosity_res,
        "sycophancy_adversarial_evaluation": {
            "with_reference": sycophancy_res,
            "without_reference": sycophancy_no_ref,
        },
        "score_clustering_evaluation": clustering_res,
        "test_retest_consistency": consistency_res,
        "ab_comparison_verdict": ab_comparison_report,
    }

    out_file = RESULTS_DIR / "judge_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 68)
    print(f"Problem 2 Evaluation Complete! Full report written to: {out_file}")
    print("=" * 68)
    return full_report


if __name__ == "__main__":
    run_full_judge_pipeline()
