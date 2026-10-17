import pytest
from common.llm import ScriptedLLM
from judge.models import TestCase
from judge.pipeline import JudgePipeline, robust_parse_json
from judge.bias import BiasMitigationHarness
from judge.validation import compute_cohens_kappa


def test_robust_parse_json_variations():
    # 1. Clean JSON
    assert robust_parse_json('{"winner": "A", "score_a": 4.5}') == {"winner": "A", "score_a": 4.5}

    # 2. Markdown code block
    fenced = '```json\n{"winner": "B", "score_b": 3.0}\n```'
    assert robust_parse_json(fenced) == {"winner": "B", "score_b": 3.0}

    # 3. Trailing comma
    trailing = '{"winner": "TIE", "score_a": 4.0,}'
    assert robust_parse_json(trailing) == {"winner": "TIE", "score_a": 4.0}

    # 4. Fallback extraction with extraneous commentary
    messy = 'Here is the verdict: {"winner": "A", "overall_score": 4.8, "overall_rationale": "High quality answer."} Hope this helps!'
    parsed = robust_parse_json(messy)
    assert parsed["winner"] == "A"
    assert parsed["overall_score"] == 4.8


def test_pointwise_evaluation_with_scripted_llm(tmp_path):
    mock_response = """```json
{
  "criteria": [
    {"criterion": "correctness", "score": 5.0, "rationale": "Factually flawless."},
    {"criterion": "faithfulness", "score": 4.5, "rationale": "Well grounded."}
  ],
  "overall_score": 4.8,
  "overall_rationale": "Strong response."
}
```"""
    llm = ScriptedLLM(responses=[mock_response])
    pipeline = JudgePipeline(judge_client=llm, log_path=str(tmp_path / "audit.jsonl"))

    tc = TestCase(
        id="case_1",
        input="What is LanceDB?",
        model_output="LanceDB is a serverless vector database.",
    )
    verdict = pipeline.evaluate_pointwise(tc, pass_threshold=3.5)

    assert verdict.passed is True
    assert verdict.overall_score == 4.8
    assert len(verdict.criteria_scores) == 2
    assert verdict.criteria_scores[0].score == 5.0


def test_dual_order_pairwise_neutralizes_position_bias(tmp_path):
    # Simulate a judge that always favors the first presented candidate
    # Forward pass (A then B): declares A winner
    # Reverse pass (B then A): declares B winner (meaning it preferred Option 2 when flipped)
    llm = ScriptedLLM(
        responses=[
            '{"winner": "A", "score_a": 4.5, "score_b": 2.0, "rationale": "Option A was first."}',
            '{"winner": "B", "score_a": 2.0, "score_b": 4.5, "rationale": "Option B was first."}',
        ]
    )
    pipeline = JudgePipeline(judge_client=llm, log_path=str(tmp_path / "audit.jsonl"))
    harness = BiasMitigationHarness(pipeline)

    res = harness.evaluate_symmetrical_pairwise(
        case_id="case_symm_1",
        user_input="Explain asyncio",
        output_a="Asyncio uses an event loop.",
        output_b="Asyncio uses an event loop.",
    )

    # Detects position flip!
    assert res.flipped is True
    # Averages scores to achieve neutral consensus
    assert res.consensus_winner == "TIE"
    assert res.mean_score_a == res.mean_score_b


def test_cohens_kappa_calculation():
    # Perfect agreement
    assert compute_cohens_kappa([True, False, True], [True, False, True]) == 1.0

    # High agreement
    rater_a = [True, True, False, False, True, True]
    rater_b = [True, True, False, False, True, False]
    kappa = compute_cohens_kappa(rater_a, rater_b)
    assert 0.60 < kappa < 1.0

    # Inverse agreement
    assert compute_cohens_kappa([True, False], [False, True]) == -1.0

