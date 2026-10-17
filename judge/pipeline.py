"""Robust LLM-as-a-Judge Pipeline (Problem 2).
Supports Pointwise and Pairwise modes, reference-based and reference-free,
with resilient JSON parsing, calibrated rubrics, and auditable JSONL logging.
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import yaml

from common.llm import LLMClient
from judge.models import (
    TestCase,
    CriterionVerdict,
    PointwiseVerdict,
    PairwiseVerdict,
    SuiteReport,
)
from judge.rubrics import STANDARD_RUBRIC, CALIBRATION_ANCHORS_PROMPT

AUDIT_LOG_DEFAULT = "logs/judge_audit.jsonl"


def robust_parse_json(raw_text: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM output, handling markdown fences,
    trailing commas, unescaped characters, and malformed structures."""
    text = raw_text.strip()
    # Strip markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

    # Find outermost matching JSON braces
    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Attempt 1: Standard JSON parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt 2: Clean trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*(\}|\])", r"\1", text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Attempt 3: Regex-based extraction of key fields
    extracted: dict[str, Any] = {}
    
    # Try extracting winner for pairwise
    winner_match = re.search(r'"winner":\s*"?(A|B|TIE)"?', text, re.IGNORECASE)
    if winner_match:
        extracted["winner"] = winner_match.group(1).upper()

    # Try extracting overall score
    score_match = re.search(r'"overall_score":\s*([0-9.]+)', text)
    if score_match:
        extracted["overall_score"] = float(score_match.group(1))

    # Try extracting overall rationale
    rat_match = re.search(r'"overall_rationale":\s*"(.*?)"(?=,\s*"|\s*\})', text, re.DOTALL)
    if rat_match:
        extracted["overall_rationale"] = rat_match.group(1)

    # Try extracting criteria block
    criteria_matches = re.findall(
        r'"criterion":\s*"([^"]+)".*?"score":\s*([0-9.]+).*?"rationale":\s*"(.*?)"',
        text,
        re.DOTALL,
    )
    if criteria_matches:
        extracted["criteria"] = [
            {"criterion": c, "score": float(s), "rationale": r.strip()}
            for c, s, r in criteria_matches
        ]

    if extracted:
        return extracted

    raise ValueError(f"Failed to parse structured JSON from judge output:\n{raw_text[:300]}...")


class JudgePipeline:
    def __init__(
        self,
        judge_client: LLMClient,
        log_path: str = AUDIT_LOG_DEFAULT,
        rubric: dict[str, Any] | None = None,
    ):
        self.client = judge_client
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rubric = rubric or STANDARD_RUBRIC

    @property
    def calls(self) -> int | None:
        return getattr(self.client, "calls", None)

    def _log_call(
        self,
        case_id: str,
        mode: str,
        prompt: str,
        response_text: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        record = {
            "timestamp": time.time(),
            "case_id": case_id,
            "mode": mode,
            "judge_model": self.client.model,
            "latency_ms": latency_ms,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "prompt": prompt,
            "raw_response": response_text,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def evaluate_pointwise(
        self,
        test_case: TestCase,
        pass_threshold: float = 3.5,
    ) -> PointwiseVerdict:
        """Pointwise scoring mode: produces structured per-criterion verdicts
        and an overall calibrated score with rationale."""
        criteria_to_use = test_case.criteria or list(self.rubric.keys())
        
        rubric_descriptions = []
        for c_name in criteria_to_use:
            crit = self.rubric.get(c_name)
            if crit:
                rubric_descriptions.append(
                    f"### Criterion: {crit.name} (Weight: {crit.weight})\n"
                    f"Description: {crit.description}\n"
                    f"- 1: {crit.anchor_1}\n"
                    f"- 3: {crit.anchor_3}\n"
                    f"- 5: {crit.anchor_5}"
                )

        rubric_text = "\n\n".join(rubric_descriptions)
        ref_text = f"\nExpected Output: {test_case.expected_output}" if test_case.expected_output else ""
        ctx_text = f"\nReference Context: {test_case.reference_context}" if test_case.reference_context else ""

        prompt = f"""You are an expert AI evaluation judge assessing the quality of a model's output.

[Input Prompt]:
{test_case.input}
{ref_text}
{ctx_text}

[Candidate Model Output]:
{test_case.model_output}

{CALIBRATION_ANCHORS_PROMPT}

### Evaluation Rubrics:
{rubric_text}

### Instructions:
1. Provide a detailed, per-criterion rationale BEFORE assigning each numerical score (1.0 to 5.0).
2. Penalize any ungrounded verbosity, fluff, or hallucinations.
3. Compute the overall score (1.0 to 5.0) reflecting the criteria weights.
4. Output your response STRICTLY as valid JSON matching this schema:
{{
  "criteria": [
    {{
      "criterion": "correctness",
      "rationale": "...",
      "score": 5.0
    }}
  ],
  "overall_score": 4.8,
  "overall_rationale": "Summary rationale..."
}}
"""
        t0 = time.perf_counter()
        res = self.client.generate(prompt=prompt, temperature=0.0, max_tokens=1024)
        latency_ms = (time.perf_counter() - t0) * 1000

        tokens_in = res.input_tokens or 0
        tokens_out = res.output_tokens or 0
        self._log_call(test_case.id, "pointwise", prompt, res.text, latency_ms, tokens_in, tokens_out)

        parsed = robust_parse_json(res.text)

        criteria_verdicts = []
        for c in parsed.get("criteria", []):
            criteria_verdicts.append(
                CriterionVerdict(
                    criterion=c.get("criterion", "general"),
                    score=float(c.get("score", 3.0)),
                    rationale=c.get("rationale", "No rationale provided."),
                )
            )

        overall_score = float(parsed.get("overall_score", 3.0))
        passed = overall_score >= pass_threshold

        return PointwiseVerdict(
            case_id=test_case.id,
            overall_score=overall_score,
            passed=passed,
            criteria_scores=criteria_verdicts,
            overall_rationale=parsed.get("overall_rationale", "Holistic assessment complete."),
            judge_model=self.client.model,
            latency_ms=latency_ms,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
        )

    def evaluate_pairwise(
        self,
        case_id: str,
        user_input: str,
        output_a: str,
        output_b: str,
        order: Literal["A_then_B", "B_then_A"] = "A_then_B",
        expected_output: str | None = None,
        reference_context: str | None = None,
    ) -> PairwiseVerdict:
        """Pairwise A-vs-B comparison mode: evaluates two candidates side-by-side,
        handling position control, and outputs winner with structured rationale."""
        first_label, second_label = ("Option A", "Option B") if order == "A_then_B" else ("Option B", "Option A")
        first_text, second_text = (output_a, output_b) if order == "A_then_B" else (output_b, output_a)

        ref_str = f"\nExpected Gold Answer: {expected_output}" if expected_output else ""
        ctx_str = f"\nReference Context: {reference_context}" if reference_context else ""

        prompt = f"""You are an expert impartial judge comparing two model responses.

[User Prompt]:
{user_input}
{ref_str}
{ctx_str}

[{first_label}]:
{first_text}

[{second_label}]:
{second_text}

### Instructions:
1. Compare both candidates on correctness, completeness, instruction-following, and conciseness.
2. Penalize unsupported length, repetition, or sycophancy: a concise and correct answer is superior to a verbose padded answer.
3. Declare the winner: 'A' (Option A is better), 'B' (Option B is better), or 'TIE' (equally good/bad).
4. Assign scores from 1.0 to 5.0 to each candidate.
5. Output your evaluation STRICTLY as valid JSON in this schema:
{{
  "winner": "A",
  "score_a": 4.5,
  "score_b": 2.0,
  "rationale": "Comparative rationale explaining why the winner prevailed..."
}}
"""
        t0 = time.perf_counter()
        res = self.client.generate(prompt=prompt, temperature=0.0, max_tokens=1024)
        latency_ms = (time.perf_counter() - t0) * 1000

        tokens_in = res.input_tokens or 0
        tokens_out = res.output_tokens or 0
        self._log_call(case_id, f"pairwise_{order}", prompt, res.text, latency_ms, tokens_in, tokens_out)

        parsed = robust_parse_json(res.text)
        raw_winner = str(parsed.get("winner", "TIE")).strip().upper()
        if raw_winner not in ("A", "B", "TIE"):
            raw_winner = "TIE"

        score_a = float(parsed.get("score_a", 3.0))
        score_b = float(parsed.get("score_b", 3.0))

        return PairwiseVerdict(
            case_id=case_id,
            order=order,
            winner=raw_winner,  # type: ignore[arg-type]
            score_a=score_a,
            score_b=score_b,
            rationale=parsed.get("rationale", "Pairwise evaluation complete."),
            judge_model=self.client.model,
            latency_ms=latency_ms,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
        )

    def evaluate_suite_pointwise(
        self,
        suite_path: str | Path,
        pass_threshold: float = 3.5,
    ) -> tuple[SuiteReport, list[PointwiseVerdict]]:
        path = Path(suite_path)
        content = path.read_text(encoding="utf-8")
        raw_cases = yaml.safe_load(content) if path.suffix in (".yaml", ".yml") else json.loads(content)

        test_cases = [TestCase(**c) for c in raw_cases]

        verdicts: list[PointwiseVerdict] = []
        criterion_totals: dict[str, list[float]] = {}
        total_lat = 0.0
        tot_in = 0
        tot_out = 0

        for tc in test_cases:
            v = self.evaluate_pointwise(tc, pass_threshold=pass_threshold)
            verdicts.append(v)
            total_lat += v.latency_ms
            tot_in += v.input_tokens
            tot_out += v.output_tokens
            for cs in v.criteria_scores:
                criterion_totals.setdefault(cs.criterion, []).append(cs.score)

        pass_count = sum(1 for v in verdicts if v.passed)
        pass_rate = pass_count / len(verdicts) if verdicts else 0.0
        mean_overall = sum(v.overall_score for v in verdicts) / len(verdicts) if verdicts else 0.0
        crit_means = {k: round(sum(vals) / len(vals), 2) for k, vals in criterion_totals.items()}

        # Cost estimation: Gemini 3.5 Flash: $0.15 / 1M input tokens, $0.60 / 1M output tokens
        est_cost = (tot_in / 1_000_000) * 0.15 + (tot_out / 1_000_000) * 0.60

        report = SuiteReport(
            mode="pointwise",
            total_cases=len(verdicts),
            pass_count=pass_count,
            pass_rate=round(pass_rate, 4),
            mean_overall_score=round(mean_overall, 2),
            criterion_means=crit_means,
            judge_model=self.client.model,
            generator_model=test_cases[0].metadata.get("generator_model", "unknown") if test_cases else "unknown",
            total_latency_ms=round(total_lat, 1),
            total_input_tokens=tot_in,
            total_output_tokens=tot_out,
            estimated_cost_usd=round(est_cost, 6),
        )
        return report, verdicts
