"""Data models for LLM-as-a-Judge Evaluation Pipeline (Problem 2).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Literal
from pydantic import BaseModel, Field


class CriterionVerdict(BaseModel):
    criterion: str = Field(..., description="Name of the rubric criterion")
    score: float = Field(..., description="Score on 1-5 scale")
    rationale: str = Field(..., description="Detailed justification grounded in rubric")


class TestCase(BaseModel):
    __test__ = False
    id: str = Field(..., description="Unique test case identifier")

    input: str = Field(..., description="User prompt or question")
    system_prompt: str = Field(default="", description="System prompt under test")
    model_output: str = Field(..., description="Output produced by candidate generator")
    expected_output: str | None = Field(default=None, description="Gold or reference output if reference-based")
    reference_context: str | None = Field(default=None, description="Context excerpts if grounding evaluation")
    criteria: list[str] | None = Field(default=None, description="Specific criteria to evaluate, or None for standard rubric")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")


class PointwiseVerdict(BaseModel):
    case_id: str
    overall_score: float = Field(..., ge=1.0, le=5.0)
    passed: bool
    criteria_scores: list[CriterionVerdict]
    overall_rationale: str
    judge_model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class PairwiseVerdict(BaseModel):
    case_id: str
    order: Literal["A_then_B", "B_then_A"]
    winner: Literal["A", "B", "TIE"]
    score_a: float
    score_b: float
    rationale: str
    judge_model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class SymmetricalPairwiseResult(BaseModel):
    case_id: str
    forward_winner: Literal["A", "B", "TIE"]  # A_then_B
    reverse_winner: Literal["A", "B", "TIE"]  # B_then_A (normalized to original A and B)
    consensus_winner: Literal["A", "B", "TIE"]
    flipped: bool
    mean_score_a: float
    mean_score_b: float
    forward_rationale: str
    reverse_rationale: str


class PositionBiasReport(BaseModel):
    total_pairs: int
    agreement_count: int
    flip_count: int
    flip_rate: float
    a_first_win_rate: float  # Percentage of times the first-presented candidate won
    b_first_win_rate: float
    tie_rate: float


class SuiteReport(BaseModel):
    mode: str
    total_cases: int
    pass_count: int
    pass_rate: float
    mean_overall_score: float
    criterion_means: dict[str, float]
    judge_model: str
    generator_model: str
    total_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
