"""Comprehensive RAG Evaluation Harness (Problem 1).
Evaluates all three required layers:
1. Retrieval IR Metrics: Recall@k / Hit Rate, MRR, nDCG@k, Context Precision
2. Answer Quality: Faithfulness/Groundedness, Answer Relevance (LLM judge), Token F1 / EM
3. Cost & Latency: Empirical p50/p95 latency and monthly cost projections at 100K/1M/10M scale vs managed DBs.
"""
from __future__ import annotations
import json
import math
import os
import re
import string
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common.config import RagConfig, JudgeConfig
from common.llm import LLMClient, make_llm
from rag.service import RagService, RagResult


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return (1.0, 1.0, 1.0) if pred_tokens == gold_tokens else (0.0, 0.0, 0.0)

    common = {}
    for t in pred_tokens:
        if t in gold_tokens:
            common[t] = common.get(t, 0) + 1

    same_count = sum(min(pred_tokens.count(k), gold_tokens.count(k)) for k in common)
    if same_count == 0:
        return 0.0, 0.0, 0.0

    precision = 1.0 * same_count / len(pred_tokens)
    recall = 1.0 * same_count / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# --- IR Metrics ---

def compute_reciprocal_rank(retrieved_doc_ids: list[str], target_doc_id: str | None) -> float:
    if not target_doc_id:
        return 1.0 if not retrieved_doc_ids else 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == target_doc_id:
            return 1.0 / rank
    return 0.0


def compute_ndcg(retrieved_doc_ids: list[str], target_doc_id: str | None, k: int = 5) -> float:
    if not target_doc_id:
        return 1.0 if not retrieved_doc_ids else 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        rel = 1.0 if doc_id == target_doc_id else 0.0
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)
    idcg = (2.0 ** 1.0 - 1.0) / math.log2(1 + 1)
    return dcg / idcg if idcg > 0 else 0.0


def compute_context_precision(retrieved_chunks: list[dict], target_doc_id: str | None) -> float:
    if not target_doc_id:
        return 1.0 if not retrieved_chunks else 0.0
    relevant_count = 0
    precision_sum = 0.0
    for i, c in enumerate(retrieved_chunks, start=1):
        if c.get("doc_id") == target_doc_id:
            relevant_count += 1
            precision_sum += relevant_count / i
    return precision_sum / relevant_count if relevant_count > 0 else 0.0


# --- LLM-as-Judge for Faithfulness & Answer Relevance ---

JUDGE_PROMPT_TEMPLATE = """You are an expert impartial AI evaluator.
Evaluate the following generated answer against the retrieved context and user question.

Question: {query}

Retrieved Context Excerpts:
{context}

Generated Answer:
{answer}

Rate on two criteria from 1 to 5:
1. Faithfulness (1-5): Does the answer contain ONLY claims directly supported by the context?
   5 = Completely faithful, no hallucinations or ungrounded facts.
   3 = Minor ungrounded extrapolation or unsupported detail.
   1 = Severe hallucinations or directly contradicts context.
   Note: If context was insufficient and the model honestly stated it does not have enough context, rate 5.

2. Answer Relevance (1-5): Does the answer directly address the user's question?
   5 = Directly, accurately, and completely addresses what was asked.
   3 = Partially answers the question or includes excessive irrelevant information.
   1 = Off-topic or fails to answer the question.
   Note: If the question was out-of-domain and the model correctly gave the no-context refusal, rate 5.

Output your verdict strictly as valid JSON in this exact schema:
{{
  "faithfulness_score": 5,
  "faithfulness_rationale": "...",
  "relevance_score": 5,
  "relevance_rationale": "..."
}}
Do NOT output any markdown backticks or commentary outside the JSON.
"""


def judge_answer(judge_llm: LLMClient, query: str, context: str, answer: str) -> dict[str, Any]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, context=context, answer=answer)
    response = judge_llm.generate(prompt=prompt, temperature=0.0, max_tokens=512)
    text = response.text.strip()
    # Strip markdown code blocks if any
    clean_json = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    clean_json = re.sub(r"\s*```$", "", clean_json, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean_json)
    except Exception:
        # Fallback regex parsing
        f_score = re.search(r'"faithfulness_score":\s*(\d+)', clean_json)
        r_score = re.search(r'"relevance_score":\s*(\d+)', clean_json)
        return {
            "faithfulness_score": int(f_score.group(1)) if f_score else 4,
            "faithfulness_rationale": "Parsed via regex fallback",
            "relevance_score": int(r_score.group(1)) if r_score else 4,
            "relevance_rationale": "Parsed via regex fallback",
        }


# --- Cost Modeling ---

def compute_cost_projections():
    """Generates cost comparison across 100K, 1M, 10M vectors.
    Assumptions:
    - 768-dimensional float32 vectors (3,072 bytes raw per vector)
    - Storage with IVF-PQ metadata & indices: ~3.5 KB per vector
    - 100K vectors: ~350 MB
    - 1M vectors: ~3.5 GB
    - 10M vectors: ~35 GB
    - AWS S3 Standard: $0.023 / GB / month
    - Pinecone Pod: Standard s1 pod ($70/pod/mo) holds ~1M 768d vectors; 10M requires 10 pods + replica = 20 pods ($1,400/mo)
    - Pinecone Serverless: $0.25 / GB / month storage + $0.008 per 1,000 query units
    - AWS OpenSearch (Self-hosted managed): m6g.large.search (2 vCPU, 8 GB RAM) at $90/mo for 1M; r6g.2xlarge ($720/mo) for 10M
    """
    scales = [
        {"vectors": 100_000, "label": "100K"},
        {"vectors": 1_000_000, "label": "1M"},
        {"vectors": 10_000_000, "label": "10M"},
    ]
    comparison = []
    for s in scales:
        n = s["vectors"]
        gb = (n * 3500) / (1024 ** 3)
        
        # LanceDB (Local SSD)
        lancedb_local_monthly = 0.0  # Shared application volume / NVMe
        # LanceDB (AWS S3)
        lancedb_s3_monthly = round(gb * 0.023, 2)
        
        # Pinecone Pod-based
        if n <= 100_000:
            pinecone_pod = 70.0  # 1 pod min
        elif n <= 1_000_000:
            pinecone_pod = 140.0 # 2 pods (primary + replica)
        else:
            pinecone_pod = 1400.0 # 10 pods * 2 replicas = 20 pods
            
        # Pinecone Serverless
        # Assuming 10,000 queries/month
        pinecone_serverless = round(gb * 0.25 + (10_000 / 1000) * 0.008, 2)

        # AWS OpenSearch (Managed)
        if n <= 100_000:
            opensearch = 85.0
        elif n <= 1_000_000:
            opensearch = 180.0
        else:
            opensearch = 1450.0

        comparison.append({
            "vector_count": s["label"],
            "raw_storage_gb": round(gb, 2),
            "lancedb_local_cost_per_month": f"${lancedb_local_monthly:.2f}",
            "lancedb_s3_cost_per_month": f"${lancedb_s3_monthly:.2f}",
            "pinecone_serverless_cost_per_month": f"${pinecone_serverless:.2f}",
            "pinecone_pod_cost_per_month": f"${pinecone_pod:.2f}",
            "aws_opensearch_cost_per_month": f"${opensearch:.2f}",
            "cost_saving_factor_vs_pod": f"{pinecone_pod / max(lancedb_s3_monthly, 0.01):.1f}x",
        })
    return comparison


def run_rag_evaluation(eval_questions_path: str = "data/eval_questions.json"):
    print("=" * 60)
    print("Starting Comprehensive RAG Evaluation (Problem 1)")
    print("=" * 60)

    cfg = RagConfig.from_env()
    jcfg = JudgeConfig.from_env()
    service = RagService(config=cfg)
    judge_llm = make_llm(jcfg.judge_provider, jcfg.judge_model)

    with open(eval_questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} evaluation cases.")

    results_per_case = []
    retrieval_latencies = []
    total_latencies = []

    mrr_list = []
    ndcg_list = []
    hit_rate_list = []
    context_precision_list = []
    em_list = []
    f1_list = []
    faithfulness_scores = []
    relevance_scores = []

    for i, item in enumerate(questions, 1):
        qid = item["id"]
        query = item["query"]
        gold_ans = item["gold_answer"]
        target_doc = item["doc_id"]
        category = item["category"]

        print(f"\n[{i}/{len(questions)}] ({category}) {query}", flush=True)

        rag_res = service.query(query)

        retrieval_latencies.append(rag_res.retrieval_ms)
        total_latencies.append(rag_res.total_ms)

        retrieved_doc_ids = [c.get("doc_id") for c in rag_res.retrieved_chunks]

        # 1. Retrieval Metrics
        if category == "out_of_domain":
            # For OOD: correct behavior is no_context == True
            is_hit = 1.0 if rag_res.no_context else 0.0
            rr = 1.0 if rag_res.no_context else 0.0
            ndcg = 1.0 if rag_res.no_context else 0.0
            c_prec = 1.0 if rag_res.no_context else 0.0
        else:
            is_hit = 1.0 if target_doc in retrieved_doc_ids else 0.0
            rr = compute_reciprocal_rank(retrieved_doc_ids, target_doc)
            ndcg = compute_ndcg(retrieved_doc_ids, target_doc, k=cfg.k)
            c_prec = compute_context_precision(rag_res.retrieved_chunks, target_doc)

        hit_rate_list.append(is_hit)
        mrr_list.append(rr)
        ndcg_list.append(ndcg)
        context_precision_list.append(c_prec)

        # 2. Answer Metrics
        em = compute_em(rag_res.answer, gold_ans)
        _, _, f1 = compute_f1(rag_res.answer, gold_ans)
        em_list.append(em)
        f1_list.append(f1)

        # 3. LLM Judge
        context_text = "\n\n".join([c.get("text", "") for c in rag_res.retrieved_chunks])
        judge_verdict = judge_answer(judge_llm, query, context_text, rag_res.answer)
        f_score = judge_verdict.get("faithfulness_score", 5)
        r_score = judge_verdict.get("relevance_score", 5)
        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)

        print(f"  -> Answer: {rag_res.answer[:80]}...")
        print(f"  -> Hit={is_hit:.0f}, MRR={rr:.2f}, nDCG={ndcg:.2f}, F1={f1:.2f}, Faithfulness={f_score}/5, Relevance={r_score}/5")

        results_per_case.append({
            "id": qid,
            "query": query,
            "category": category,
            "target_doc": target_doc,
            "gold_answer": gold_ans,
            "model_answer": rag_res.answer,
            "citations": rag_res.citations,
            "no_context_triggered": rag_res.no_context,
            "retrieval_ms": rag_res.retrieval_ms,
            "generation_ms": rag_res.generation_ms,
            "total_ms": rag_res.total_ms,
            "input_tokens": rag_res.input_tokens,
            "output_tokens": rag_res.output_tokens,
            "retrieval_metrics": {
                "hit": is_hit,
                "reciprocal_rank": rr,
                "ndcg": ndcg,
                "context_precision": c_prec,
            },
            "answer_metrics": {
                "exact_match": em,
                "f1": f1,
                "faithfulness_score": f_score,
                "faithfulness_rationale": judge_verdict.get("faithfulness_rationale", ""),
                "relevance_score": r_score,
                "relevance_rationale": judge_verdict.get("relevance_rationale", ""),
            }
        })

    # Summary Aggregation
    summary = {
        "dataset_size": len(questions),
        "retrieval_evaluation": {
            "hit_rate_at_k": round(float(np.mean(hit_rate_list)), 4),
            "mrr": round(float(np.mean(mrr_list)), 4),
            "ndcg_at_k": round(float(np.mean(ndcg_list)), 4),
            "context_precision": round(float(np.mean(context_precision_list)), 4),
        },
        "answer_evaluation": {
            "mean_faithfulness_1_to_5": round(float(np.mean(faithfulness_scores)), 2),
            "mean_relevance_1_to_5": round(float(np.mean(relevance_scores)), 2),
            "exact_match": round(float(np.mean(em_list)), 4),
            "mean_f1": round(float(np.mean(f1_list)), 4),
        },
        "latency_benchmarks_ms": {
            "retrieval_p50": round(float(np.percentile(retrieval_latencies, 50)), 2),
            "retrieval_p95": round(float(np.percentile(retrieval_latencies, 95)), 2),
            "total_p50": round(float(np.percentile(total_latencies, 50)), 2),
            "total_p95": round(float(np.percentile(total_latencies, 95)), 2),
        },
        "cost_projections": compute_cost_projections(),
    }

    output_payload = {
        "summary": summary,
        "cases": results_per_case,
    }

    out_file = RESULTS_DIR / "rag_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY REPORT")
    print("=" * 60)
    print(f"Retrieval HitRate@k:     {summary['retrieval_evaluation']['hit_rate_at_k'] * 100:.1f}%")
    print(f"Retrieval MRR:           {summary['retrieval_evaluation']['mrr']:.4f}")
    print(f"Retrieval nDCG@k:        {summary['retrieval_evaluation']['ndcg_at_k']:.4f}")
    print(f"Context Precision:       {summary['retrieval_evaluation']['context_precision']:.4f}")
    print(f"Answer Faithfulness:     {summary['answer_evaluation']['mean_faithfulness_1_to_5']:.2f} / 5.0")
    print(f"Answer Relevance:        {summary['answer_evaluation']['mean_relevance_1_to_5']:.2f} / 5.0")
    print(f"Token F1 Score:          {summary['answer_evaluation']['mean_f1']:.4f}")
    print(f"Retrieval Latency (p50): {summary['latency_benchmarks_ms']['retrieval_p50']:.1f} ms")
    print(f"Retrieval Latency (p95): {summary['latency_benchmarks_ms']['retrieval_p95']:.1f} ms")
    print(f"Total Latency (p50):     {summary['latency_benchmarks_ms']['total_p50']:.1f} ms")
    print(f"Total Latency (p95):     {summary['latency_benchmarks_ms']['total_p95']:.1f} ms")
    print(f"\nFull report written to: {out_file}")

    return output_payload


if __name__ == "__main__":
    run_rag_evaluation()
