"""RAG evaluation harness (Problem 1) - v2.

Retrieval : chunk-level relevance from `evidence` quotes; answerable questions only.
Refusal   : out-of-domain refusal accuracy and false-refusal rate reported separately.
Answers   : LLM judge (faithfulness, correctness vs gold, relevance, citation accuracy) with NO silent
            defaults: unparsable verdicts are counted and excluded, never turned into a score.
Latency   : p50/p95 with sample sizes. Cost lives in eval/cost_model.py (needs sourced prices).

eval_questions.json item:
  {id, query, category: in_domain|cross_doc|out_of_domain, gold_answer, doc_id?, evidence: [verbatim quotes]}
"""
from __future__ import annotations
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (bootstrap_ci, context_precision, context_precision_ap, count_relevant_in_corpus,
                          evidence_recall, hit_at_k, ndcg_at_k, percentile, reciprocal_rank, relevance_flags)
from judge.stats import score_spread, wilson_ci

RESULTS_DIR = Path("results")
REFUSAL = "i do not have enough relevant context"
ANSWERABLE = ("in_domain", "cross_doc")


# ----------------------------------------------------------------------------- text metrics
def normalize_answer(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def is_refusal(answer: str, no_context: bool) -> bool:
    return bool(no_context) or REFUSAL in answer.lower()


# ----------------------------------------------------------------------------- LLM judge
JUDGE_PROMPT = """You are a strict evaluator of a retrieval-augmented answer. Judge ONLY from the material below.

Question: {query}
Question type: {qtype}
Reference answer (written by a human from the source documents): {gold}

Retrieved context, numbered as shown to the answering model:
{context}

Answer under evaluation:
{answer}

Score each criterion 1-5 (integers). Quote the shortest phrase that justifies each score.
- faithfulness: every claim in the answer is supported by the retrieved context. Unsupported claims lower the score,
  even if true in the real world. 5 = all claims supported, 1 = mostly unsupported or contradicted.
- correctness: agreement with the reference answer on the key facts. A refusal on an answerable question is 1.
  For an out-of-domain question (reference = refusal) a refusal is 5 and any substantive answer is 1.
- relevance: the answer addresses what was asked, without padding.
- citation_accuracy: each [Chunk N] marker points to a chunk that supports the sentence it follows.
  If the answer has no citations and is not a refusal, give 1. If it is a refusal, give null.

Return ONLY JSON:
{{"faithfulness": int, "faithfulness_evidence": str, "correctness": int, "correctness_evidence": str,
  "relevance": int, "relevance_evidence": str, "citation_accuracy": int|null, "citation_evidence": str}}"""

SCORE_KEYS = ("faithfulness", "correctness", "relevance")


def parse_judge_json(text: str) -> dict | None:
    """Strict: returns None unless every required score is an int in 1..5."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    for k in SCORE_KEYS:
        if not isinstance(d.get(k), int) or isinstance(d.get(k), bool) or not 1 <= d[k] <= 5:
            return None
    ca = d.get("citation_accuracy")
    if ca is not None and (not isinstance(ca, int) or not 1 <= ca <= 5):
        return None
    return d


def judge_answer(judge_llm, item: dict, chunks: list[dict], answer: str) -> dict | None:
    ctx = "\n\n".join(f"[Chunk {i}] {c.get('text', '')}" for i, c in enumerate(chunks, 1)) or "(no context retrieved)"
    prompt = JUDGE_PROMPT.format(query=item["query"], qtype=item["category"], gold=item["gold_answer"],
                                 context=ctx, answer=answer)
    for attempt in range(2):
        r = judge_llm.generate(prompt=prompt, temperature=0.0, max_tokens=700, tag=f"rag_eval:{item['id']}:{attempt}")
        v = parse_judge_json(r.text)
        if v is not None:
            return v
        prompt += "\n\nYour previous reply was not valid JSON with integer scores 1-5. Reply with ONLY the JSON object."
    return None


def citation_validity(answer: str, n_chunks: int) -> float | None:
    cited = [int(x) for x in re.findall(r"\[Chunk\s+(\d+)", answer)]
    return None if not cited else sum(1 <= c <= n_chunks for c in cited) / len(cited)


# ----------------------------------------------------------------------------- aggregation
def _ci(values):
    m, lo, hi = bootstrap_ci(values)
    return {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(values)}


def build_summary(cases: list[dict], k: int, meta: dict) -> dict:
    ans = [c for c in cases if c["category"] in ANSWERABLE]
    ood = [c for c in cases if c["category"] == "out_of_domain"]
    ir = lambda key: [c["retrieval_metrics"][key] for c in ans]
    judged = lambda key, rows=ans: [c["answer_metrics"][key] for c in rows if c["answer_metrics"].get(key) is not None]
    refused_ans = sum(c["refused"] for c in ans)
    refused_ood = sum(c["refused"] for c in ood)
    unparsed = sum(c["answer_metrics"]["judge_parse_failed"] for c in cases)
    per_cat = {cat: {"n": len([c for c in ans if c["category"] == cat]),
                     "hit": round(sum(c["retrieval_metrics"]["hit"] for c in ans if c["category"] == cat) /
                                  max(1, len([c for c in ans if c["category"] == cat])), 3),
                     "mrr": round(sum(c["retrieval_metrics"]["reciprocal_rank"] for c in ans if c["category"] == cat) /
                                  max(1, len([c for c in ans if c["category"] == cat])), 3)}
               for cat in ANSWERABLE}
    tot = [c["total_ms"] for c in cases]
    gen = [c["generation_ms"] for c in cases if c["generation_ms"] > 0]
    slow = sorted(cases, key=lambda c: -c["total_ms"])[:3]
    return {
        "meta": meta, "k": k, "n_questions": len(cases), "n_answerable": len(ans), "n_out_of_domain": len(ood),
        "retrieval_answerable_only": {
            f"hit_rate@{k}": _ci(ir("hit")), "mrr": _ci(ir("reciprocal_rank")), f"ndcg@{k}": _ci(ir("ndcg")),
            f"evidence_recall@{k}": _ci(ir("evidence_recall")),
            "context_precision": _ci(ir("context_precision")), "context_precision_rank_aware": _ci(ir("context_precision_ap")),
            "by_category": per_cat},
        "refusal_behaviour": {
            "out_of_domain_refused": f"{refused_ood}/{len(ood)}",
            "out_of_domain_refusal_rate_ci95_wilson": [round(x, 3) for x in wilson_ci(refused_ood, len(ood))],
            "false_refusals_on_answerable": f"{refused_ans}/{len(ans)}",
            "false_refusal_rate_ci95_wilson": [round(x, 3) for x in wilson_ci(refused_ans, len(ans))],
            "note": f"threshold was chosen on a dev set of its own; only {len(ood)} OOD questions here"},
        "answer_quality": {
            "judge_parse_failures": f"{unparsed}/{len(cases)} (excluded, not defaulted)",
            "faithfulness": {**_ci(judged("faithfulness", cases)), "spread": score_spread(judged("faithfulness", cases))},
            "correctness_vs_gold": {**_ci(judged("correctness", cases)), "spread": score_spread(judged("correctness", cases))},
            "relevance": {**_ci(judged("relevance", cases)), "spread": score_spread(judged("relevance", cases))},
            "citation_accuracy": _ci(judged("citation_accuracy", cases)) if judged("citation_accuracy", cases) else None,
            "citation_marker_validity": _ci([c["citation_validity"] for c in cases if c["citation_validity"] is not None]),
            "token_f1_answerable": _ci([c["answer_metrics"]["f1"] for c in ans]),
            "exact_match_answerable": _ci([c["answer_metrics"]["exact_match"] for c in ans]),
            "em_note": "EM is near-meaningless for free-form answers; kept because the template asks for it"},
        "latency_ms": {
            "n": len(cases),
            "retrieval_incl_query_embedding_p50": round(percentile([c["retrieval_ms"] for c in cases], 50), 1),
            "retrieval_incl_query_embedding_p95": round(percentile([c["retrieval_ms"] for c in cases], 95), 1),
            "generation_p50": round(percentile(gen, 50), 1) if gen else None,
            "generation_p95": round(percentile(gen, 95), 1) if gen else None,
            "total_p50": round(percentile(tot, 50), 1), "total_p95": round(percentile(tot, 95), 1),
            "slowest": [{"id": c["id"], "total_ms": round(c["total_ms"])} for c in slow],
            "note": "retrieval here includes the query-embedding API call; run eval/bench_latency.py for vector-search-only latency. p95 of ~25 requests is effectively the 2nd-slowest."},
        "tokens": {"generator_input": sum(c["input_tokens"] for c in cases),
                   "generator_output": sum(c["output_tokens"] for c in cases)},
    }


# ----------------------------------------------------------------------------- run
def load_corpus_texts(lancedb_uri: str) -> list[str]:
    import lancedb
    return [r["text"] for r in lancedb.connect(lancedb_uri).open_table("chunks").to_arrow().select(["text"]).to_pylist()]


def run_rag_evaluation(path: str = "data/eval_questions.json"):
    from common.config import RagConfig, JudgeConfig
    from common.llm import LoggedLLM, make_llm
    from rag.service import RagService

    cfg, jcfg = RagConfig.from_env(), JudgeConfig.from_env()
    service = RagService(config=cfg)
    judge = LoggedLLM(make_llm(jcfg.judge_provider, jcfg.judge_model), "logs/rag_eval_judge.jsonl", "rag_answer_judge")
    questions = json.load(open(path, encoding="utf-8"))
    corpus = load_corpus_texts(cfg.lancedb_uri)
    same = (jcfg.judge_provider, jcfg.judge_model) == (cfg.gen_provider, cfg.gen_model)
    if same:
        print("WARNING: judge and generator are the same model - faithfulness scores are self-graded.")
    missing = [q["id"] for q in questions if q["category"] in ANSWERABLE and not q.get("evidence")]
    if missing:
        raise SystemExit(f"questions without `evidence` quotes (chunk-level labels required): {missing}")

    cases = []
    for i, q in enumerate(questions, 1):
        res = service.query(q["query"])
        texts = [c.get("text", "") for c in res.retrieved_chunks]
        answerable = q["category"] in ANSWERABLE
        flags = relevance_flags(texts, q["evidence"]) if answerable else []
        n_rel = count_relevant_in_corpus(corpus, q["evidence"]) if answerable else 0
        if answerable and n_rel == 0:
            raise SystemExit(f"{q['id']}: evidence quote not found in any stored chunk - fix the quote or re-ingest")
        rm = ({"hit": hit_at_k(flags, cfg.k), "reciprocal_rank": reciprocal_rank(flags, cfg.k),
               "ndcg": ndcg_at_k(flags, n_rel, cfg.k), "evidence_recall": evidence_recall(texts, q["evidence"], cfg.k),
               "context_precision": context_precision(flags, cfg.k), "context_precision_ap": context_precision_ap(flags, cfg.k),
               "relevant_chunks_in_corpus": n_rel} if answerable else {})
        v = judge_answer(judge, q, res.retrieved_chunks, res.answer)
        am = {"exact_match": exact_match(res.answer, q["gold_answer"]), "f1": token_f1(res.answer, q["gold_answer"]),
              "judge_parse_failed": v is None,
              **{k2: (v or {}).get(k2) for k2 in (*SCORE_KEYS, "citation_accuracy")},
              "judge_evidence": {k2: (v or {}).get(k2 + "_evidence") for k2 in ("faithfulness", "correctness", "relevance", "citation")}}
        cases.append({"id": q["id"], "query": q["query"], "category": q["category"], "gold_answer": q["gold_answer"],
                      "model_answer": res.answer, "citations": res.citations, "refused": is_refusal(res.answer, res.no_context),
                      "no_context_triggered": res.no_context, "citation_validity": citation_validity(res.answer, len(texts)),
                      "retrieval_ms": res.retrieval_ms, "generation_ms": res.generation_ms, "total_ms": res.total_ms,
                      "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                      "retrieval_metrics": rm, "answer_metrics": am})
        print(f"[{i}/{len(questions)}] {q['id']} {q['category']}: refused={cases[-1]['refused']} "
              f"hit={rm.get('hit')} f={am['faithfulness']} c={am['correctness']}", flush=True)

    meta = {"embed_model": cfg.embed_model, "embed_dim": cfg.embed_dim, "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap, "min_score": cfg.min_score, "generator": cfg.gen_model,
            "judge": jcfg.judge_model, "judge_equals_generator": same,
            "judge_calls": judge.calls, "judge_input_tokens": judge.input_tokens, "judge_output_tokens": judge.output_tokens,
            "corpus_chunks": len(corpus)}
    out = {"summary": build_summary(cases, cfg.k, meta), "cases": cases}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "rag_evaluation_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))
    return out


if __name__ == "__main__":
    run_rag_evaluation()
