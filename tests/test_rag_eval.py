import pytest
from eval.evaluate_rag import (build_summary, citation_validity, is_refusal, parse_judge_json, token_f1, exact_match)

GOOD = '{"faithfulness":5,"faithfulness_evidence":"x","correctness":4,"correctness_evidence":"y","relevance":5,"relevance_evidence":"z","citation_accuracy":null,"citation_evidence":""}'


def test_parse_judge_json_variants():
    assert parse_judge_json(GOOD)["correctness"] == 4
    assert parse_judge_json("```json\n" + GOOD + "\n```")["faithfulness"] == 5
    assert parse_judge_json("Sure! Here you go: " + GOOD + " Hope it helps")["relevance"] == 5


@pytest.mark.parametrize("bad", ["", "no json", "{bad json}", '{"faithfulness":5}', GOOD.replace('"faithfulness":5', '"faithfulness":9'),
                                 GOOD.replace('"relevance":5', '"relevance":"5"'), GOOD.replace('"faithfulness":5', '"faithfulness":true')])
def test_parse_failures_return_none_not_a_default_score(bad):
    assert parse_judge_json(bad) is None


def test_f1_em_refusal_citations():
    # articles are stripped: "cat sat" vs "cat sat down" -> P=1, R=2/3, F1=0.8
    assert token_f1("the cat sat", "a cat sat down") == pytest.approx(0.8)
    assert exact_match("The Cat.", "cat") == 1.0
    assert is_refusal("I do not have enough relevant context in the docs", False) and is_refusal("x", True)
    assert citation_validity("a [Chunk 1] b [Chunk 7]", 5) == 0.5 and citation_validity("none", 5) is None


def _case(cid, cat, hit=1.0, rr=1.0, refused=False, f=5, parsed=True):
    ir = {"hit": hit, "reciprocal_rank": rr, "ndcg": rr, "evidence_recall": hit, "context_precision": 0.5,
          "context_precision_ap": 0.5} if cat != "out_of_domain" else {}
    return {"id": cid, "category": cat, "refused": refused, "citation_validity": None, "retrieval_ms": 100, "generation_ms": 500,
            "total_ms": 600, "input_tokens": 10, "output_tokens": 5, "retrieval_metrics": ir,
            "answer_metrics": {"exact_match": 0.0, "f1": 0.4, "faithfulness": f if parsed else None, "correctness": 4 if parsed else None,
                               "relevance": 5 if parsed else None, "citation_accuracy": None, "judge_parse_failed": not parsed}}


def test_summary_excludes_ood_from_retrieval_and_counts_refusals():
    cases = [_case("a", "in_domain"), _case("b", "in_domain", hit=0.0, rr=0.0), _case("c", "cross_doc", refused=True),
             _case("o1", "out_of_domain", refused=True), _case("o2", "out_of_domain", refused=False, parsed=False)]
    s = build_summary(cases, 5, {})
    assert s["retrieval_answerable_only"]["hit_rate@5"]["n"] == 3
    assert s["retrieval_answerable_only"]["hit_rate@5"]["mean"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["refusal_behaviour"]["out_of_domain_refused"] == "1/2"
    assert s["refusal_behaviour"]["false_refusals_on_answerable"] == "1/3"
    assert s["answer_quality"]["judge_parse_failures"].startswith("1/5")
    assert s["answer_quality"]["faithfulness"]["n"] == 4            # parse failure excluded, not defaulted
