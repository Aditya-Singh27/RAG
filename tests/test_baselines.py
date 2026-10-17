import json
import math
import pytest
from common.embeddings import HashEmbedder
from eval.retrieval_baselines import BM25, random_baseline, threshold_sweep, run
from rag.ingest import ingest_path
from rag.store import VectorStore


def test_random_baseline_matches_closed_form():
    N, k = 28, 5
    r = random_baseline(N, 1, k, trials=20000, seed=3)
    assert r["hit"] == pytest.approx(5 / 28)
    assert r["reciprocal_rank"] == pytest.approx(sum(1 / i for i in range(1, k + 1)) / N, abs=0.01)
    assert 0 <= r["ndcg"] <= 1
    assert random_baseline(10, 10, 3)["hit"] == 1.0


def test_bm25_ranks_matching_chunk_first():
    texts = ["cats purr softly on the sofa", "rockets burn propellant for thrust", "the sofa is blue"]
    assert BM25(texts).rank("how do rockets get thrust")[0] == 1
    assert BM25(texts).rank("blue sofa")[0] == 2


def test_threshold_sweep_counts():
    s = threshold_sweep([0.9, 0.8, 0.55], [0.3, 0.62], [0.6])[0]
    assert s["false_refusals"] == "1/3" and s["ood_refused"] == "1/2"


def test_end_to_end_small_corpus(tmp_path):
    corpus = tmp_path / "c"; corpus.mkdir()
    (corpus / "cats.md").write_text("Cats are small carnivorous mammals. " * 3 + "\n\nCats purr when content and groom themselves daily.")
    (corpus / "space.md").write_text("Rockets burn propellant to produce thrust. " * 3 + "\n\nOrbital velocity near Earth is about 7.8 km per second.")
    emb = HashEmbedder(256)
    store = VectorStore(str(tmp_path / "db"), emb.name, emb.dim)
    ingest_path(corpus, store, emb, 120, 20)
    qs = [{"id": "a1", "category": "in_domain", "query": "what do cats do when content", "evidence": ["Cats purr when content"]},
          {"id": "a2", "category": "in_domain", "query": "orbital velocity near earth", "evidence": ["Orbital velocity near Earth is about 7.8"]},
          {"id": "o1", "category": "out_of_domain", "query": "zzzz qqqq xxxx unrelated gibberish"}]
    res = run(str(tmp_path / "db"), qs, emb, store, k=3, min_score=0.3, out_dir=str(tmp_path / "out"))
    assert res["n_answerable"] == 2 and res["n_out_of_domain"] == 1
    assert res["retrievers"]["dense"]["hit"]["mean"] == 1.0 and res["retrievers"]["bm25"]["hit"]["mean"] == 1.0
    assert res["retrievers"]["random"]["hit"]["mean"] < 1.0
    sm = res["similarity_margin"]
    assert sm["gap_min_answerable_minus_max_ood"] > 0
    assert (tmp_path / "out" / "similarity_margin.png").stat().st_size > 5000
    assert json.load(open(tmp_path / "out" / "retrieval_baselines.json"))["k"] == 3
    assert any("out-of-domain" in c for c in res["caveats"])


def test_missing_evidence_stops(tmp_path):
    corpus = tmp_path / "c"; corpus.mkdir(); (corpus / "a.md").write_text("alpha beta gamma delta " * 10)
    emb = HashEmbedder(64); store = VectorStore(str(tmp_path / "db"), emb.name, emb.dim); ingest_path(corpus, store, emb, 100, 10)
    with pytest.raises(SystemExit):
        run(str(tmp_path / "db"), [{"id": "x", "category": "in_domain", "query": "q", "evidence": ["not in corpus"]}], emb, store, 3, 0.3, str(tmp_path / "o"))
