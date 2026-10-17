import pytest
from fastapi.testclient import TestClient

from common.embeddings import HashEmbedder
from common.llm import ScriptedLLM
from rag.api import app, get_service
from rag.service import RagService, NO_CONTEXT_MESSAGE
from rag.store import VectorStore
from rag.ingest import ingest_path


@pytest.fixture
def rag_setup(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.md").write_text("# LanceDB\n\nLanceDB is a serverless vector database built on the Lance columnar format.")
    (corpus_dir / "doc2.html").write_text("<html><body><p>Python asyncio provides event loop and coroutines for async I/O.</p></body></html>")

    emb = HashEmbedder(dim=128)
    store = VectorStore(str(tmp_path / "db"), emb.name, emb.dim)
    ingest_path(corpus_dir, store, emb, size=300, overlap=50)

    llm = ScriptedLLM(
        responses=[
            "LanceDB is built on the Lance format [Chunk 1].",
            "Asyncio uses an event loop [Chunk 1].",
        ]
    )

    svc = RagService(
        store=store,
        embedder=emb,
        generator=llm,
    )
    return svc, store, emb, llm, corpus_dir


def test_rag_service_grounded_answer(rag_setup):
    svc, store, emb, llm, _ = rag_setup
    result = svc.query("What is LanceDB?", k=3, min_score=-1.0)
    assert not result.no_context
    assert "LanceDB is built on the Lance format" in result.answer
    assert len(result.citations) >= 1
    assert any("doc1.md" in c for c in result.citations)
    assert result.retrieval_ms > 0
    assert result.total_ms > 0


def test_rag_service_no_context_fallback(rag_setup):
    svc, store, emb, llm, _ = rag_setup
    calls_before = len(llm.calls)
    # Require an impossible cosine similarity
    result = svc.query("Completely irrelevant query", k=3, min_score=2.0)
    assert result.no_context
    assert result.answer == NO_CONTEXT_MESSAGE
    assert result.citations == []
    assert result.chunks_used == []
    assert len(llm.calls) == calls_before  # LLM was never called!


def test_rag_service_metadata_filter(rag_setup):
    svc, store, emb, llm, _ = rag_setup
    hits_html = svc.retrieve("LanceDB", k=5, doc_type="html")
    assert all(h["doc_type"] == "html" for h in hits_html)
    hits_md = svc.retrieve("LanceDB", k=5, doc_type="md")
    assert all(h["doc_type"] == "md" for h in hits_md)


def test_fastapi_endpoints(rag_setup, monkeypatch):
    svc, _, _, _, corpus_dir = rag_setup
    # Override app get_service to use test fixture
    import rag.api
    rag.api._service = svc

    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "healthy"
    assert data["vectors_in_store"] > 0

    # Stats
    res_stats = client.get("/stats")
    assert res_stats.status_code == 200
    assert "vectors_in_store" in res_stats.json()

    # Query
    res_query = client.post("/query", json={"query": "What is LanceDB?", "k": 2, "min_score": -1.0})
    assert res_query.status_code == 200
    q_data = res_query.json()
    assert "answer" in q_data
    assert len(q_data["retrieved_chunks"]) <= 2

    # Ingest with same params should skip unchanged docs
    res_ingest = client.post("/ingest", json={"path": str(corpus_dir), "chunk_size": 300, "chunk_overlap": 50})
    assert res_ingest.status_code == 200
    i_data = res_ingest.json()
    assert i_data["docs_seen"] == 2
    assert i_data["docs_skipped_unchanged"] == 2

