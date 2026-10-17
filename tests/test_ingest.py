import pytest
from common.embeddings import HashEmbedder
from rag.ingest import ingest_path
from rag.store import VectorStore


@pytest.fixture
def corpus(tmp_path):
    c = tmp_path / "corpus"; c.mkdir()
    (c / "a.md").write_text("# Cats\n\n" + "Cats are small carnivorous mammals that purr. " * 30)
    (c / "b.html").write_text("<html><body><nav>menu</nav><p>" + "Rockets burn propellant to produce thrust. " * 30 + "</p></body></html>")
    return c


@pytest.fixture
def env(tmp_path):
    emb = HashEmbedder(128)
    return emb, VectorStore(str(tmp_path / "db"), emb.name, emb.dim)


def test_reingest_is_idempotent(corpus, env):
    emb, store = env
    r1 = ingest_path(corpus, store, emb, 300, 50)
    n = store.count()
    assert n > 2 and r1["docs_ingested"] == 2
    r2 = ingest_path(corpus, store, emb, 300, 50)
    assert store.count() == n
    assert r2["docs_skipped_unchanged"] == 2 and r2["chunks_upserted"] == 0


def test_forced_reupsert_does_not_duplicate(corpus, env):
    emb, store = env
    ingest_path(corpus, store, emb, 300, 50)
    n = store.count()
    rows = store.tbl.search().limit(1000).to_list()
    store.upsert(rows)                       # same chunk_ids again
    assert store.count() == n


def test_changed_doc_replaces_old_chunks(corpus, env):
    emb, store = env
    ingest_path(corpus, store, emb, 300, 50)
    (corpus / "a.md").write_text("Dogs are loyal. " * 20)
    ingest_path(corpus, store, emb, 300, 50)
    texts = " ".join(r["text"] for r in store.tbl.search().limit(1000).to_list() if r["doc_id"] == "a.md")
    assert "Dogs" in texts and "Cats" not in texts


def test_changed_chunk_params_reingest_without_dupes(corpus, env):
    emb, store = env
    ingest_path(corpus, store, emb, 300, 50)
    ingest_path(corpus, store, emb, 500, 50)
    ids = [r["chunk_id"] for r in store.tbl.search().limit(1000).to_list()]
    assert len(ids) == len(set(ids))
    assert {r["doc_hash"] for r in store.tbl.search().limit(1000).to_list() if r["doc_id"] == "a.md"}.__len__() == 1


def test_metadata_filter(corpus, env):
    emb, store = env
    ingest_path(corpus, store, emb, 300, 50)
    hits = store.search(emb.embed_query("cats purr"), k=5, doc_type="html")
    assert hits and all(h["doc_type"] == "html" for h in hits)
    top = store.search(emb.embed_query("cats purr"), k=1)[0]
    assert top["doc_id"] == "a.md"


def test_embedding_space_mismatch_refused(tmp_path):
    VectorStore(str(tmp_path / "db"), "m1", 64)
    with pytest.raises(ValueError):
        VectorStore(str(tmp_path / "db"), "m2", 64)
