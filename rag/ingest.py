"""Idempotent ingestion.

Guarantees (see tests/test_ingest.py):
  * chunk_id = sha256(doc_id | chunk_index | text)  -> same input => same key => merge_insert updates in place
  * doc_hash covers text + chunk params + embed model -> unchanged docs are skipped without re-embedding
  * on change, new rows are upserted first, then rows of that doc with an older doc_hash are deleted,
    so a crash mid-way leaves duplicates-by-old-hash at worst, never missing data.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

from rag.chunking import split_text
from rag.loaders import iter_documents


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def ingest_path(path: str | Path, store, embedder, size: int = 800, overlap: int = 100) -> dict:
    path = Path(path)
    rep = {"docs_seen": 0, "docs_skipped_unchanged": 0, "docs_ingested": 0, "chunks_upserted": 0}
    for doc in iter_documents(path):
        rep["docs_seen"] += 1
        dh = _sha(doc.text, str(size), str(overlap), embedder.name, str(embedder.dim))
        if store.doc_hash(doc.doc_id) == dh:
            rep["docs_skipped_unchanged"] += 1
            continue
        pieces: list[tuple[int | None, str]] = []
        for page, text in doc.sections:
            pieces += [(page, c) for c in split_text(text, size, overlap)]
        if not pieces:
            continue
        vecs = embedder.embed_documents([c for _, c in pieces])
        rows = [{
            "chunk_id": _sha(doc.doc_id, str(i), c)[:32], "doc_id": doc.doc_id,
            "source": doc.source, "doc_type": doc.doc_type, "page": page if page else 0,
            "chunk_index": i, "doc_hash": dh, "text": c, "vector": v,
        } for i, ((page, c), v) in enumerate(zip(pieces, vecs))]
        store.upsert(rows)
        store.delete_stale(doc.doc_id, dh)
        rep["docs_ingested"] += 1
        rep["chunks_upserted"] += len(rows)
    rep["vectors_in_store"] = store.count()
    return rep


def main():
    import argparse, json
    from common.config import RagConfig
    from common.embeddings import make_embedder
    from rag.store import VectorStore
    cfg = RagConfig.from_env()
    ap = argparse.ArgumentParser(description="Ingest a folder/file of PDF/HTML/MD")
    ap.add_argument("path")
    ap.add_argument("--size", type=int, default=cfg.chunk_size)
    ap.add_argument("--overlap", type=int, default=cfg.chunk_overlap)
    a = ap.parse_args()
    emb = make_embedder(cfg.embed_provider, cfg.embed_model, cfg.embed_dim)
    store = VectorStore(cfg.lancedb_uri, emb.name, emb.dim)
    print(json.dumps(ingest_path(a.path, store, emb, a.size, a.overlap), indent=2))


if __name__ == "__main__":
    main()
