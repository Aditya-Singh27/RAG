"""FastAPI HTTP service for RAG QA and Ingestion.
Logs per-query latency, chunk count, and token usage.
"""
from __future__ import annotations
import logging
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.config import RagConfig
from rag.ingest import ingest_path
from rag.service import RagService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag.api")

app = FastAPI(
    title="Cost-Efficient RAG API",
    description="QA service over document corpus backed by LanceDB vector store",
    version="1.0.0",
)

_service: RagService | None = None


def get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question", min_length=1)
    k: int | None = Field(default=None, description="Number of chunks to retrieve")
    doc_type: str | None = Field(default=None, description="Optional metadata filter: pdf, html, md")
    doc_id: str | None = Field(default=None, description="Optional metadata filter for specific document")
    min_score: float | None = Field(default=None, description="Cosine similarity score threshold")


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    doc_type: str
    page: int
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[str]
    retrieved_chunks: list[dict[str, Any]]
    chunks_used: list[dict[str, Any]]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    input_tokens: int
    output_tokens: int
    model: str
    no_context: bool


class IngestRequest(BaseModel):
    path: str = Field(..., description="Path to folder or file to ingest")
    chunk_size: int | None = Field(default=None, description="Chunk size in characters")
    chunk_overlap: int | None = Field(default=None, description="Chunk overlap in characters")


class IngestResponse(BaseModel):
    docs_seen: int
    docs_skipped_unchanged: int
    docs_ingested: int
    chunks_upserted: int
    vectors_in_store: int


@app.get("/health")
def health_check():
    svc = get_service()
    try:
        count = svc.store.count()
        return {
            "status": "healthy",
            "store_ok": True,
            "vectors_in_store": count,
            "embed_model": svc.embedder.name,
            "generator_model": svc.generator.model,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats")
def stats():
    svc = get_service()
    return {
        "vectors_in_store": svc.store.count(),
        "embed_provider": svc.config.embed_provider,
        "embed_model": svc.embedder.name,
        "embed_dim": svc.embedder.dim,
        "generator_provider": svc.config.gen_provider,
        "generator_model": svc.generator.model,
        "lancedb_uri": svc.config.lancedb_uri,
        "default_k": svc.config.k,
        "min_score": svc.config.min_score,
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    svc = get_service()
    try:
        result = svc.query(
            query=req.query,
            k=req.k,
            doc_type=req.doc_type,
            doc_id=req.doc_id,
            min_score=req.min_score,
        )
        logger.info(
            f"Query: {req.query!r} | "
            f"Retrieved: {len(result.retrieved_chunks)} | "
            f"Used: {len(result.chunks_used)} | "
            f"Tokens: in={result.input_tokens}, out={result.output_tokens} | "
            f"Latency: ret={result.retrieval_ms:.1f}ms, gen={result.generation_ms:.1f}ms, total={result.total_ms:.1f}ms"
        )
        return QueryResponse(**result.to_dict())
    except Exception as exc:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(req: IngestRequest):
    svc = get_service()
    target_path = Path(req.path)
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    
    cfg = svc.config
    size = req.chunk_size or cfg.chunk_size
    overlap = req.chunk_overlap or cfg.chunk_overlap

    logger.info(f"Ingesting documents from {req.path} (size={size}, overlap={overlap})")
    try:
        rep = ingest_path(target_path, svc.store, svc.embedder, size=size, overlap=overlap)
        logger.info(f"Ingestion result: {rep}")
        return IngestResponse(**rep)
    except Exception as exc:
        logger.exception("Error ingesting documents")
        raise HTTPException(status_code=500, detail=str(exc))
