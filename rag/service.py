"""RAG retrieval and grounded answer service.
Handles top-k retrieval, score thresholding, strict chunk citation,
and the no-context fallback branch without hallucinating.
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, asdict
from typing import Any

from common.config import RagConfig
from common.embeddings import Embedder, make_embedder
from common.llm import LLMClient, make_llm
from rag.store import VectorStore

NO_CONTEXT_MESSAGE = (
    "I do not have enough relevant context in the provided documents to answer this question."
)

GROUNDED_SYSTEM_PROMPT = """You are a strictly grounded question-answering assistant.
Answer the user's question using ONLY the provided context excerpts below.
Rules:
1. Every factual statement in your answer must be supported by the provided excerpts.
2. Cite the exact chunk(s) supporting your statements using bracketed citations, e.g., [Chunk 1] or [Chunk 2].
3. If the context does not contain enough information to fully answer the question, state what is known from the context and clearly state what cannot be answered. Do not speculate or extrapolate beyond the text.
4. Keep the answer direct, concise, and factual.
"""


@dataclass
class RagResult:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagService:
    def __init__(
        self,
        config: RagConfig | None = None,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        generator: LLMClient | None = None,
    ):
        self.config = config or RagConfig.from_env()
        self.embedder = embedder or make_embedder(
            self.config.embed_provider, self.config.embed_model, self.config.embed_dim
        )
        self.store = store or VectorStore(
            self.config.lancedb_uri, self.embedder.name, self.embedder.dim
        )
        if generator:
            self.generator = generator
        else:
            base_url = getattr(self.config, "gen_base_url", "")
            api_key = getattr(self.config, "gen_api_key", "")
            self.generator = make_llm(
                self.config.gen_provider, self.config.gen_model, base_url, api_key
            )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        doc_type: str | None = None,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        k = k if k is not None else self.config.k
        qvec = self.embedder.embed_query(query)
        return self.store.search(qvec, k=k, doc_type=doc_type, doc_id=doc_id)

    def query(
        self,
        query: str,
        k: int | None = None,
        doc_type: str | None = None,
        doc_id: str | None = None,
        min_score: float | None = None,
    ) -> RagResult:
        t_start = time.perf_counter()
        k = k if k is not None else self.config.k
        min_score = min_score if min_score is not None else self.config.min_score

        # 1. Retrieval
        t_ret_start = time.perf_counter()
        hits = self.retrieve(query, k=k, doc_type=doc_type, doc_id=doc_id)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        # Filter by min_score
        valid_chunks = [h for h in hits if h.get("score", 0.0) >= min_score]

        # 2. No-context branch: If no chunks retrieved or top score below threshold
        if not valid_chunks:
            total_ms = (time.perf_counter() - t_start) * 1000
            return RagResult(
                query=query,
                answer=NO_CONTEXT_MESSAGE,
                citations=[],
                retrieved_chunks=hits,
                chunks_used=[],
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_ms=total_ms,
                input_tokens=0,
                output_tokens=0,
                model=self.generator.model,
                no_context=True,
            )

        # 3. Context formatting
        context_blocks: list[str] = []
        for i, chunk in enumerate(valid_chunks, 1):
            source = chunk.get("source", "unknown")
            page = chunk.get("page")
            page_info = f", Page {page}" if page else ""
            header = f"[Chunk {i}] (Source: {source}{page_info}, Score: {chunk.get('score', 0):.3f})"
            context_blocks.append(f"{header}\n{chunk.get('text', '')}")

        formatted_context = "\n\n".join(context_blocks)
        user_prompt = f"""Context Information:
{formatted_context}

Question: {query}

Provide a grounded answer with citations to the chunk numbers ([Chunk X]):"""

        # 4. Generation
        t_gen_start = time.perf_counter()
        llm_res = self.generator.generate(
            prompt=user_prompt,
            system=GROUNDED_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
        )
        generation_ms = (time.perf_counter() - t_gen_start) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        # 5. Extract citations [Chunk X]
        raw_citations = re.findall(r"\[Chunk\s+(\d+)\]", llm_res.text)
        cited_indices = sorted(set(int(c) for c in raw_citations))
        
        chunks_used = []
        citations = []
        for idx in cited_indices:
            if 1 <= idx <= len(valid_chunks):
                chunk = valid_chunks[idx - 1]
                chunks_used.append(chunk)
                source = chunk.get("source", "unknown")
                page = chunk.get("page")
                page_str = f":p{page}" if page else ""
                citations.append(f"{source}{page_str}#chunk{chunk.get('chunk_index', idx-1)}")

        # If LLM cited no specific chunks but gave an answer, include all valid chunks presented
        if not chunks_used and llm_res.text.strip():
            chunks_used = valid_chunks

        return RagResult(
            query=query,
            answer=llm_res.text.strip(),
            citations=citations,
            retrieved_chunks=hits,
            chunks_used=chunks_used,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            input_tokens=llm_res.input_tokens or 0,
            output_tokens=llm_res.output_tokens or 0,
            model=self.generator.model,
            no_context=False,
        )


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Query the RAG Service")
    parser.add_argument("query", help="User question")
    parser.add_argument("--k", type=int, default=None, help="Top k chunks to retrieve")
    parser.add_argument("--doc-type", default=None, help="Filter by doc_type (pdf, html, md)")
    parser.add_argument("--min-score", type=float, default=None, help="Cosine similarity threshold")
    args = parser.parse_args()

    service = RagService()
    res = service.query(
        query=args.query,
        k=args.k,
        doc_type=args.doc_type,
        min_score=args.min_score,
    )
    print(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    main()
