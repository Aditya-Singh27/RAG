"""Environment-driven configuration. No secrets in code; names live in .env.example."""
from __future__ import annotations
import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class RagConfig:
    embed_provider: str
    embed_model: str
    embed_dim: int
    gen_provider: str
    gen_model: str
    lancedb_uri: str
    chunk_size: int
    chunk_overlap: int
    k: int
    min_score: float

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            embed_provider=_env("EMBED_PROVIDER", "gemini"),
            embed_model=_env("EMBED_MODEL"),
            embed_dim=int(_env("EMBED_DIM", "768")),
            gen_provider=_env("RAG_GEN_PROVIDER", "gemini"),
            gen_model=_env("RAG_GEN_MODEL"),
            lancedb_uri=_env("LANCEDB_URI", "./lancedb_data"),
            chunk_size=int(_env("CHUNK_SIZE", "800")),
            chunk_overlap=int(_env("CHUNK_OVERLAP", "100")),
            k=int(_env("RETRIEVAL_K", "5")),
            min_score=float(_env("MIN_SCORE", "0.0")),
        )
