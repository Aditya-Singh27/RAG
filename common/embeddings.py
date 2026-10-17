"""Embedder interface. HashEmbedder is deterministic and offline: for tests only."""
from __future__ import annotations
import hashlib
import math
import os
import re
from typing import Protocol

import numpy as np
from common.config import _load_dotenv
_load_dotenv()


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


class HashEmbedder:
    """Bag-of-words feature hashing. Shared vocabulary => higher cosine, so retrieval
    tests are meaningful, but it is NOT a semantic model. Never use it for reported results."""

    def __init__(self, dim: int = 256):
        self.name = "hash-bow"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return _normalize(v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class GeminiEmbedder:
    """Gemini embeddings via google-genai. Truncated dimensions are re-normalised
    (Matryoshka-style truncation does not keep unit norm)."""

    BATCH = 50

    def __init__(self, model: str, dim: int, api_key: str | None = None):
        from google import genai
        if not model:
            raise ValueError("EMBED_MODEL is not set")
        self._types = __import__("google.genai.types", fromlist=["types"])
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.name, self.dim = model, dim

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            res = self.client.models.embed_content(
                model=self.name,
                contents=texts[i : i + self.BATCH],
                config=self._types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dim
                ),
            )
            out += [_normalize(np.asarray(e.values, dtype=np.float32)).tolist() for e in res.embeddings]
        return out

    def embed_documents(self, texts):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def make_embedder(provider: str, model: str, dim: int) -> Embedder:
    if provider == "hash":
        return HashEmbedder(dim)
    if provider == "gemini":
        return GeminiEmbedder(model, dim)
    raise ValueError(f"unknown EMBED_PROVIDER {provider!r}")
