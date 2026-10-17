"""Environment-driven configuration. No secrets in code; names live in .env.example."""
from __future__ import annotations
import os
from dataclasses import dataclass


def _load_dotenv(path: str = ".env") -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    # Strip inline comments if any
                    if "#" in v:
                        # only strip if not inside quotes
                        v = v.split("#")[0]
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ and v:
                        os.environ[k] = v

_load_dotenv()


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
        _load_dotenv()
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
            min_score=float(_env("MIN_SCORE", "0.60")),
        )


@dataclass(frozen=True)
class JudgeConfig:
    judge_provider: str
    judge_model: str
    generator_provider: str
    generator_model: str
    generator_base_url: str
    generator_api_key: str
    log_path: str

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        _load_dotenv()
        return cls(
            judge_provider=_env("JUDGE_PROVIDER", "gemini"),
            judge_model=_env("JUDGE_MODEL", "gemini-2.5-flash"),
            generator_provider=_env("GENERATOR_PROVIDER", "openai_compat"),
            generator_model=_env("GENERATOR_MODEL", "llama3.2:latest"),
            generator_base_url=_env("GENERATOR_BASE_URL", "http://localhost:11434/v1"),
            generator_api_key=_env("GENERATOR_API_KEY", "ollama"),
            log_path=_env("LLM_LOG_PATH", "./logs/llm_calls.jsonl"),
        )

