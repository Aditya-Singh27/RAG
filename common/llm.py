"""Provider-agnostic text-generation client with an auditable JSONL call log.
Judge and generator are built independently (different provider/model/env vars)."""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Protocol

from common.config import _load_dotenv
_load_dotenv()


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


class LLMClient(Protocol):
    model: str

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 1024) -> LLMResponse: ...


class GeminiLLM:
    def __init__(self, model: str, api_key: str | None = None):
        from google import genai
        from google.genai import types
        if not model:
            raise ValueError("Gemini model id is not set")
        self._types = types
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model

    def generate(self, prompt, system="", temperature=0.0, max_tokens=1024):
        t0 = time.perf_counter()
        config_kwargs = {
            "system_instruction": system or None,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if max_tokens < 256 and hasattr(self._types, "ThinkingConfig"):
            try:
                config_kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        r = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=self._types.GenerateContentConfig(**config_kwargs),
        )
        ms = (time.perf_counter() - t0) * 1000
        u = getattr(r, "usage_metadata", None)
        text = r.text or ""
        if not text and getattr(r, "candidates", None) and len(r.candidates) > 0:
            for part in getattr(r.candidates[0].content, "parts", []):
                if getattr(part, "text", None):
                    text += part.text
        return LLMResponse(text, self.model,
                           getattr(u, "prompt_token_count", None),
                           getattr(u, "candidates_token_count", None), ms)



class OpenAICompatLLM:
    """Any OpenAI-compatible endpoint (Ollama, Groq, OpenRouter, vLLM...)."""

    def __init__(self, model: str, base_url: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, prompt, system="", temperature=0.0, max_tokens=1024):
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        ms = (time.perf_counter() - t0) * 1000
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", self.model,
                           getattr(u, "prompt_tokens", None),
                           getattr(u, "completion_tokens", None), ms)


class ScriptedLLM:
    """Test double: returns queued responses or calls fn(prompt, system)."""

    def __init__(self, responses=None, fn=None, model="scripted"):
        self.model, self._q, self._fn = model, list(responses or []), fn
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt, system="", temperature=0.0, max_tokens=1024):
        self.calls.append((system, prompt))
        text = self._fn(prompt, system) if self._fn else self._q.pop(0)
        return LLMResponse(text, self.model, len(prompt) // 4, len(text) // 4, 0.0)


class LoggedLLM:
    """Wraps a client; appends every prompt + raw response to a JSONL file and
    keeps running token/call totals (needed for judge cost tracking and replay)."""

    def __init__(self, inner: LLMClient, log_path: str, role: str):
        self.inner, self.log_path, self.role = inner, log_path, role
        self.model = inner.model
        self.calls = self.input_tokens = self.output_tokens = 0
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def generate(self, prompt, system="", temperature=0.0, max_tokens=1024, tag: str = ""):
        r = self.inner.generate(prompt, system, temperature, max_tokens)
        self.calls += 1
        self.input_tokens += r.input_tokens or 0
        self.output_tokens += r.output_tokens or 0
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "role": self.role, "tag": tag,
                                "temperature": temperature, "system": system,
                                "prompt": prompt, "response": asdict(r)}) + "\n")
        return r


def make_llm(provider: str, model: str, base_url: str = "", api_key: str = "") -> LLMClient:
    if provider == "gemini":
        return GeminiLLM(model)
    if provider == "openai_compat":
        return OpenAICompatLLM(model, base_url, api_key)
    raise ValueError(f"unknown provider {provider!r}")
