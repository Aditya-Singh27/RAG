"""Live smoke test. Run on YOUR machine with real keys; prints model ids and one real call each.
    python scripts/check_providers.py
Nothing here is mocked: if it fails, the failure is the information."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import RagConfig
from common.embeddings import make_embedder
from common.llm import make_llm

if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("GEMINI_API_KEY not set")
from google import genai
c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
print("== models visible to your key (embedding + generateContent) ==")
for m in c.models.list():
    acts = getattr(m, "supported_actions", None) or []
    if "embedContent" in acts or "generateContent" in acts:
        print(f"{m.name:55s} {','.join(acts)}")

cfg = RagConfig.from_env()
if cfg.embed_model:
    e = make_embedder("gemini", cfg.embed_model, cfg.embed_dim)
    v = e.embed_query("hello world")
    print(f"\nembed OK: model={e.name} requested_dim={cfg.embed_dim} got_dim={len(v)}")
if cfg.gen_model:
    r = make_llm("gemini", cfg.gen_model).generate("Reply with the single word: pong", max_tokens=16)
    print(f"generate OK: {r.text!r} in={r.input_tokens} out={r.output_tokens} {r.latency_ms:.0f}ms")
