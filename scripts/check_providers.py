"""Live smoke test. Run on YOUR machine with real keys; prints model ids and one real call each.
    python scripts/check_providers.py
Nothing here is mocked: if it fails, the failure is the information."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import RagConfig, JudgeConfig
from common.embeddings import make_embedder
from common.llm import make_llm

print("=== Provider & Environment Check ===")
cfg = RagConfig.from_env()
jcfg = JudgeConfig.from_env()

gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not gemini_key:
    print("[!] GEMINI_API_KEY is not set in .env or environment.")
    print("    Please open .env and set your GEMINI_API_KEY.")
else:
    print("[+] GEMINI_API_KEY is present.")
    try:
        from google import genai
        c = genai.Client(api_key=gemini_key)
        print("\n== Gemini models visible to your key (embedding + generateContent) ==")
        for m in c.models.list():
            acts = getattr(m, "supported_actions", None) or []
            if "embedContent" in acts or "generateContent" in acts:
                print(f"  {m.name:50s} {','.join(acts)}")
        
        if cfg.embed_model:
            e = make_embedder("gemini", cfg.embed_model, cfg.embed_dim)
            v = e.embed_query("hello world")
            print(f"\n[+] Gemini embed OK: model={e.name} requested_dim={cfg.embed_dim} got_dim={len(v)}")
        if cfg.gen_model:
            r = make_llm("gemini", cfg.gen_model).generate("Reply with the single word: pong", max_tokens=16)
            print(f"[+] Gemini generate OK: {r.text.strip()!r} in={r.input_tokens} out={r.output_tokens} latency={r.latency_ms:.0f}ms")
    except Exception as exc:
        print(f"[-] Gemini test error: {exc}")

print("\n== Checking Generator (Problem 2 separate model family) ==")
print(f"Provider: {jcfg.generator_provider}, Model: {jcfg.generator_model}, URL: {jcfg.generator_base_url}")
try:
    gen = make_llm(jcfg.generator_provider, jcfg.generator_model, jcfg.generator_base_url, jcfg.generator_api_key)
    res = gen.generate("Reply with the single word: pong", max_tokens=16)
    print(f"[+] Generator OK: {res.text.strip()!r} in={res.input_tokens} out={res.output_tokens} latency={res.latency_ms:.0f}ms")
except Exception as exc:
    print(f"[-] Generator error: {exc}")

