import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import RagConfig, JudgeConfig
from common.embeddings import make_embedder
from common.llm import make_llm
from rag.store import VectorStore
from rag.service import GROUNDED_SYSTEM_PROMPT as RAG_SYSTEM_PROMPT

cfg = RagConfig.from_env()
jcfg = JudgeConfig.from_env()

# Use generator configured for Problem 2 (Llama 3.2 via Ollama)
gen = make_llm(jcfg.generator_provider, jcfg.generator_model, jcfg.generator_base_url, jcfg.generator_api_key)
emb = make_embedder(cfg.embed_provider, cfg.embed_model, cfg.embed_dim)
store = VectorStore(cfg.lancedb_uri, emb.name, emb.dim)

ab_file = Path("data/judge_suites/ab_comparison_cases.json")
cases = json.load(open(ab_file, encoding="utf-8"))

print(f"Generating REAL model outputs for {len(cases)} A/B comparison cases using {jcfg.generator_model}...")

NAIVE_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question directly based on the context provided."

for i, c in enumerate(cases, 1):
    q = c["input"]
    # Retrieve real chunks
    q_vec = emb.embed_query(q)
    results = store.search(q_vec, k=cfg.k)
    
    # Check similarity
    top_score = results[0]["score"] if results else 0.0
    
    # Context representations
    raw_ctx = "\n\n".join(r["text"] for r in results)
    numbered_ctx = "\n\n".join(f"[Chunk {idx}] {r['text']}" for idx, r in enumerate(results, 1))
    
    # Prompt v1: Naive ungrounded / uncited
    p1 = f"Question: {q}\n\nContext:\n{raw_ctx}"
    r1 = gen.generate(prompt=p1, system=NAIVE_SYSTEM_PROMPT, temperature=0.0, max_tokens=350)
    out_a = r1.text.strip()
    
    # Prompt v2: Grounded RAG with citations and guardrails
    if top_score < cfg.min_score:
        out_b = "I do not have enough relevant context in the provided documents to answer this question."
    else:
        p2 = f"User Question: {q}\n\nRetrieved Context:\n{numbered_ctx}"
        r2 = gen.generate(prompt=p2, system=RAG_SYSTEM_PROMPT, temperature=0.0, max_tokens=350)
        out_b = r2.text.strip()
    
    c["output_a"] = out_a
    c["output_b"] = out_b
    c["source_generator"] = jcfg.generator_model
    c["prompt_a_type"] = "naive_uncited_v1"
    c["prompt_b_type"] = "grounded_cited_v2"
    
    print(f"[{i}/{len(cases)}] {c['id']}: '{q[:40]}...'", flush=True)
    print(f"  Output A (Prompt v1): {out_a[:75]}...", flush=True)
    print(f"  Output B (Prompt v2): {out_b[:75]}...", flush=True)

with open(ab_file, "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=2)

print("\nSuccessfully updated data/judge_suites/ab_comparison_cases.json with REAL model outputs!", flush=True)
