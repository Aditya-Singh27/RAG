"""Latency benchmarks.

  python -m eval.bench_latency real  --questions data/eval_questions.json     # YOUR store: embed vs search, separately
  python -m eval.bench_latency synth --sizes 100000 1000000 --dim 768         # SYNTHETIC vectors: flat vs IVF_PQ

`synth` numbers are synthetic (clustered Gaussian vectors, random queries): they say how latency and index recall scale on
THIS machine, not how your embeddings behave. Label them so in the report. Output: results/latency_benchmark.json
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import percentile


def _stats(ms):
    return {"n": len(ms), "p50_ms": round(percentile(ms, 50), 3), "p95_ms": round(percentile(ms, 95), 3),
            "mean_ms": round(float(np.mean(ms)), 3)}


def _hw():
    return {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version()}


def bench_real(questions_path: str, repeats: int = 3, k: int | None = None):
    from common.config import RagConfig
    from common.embeddings import make_embedder
    from rag.store import VectorStore
    cfg = RagConfig.from_env()
    emb = make_embedder(cfg.embed_provider, cfg.embed_model, cfg.embed_dim)
    store = VectorStore(cfg.lancedb_uri, emb.name, emb.dim)
    qs = [q["query"] for q in json.load(open(questions_path, encoding="utf-8"))]
    t_embed, t_search = [], []
    store.search(emb.embed_query(qs[0]), k=k or cfg.k)                   # warm-up, not recorded
    for _ in range(repeats):
        for q in qs:
            t0 = time.perf_counter(); v = emb.embed_query(q); t1 = time.perf_counter()
            store.search(v, k=k or cfg.k); t2 = time.perf_counter()
            t_embed.append((t1 - t0) * 1e3); t_search.append((t2 - t1) * 1e3)
    return {"mode": "real", "vectors_in_store": store.count(), "k": k or cfg.k, "queries": len(qs), "repeats": repeats,
            "query_embedding_api": _stats(t_embed), "vector_search_only": _stats(t_search), "hardware": _hw()}


def _batches(n, dim, centers, seed, batch=50_000):
    rng = np.random.default_rng(seed)
    import pyarrow as pa
    done = 0
    while done < n:
        m = min(batch, n - done)
        x = centers[rng.integers(0, len(centers), m)] + 0.35 * rng.standard_normal((m, dim)).astype("float32")
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        vec = pa.FixedSizeListArray.from_arrays(pa.array(x.ravel(), type=pa.float32()), dim)
        yield pa.RecordBatch.from_arrays([pa.array(np.arange(done, done + m, dtype="int64")), vec], names=["id", "vector"])
        done += m


def _dir_bytes(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def bench_synth(sizes, dim=768, k=5, n_queries=100, seed=0, nprobes=20):
    import lancedb
    import pyarrow as pa
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((256, dim)).astype("float32")
    out = []
    for n in sizes:
        tmp = tempfile.mkdtemp(prefix="lance_synth_")
        try:
            db = lancedb.connect(tmp)
            schema = pa.schema([("id", pa.int64()), ("vector", pa.list_(pa.float32(), dim))])
            t0 = time.perf_counter()
            tbl = db.create_table("t", data=_batches(n, dim, centers, seed + 1), schema=schema)
            build_s = time.perf_counter() - t0
            qs = centers[rng.integers(0, 256, n_queries)] + 0.35 * rng.standard_normal((n_queries, dim)).astype("float32")
            qs /= np.linalg.norm(qs, axis=1, keepdims=True)

            def run(use_nprobes=None, refine=None, exact=False):
                lat, ids = [], []
                tbl.search(qs[0].tolist()).metric("cosine").limit(k).to_list()          # warm-up
                for q in qs:
                    s = tbl.search(q.tolist()).metric("cosine").limit(k)
                    if exact:
                        s = s.bypass_vector_index()
                    if use_nprobes:
                        s = s.nprobes(use_nprobes)
                    if refine:
                        s = s.refine_factor(refine)
                    t = time.perf_counter(); r = s.to_list(); lat.append((time.perf_counter() - t) * 1e3)
                    ids.append([x["id"] for x in r])
                return lat, ids

            flat_lat, flat_ids = run()
            row = {"vectors": n, "dim": dim, "k": k, "disk_bytes": _dir_bytes(tmp), "bytes_per_vector_on_disk": round(_dir_bytes(tmp) / n, 1),
                   "raw_float32_bytes_per_vector": dim * 4, "build_seconds": round(build_s, 1), "flat_exact": _stats(flat_lat)}
            parts = max(1, int(n ** 0.5))
            sub = next(d for d in (dim // 8, dim // 4, dim // 16, dim // 2, 1) if d >= 1 and dim % d == 0)
            t0 = time.perf_counter()
            try:
                tbl.create_index(metric="cosine", index_type="IVF_PQ", num_partitions=parts, num_sub_vectors=sub, replace=True)
                row["index_build_seconds"] = round(time.perf_counter() - t0, 1)
                row["ivf_pq"] = {"num_partitions_requested": parts, "num_sub_vectors": sub, "nprobes": nprobes,
                                 "disk_bytes_with_index": _dir_bytes(tmp), "settings": {}}
                for name, kw in (("no_refine", {}), ("refine_factor_20", {"refine": 20})):
                    lat_i, ids_i = run(nprobes, **kw)
                    rec = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(flat_ids, ids_i)]))
                    row["ivf_pq"]["settings"][name] = {**_stats(lat_i), f"recall@{k}_vs_exact": round(rec, 4)}
            except Exception as e:                               # e.g. too few rows to train the index
                row["ivf_pq"] = {"error": f"{type(e).__name__}: {e}"}
            out.append(row)
            print(json.dumps(row), flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return {"mode": "synthetic", "note": "clustered Gaussian vectors; random queries from the same clusters", "hardware": _hw(), "runs": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("real"); r.add_argument("--questions", default="data/eval_questions.json"); r.add_argument("--repeats", type=int, default=3)
    s = sp.add_parser("synth"); s.add_argument("--sizes", type=int, nargs="+", default=[100_000]); s.add_argument("--dim", type=int, default=768)
    s.add_argument("--queries", type=int, default=100)
    a = ap.parse_args()
    res = bench_real(a.questions, a.repeats) if a.cmd == "real" else bench_synth(a.sizes, a.dim, n_queries=a.queries)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/latency_benchmark_{a.cmd}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:2000])
