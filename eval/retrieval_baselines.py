"""Is the retriever actually good, and does MIN_SCORE have any margin?

  python -m eval.retrieval_baselines --questions data/eval_questions.json

For every answerable question (chunk-level `evidence`, see eval/metrics.py) it scores three retrievers on the SAME chunks:
  random : analytic Hit@k + Monte-Carlo MRR/nDCG for a retriever that picks k chunks uniformly
  bm25   : lexical Okapi BM25 over chunk text (no stopword list, no stemming: deliberately plain)
  dense  : your real embedder + LanceDB search
It then takes the dense top-1 cosine similarity of answerable vs out-of-domain questions, reports the gap, sweeps thresholds,
and draws results/similarity_margin.png. With a tiny OOD set (the default 3 are far-off topics) the margin is flattering: add
near-domain negatives ("what does Pinecone charge per pod?" when the corpus never says) before trusting MIN_SCORE.
"""
from __future__ import annotations
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (bootstrap_ci, count_relevant_in_corpus, evidence_recall, hit_at_k, ndcg_at_k,
                          reciprocal_rank, relevance_flags)
from judge.stats import wilson_ci

ANSWERABLE = ("in_domain", "cross_doc")


# --------------------------------------------------------------------------- retrievers
def load_chunks(lancedb_uri: str) -> list[dict]:
    import lancedb
    rows = lancedb.connect(lancedb_uri).open_table("chunks").to_arrow().select(["chunk_id", "doc_id", "text"]).to_pylist()
    return sorted(rows, key=lambda r: (r["doc_id"], r["chunk_id"]))      # stable order => reproducible random baseline


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(_tok(t)) for t in texts]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        n = len(texts)
        df = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query: str) -> list[int]:
        q = _tok(query)
        scores = []
        for d, ln in zip(self.docs, self.len):
            s = 0.0
            for w in q:
                f = d.get(w, 0)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            scores.append(s)
        return sorted(range(len(scores)), key=lambda i: -scores[i])


def random_baseline(n_chunks: int, n_rel: int, k: int, trials: int = 2000, seed: int = 0) -> dict:
    """Exact Hit@k = 1 - C(N-r,k)/C(N,k); MRR and nDCG by Monte Carlo (relevant chunks = indices < n_rel)."""
    hit = 1 - math.comb(n_chunks - n_rel, k) / math.comb(n_chunks, k)
    rng = random.Random(seed)
    rr = nd = 0.0
    for _ in range(trials):
        flags = [int(i < n_rel) for i in rng.sample(range(n_chunks), k)]
        rr += reciprocal_rank(flags)
        nd += ndcg_at_k(flags, n_rel, k)
    return {"hit": hit, "reciprocal_rank": rr / trials, "ndcg": nd / trials}


def _score(texts, evidence, n_rel, k):
    flags = relevance_flags(texts, evidence)
    return {"hit": hit_at_k(flags, k), "reciprocal_rank": reciprocal_rank(flags, k),
            "ndcg": ndcg_at_k(flags, n_rel, k), "evidence_recall": evidence_recall(texts, evidence, k)}


# --------------------------------------------------------------------------- threshold analysis
def threshold_sweep(ans_scores: list[float], ood_scores: list[float], thresholds: list[float]) -> list[dict]:
    """A query is refused when its top-1 similarity < threshold."""
    out = []
    for t in thresholds:
        fr = sum(s < t for s in ans_scores)
        orr = sum(s < t for s in ood_scores)
        out.append({"threshold": round(t, 3), "false_refusals": f"{fr}/{len(ans_scores)}",
                    "false_refusal_rate": round(fr / len(ans_scores), 3) if ans_scores else None,
                    "ood_refused": f"{orr}/{len(ood_scores)}",
                    "ood_refusal_rate": round(orr / len(ood_scores), 3) if ood_scores else None})
    return out


def plot_margin(ans: list[float], ood: list[float], min_score: float, path: Path, gap: float | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"   # validated: blue/orange ΔE 24.7 CVD
    rng = random.Random(1)
    fig, ax = plt.subplots(figsize=(8, 2.9), dpi=160, facecolor=SURF)
    ax.set_facecolor(SURF)
    for y, vals, col in ((1, ans, BLUE), (0, ood, ORANGE)):
        ax.scatter(vals, [y + rng.uniform(-0.18, 0.18) for _ in vals], s=70, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.axvline(min_score, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.text(min_score, 1.52, f" MIN_SCORE = {min_score:g}", color=INK2, fontsize=8, va="center")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([f"Answerable\n(n={len(ans)})", f"Out-of-domain\n(n={len(ood)})"], color=INK, fontsize=9)
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlabel("Top-1 cosine similarity of the retrieved chunk", color=INK2, fontsize=9)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e3e2de", linewidth=0.8, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    sub = "no overlap" if gap is not None and gap > 0 else "distributions overlap"
    if gap is not None:
        sub += f" (lowest answerable minus highest out-of-domain = {gap:+.3f})"
    fig.text(0.012, 0.955, "Does the refusal threshold have any margin?", color=INK, fontsize=11, va="top", fontweight="bold")
    fig.text(0.012, 0.865, sub, color=INK2, fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


# --------------------------------------------------------------------------- driver
def _agg(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    h = [r["hit"] for r in rows]
    lo, hi = wilson_ci(int(round(sum(h))), len(h)) if all(x in (0, 1, 0.0, 1.0) for x in h) else (float("nan"),) * 2
    out["hit"] = {"mean": round(sum(h) / len(h), 4), "ci95": [round(lo, 3), round(hi, 3)] if lo == lo else None}
    for key in ("reciprocal_rank", "ndcg"):
        m, a, b = bootstrap_ci([r[key] for r in rows])
        out[key] = {"mean": round(m, 4), "ci95": [round(a, 3), round(b, 3)]}
    if "evidence_recall" in rows[0]:
        out["evidence_recall"] = round(sum(r["evidence_recall"] for r in rows) / len(rows), 4)
    return out


def run(lancedb_uri: str, questions: list[dict], embedder, store, k: int, min_score: float, out_dir: str = "results") -> dict:
    chunks = load_chunks(lancedb_uri)
    texts = [c["text"] for c in chunks]
    N = len(texts)
    bm25 = BM25(texts)
    ans_q = [q for q in questions if q["category"] in ANSWERABLE]
    ood_q = [q for q in questions if q["category"] == "out_of_domain"]
    per = {"random": [], "bm25": [], "dense": []}
    rr_pairs, top1_ans, top1_ood = [], [], []
    for q in ans_q:
        ev = q["evidence"]
        n_rel = count_relevant_in_corpus(texts, ev)
        if n_rel == 0:
            raise SystemExit(f"{q['id']}: evidence not found in any chunk")
        per["random"].append(random_baseline(N, n_rel, k))
        bm = _score([texts[i] for i in bm25.rank(q["query"])[:k]], ev, n_rel, k)
        hits = store.search(embedder.embed_query(q["query"]), k=k)
        de = _score([h["text"] for h in hits], ev, n_rel, k)
        per["bm25"].append(bm)
        per["dense"].append(de)
        rr_pairs.append((de["reciprocal_rank"], bm["reciprocal_rank"]))
        top1_ans.append(hits[0]["score"] if hits else 0.0)
    for q in ood_q:
        hits = store.search(embedder.embed_query(q["query"]), k=1)
        top1_ood.append(hits[0]["score"] if hits else 0.0)

    gap = (min(top1_ans) - max(top1_ood)) if top1_ans and top1_ood else None
    ths = sorted({round(0.30 + 0.05 * i, 2) for i in range(13)} | {round(min_score, 3)})
    res = {
        "k": k, "n_chunks": N, "n_answerable": len(ans_q), "n_out_of_domain": len(ood_q),
        "random_hit_at_k_if_one_relevant_chunk": round(k / N, 4),
        "retrievers": {name: _agg(rows) for name, rows in per.items() if rows},
        "dense_vs_bm25_reciprocal_rank": {"dense_better": sum(a > b for a, b in rr_pairs),
                                          "tie": sum(a == b for a, b in rr_pairs), "bm25_better": sum(a < b for a, b in rr_pairs)},
        "similarity_margin": {
            "top1_answerable": {"min": round(min(top1_ans), 4), "max": round(max(top1_ans), 4)} if top1_ans else None,
            "top1_out_of_domain": {"min": round(min(top1_ood), 4), "max": round(max(top1_ood), 4)} if top1_ood else None,
            "gap_min_answerable_minus_max_ood": None if gap is None else round(gap, 4),
            "configured_min_score": min_score,
            "at_configured_threshold": threshold_sweep(top1_ans, top1_ood, [min_score])[0],
            "sweep": threshold_sweep(top1_ans, top1_ood, ths),
            "per_question_top1": {**{q["id"]: round(s, 4) for q, s in zip(ans_q, top1_ans)},
                                  **{q["id"]: round(s, 4) for q, s in zip(ood_q, top1_ood)}}},
        "caveats": ([f"only {len(ood_q)} out-of-domain questions: too few, and far-off topics are easy negatives; add >=15 incl. near-domain ones"]
                    if len(ood_q) < 15 else []) +
                   ([f"corpus has {N} chunks: random retrieval already hits {k / N:.0%} of the time at k={k}"] if N < 200 else []),
    }
    Path(out_dir).mkdir(exist_ok=True)
    (Path(out_dir) / "retrieval_baselines.json").write_text(json.dumps(res, indent=2))
    if top1_ans and top1_ood:
        plot_margin(top1_ans, top1_ood, min_score, Path(out_dir) / "similarity_margin.png", gap)
    return res


def main():
    import argparse
    from common.config import RagConfig
    from common.embeddings import make_embedder
    from rag.store import VectorStore
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval_questions.json")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = RagConfig.from_env()
    emb = make_embedder(cfg.embed_provider, cfg.embed_model, cfg.embed_dim)
    store = VectorStore(cfg.lancedb_uri, emb.name, emb.dim)
    qs = json.load(open(a.questions, encoding="utf-8"))
    res = run(cfg.lancedb_uri, qs, emb, store, cfg.k, cfg.min_score, a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "similarity_margin"}, indent=2))
    sm = res["similarity_margin"]
    print(json.dumps({k: v for k, v in sm.items() if k not in ("per_question_top1",)}, indent=2))


if __name__ == "__main__":
    main()
