"""Architecture diagrams for the submission (matplotlib, no data in them - structure only).

Run: python scripts/create_diagrams.py   -> results/figures/problem{1,2}_architecture.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("results/figures")
BLUE, ORANGE, GREY, RED, INK = "#2166ac", "#d9730d", "#6b7280", "#b42318", "#111827"


def box(ax, x, y, w, h, title, sub="", edge=BLUE, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=2))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5, weight="bold", color=edge)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.35)


def arrow(ax, p, q, label="", color=GREY, rad=0.0, lx=0, ly=0.13):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                                   connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha="center", va="center", fontsize=7.8,
                color=color, bbox=dict(fc="white", ec="none", pad=1.2))


def canvas(title, subtitle, w=15, h=8.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(w / 2, h - 0.35, title, ha="center", fontsize=15, weight="bold", color=INK)
    ax.text(w / 2, h - 0.80, subtitle, ha="center", fontsize=9.5, color=GREY)
    return fig, ax


def problem1():
    fig, ax = canvas("Problem 1 - RAG service", "Ingestion path (top) and query path (bottom). The query can be refused at the score gate or by the generator itself.")
    bw, bh = 2.6, 1.25
    ax.text(0.2, 6.55, "INGESTION  (idempotent: skipped when doc_hash is unchanged)", fontsize=9, weight="bold", color=GREY)
    top = [("Loaders", "PDF / HTML / Markdown"), ("Chunker", "size + overlap\n(default 800 / 100)"),
           ("Embedder", "gemini-embedding-001\n768-d, document task type"), ("LanceDB table", "vectors + metadata\n(doc_id, source, page ...)")]
    xs = [0.4, 4.0, 7.6, 11.2]
    for (t, s), x in zip(top, xs):
        box(ax, x, 5.0, bw, bh, t, s, edge=BLUE if t != "LanceDB table" else ORANGE)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + bw, 5.62), (b, 5.62))
    ax.text(12.5, 6.85, "upsert by chunk_id, then delete stale rows\nwith an older doc_hash", ha="center", va="center", fontsize=7.6, color=GREY)

    ax.text(0.2, 3.85, "QUERY", fontsize=9, weight="bold", color=GREY)
    qy = 2.35
    box(ax, 0.4, qy, bw, bh, "Query", "HTTP /query or CLI\n(k, optional filter)")
    box(ax, 4.0, qy, bw, bh, "Embed query", "same model,\nquery task type")
    box(ax, 7.6, qy, bw, bh, "Top-k search", "cosine similarity\n+ metadata filter", edge=ORANGE)
    box(ax, 11.2, qy, bw, bh, "Score gate", "top-1 score >= MIN_SCORE ?", edge=ORANGE)
    for a_, b_ in [(0.4, 4.0), (4.0, 7.6), (7.6, 11.2)]:
        arrow(ax, (a_ + bw, qy + 0.62), (b_, qy + 0.62))
    arrow(ax, (11.6, 5.0), (9.2, qy + bh + 0.02), "reads the same table", color=ORANGE, lx=-1.6, ly=0.25)

    box(ax, 7.6, 0.45, bw, bh, "Generator", "grounded prompt, [Chunk N]\ncitations; may itself decline\nif the context lacks the answer", edge=BLUE)
    box(ax, 11.2, 0.45, bw, bh, "Refusal: no context", "no LLM call; returns\n'no relevant context'", edge=RED)
    arrow(ax, (11.4, qy), (9.9, 0.45 + bh), "yes", color=BLUE, lx=0.15, ly=0.2)
    arrow(ax, (12.5, qy), (12.5, 0.45 + bh), "no (below MIN_SCORE)", color=RED, lx=0.0, ly=0.0)
    box(ax, 0.4, 0.45, bw, bh, "Response + log", "answer, citations, chunk ids,\nlatency, tokens (JSONL)", edge=GREY)
    arrow(ax, (7.6, 1.07), (3.0, 1.07))
    ax.plot([12.5, 12.5, 1.7], [0.45, 0.12, 0.12], color=RED, lw=1.8)
    arrow(ax, (1.7, 0.12), (1.7, 0.45), color=RED)
    ax.text(6.6, 0.22, "refusal returned directly", fontsize=7.8, color=RED, ha="center", bbox=dict(fc="white", ec="none", pad=1))
    fig.savefig(OUT / "problem1_architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def problem2():
    fig, ax = canvas("Problem 2 - LLM-as-judge pipeline", "Judge and generator are configured independently. Pairwise cases run in both orders.")
    bw, bh = 2.7, 1.25
    box(ax, 0.4, 5.3, bw, bh, "Test suite", "JSON: input, output(s),\nexpected / reference")
    box(ax, 4.2, 5.3, bw, bh, "Prompt builder", "rubric + score anchors,\ncriterion order fixed")
    box(ax, 8.0, 5.3, bw, bh, "Judge model", "own provider / model /\nkey (JUDGE_*)", edge=ORANGE)
    box(ax, 11.8, 5.3, bw, bh, "Verdict parser", "strict JSON; markdown strip;\nretry; failures counted", edge=BLUE)
    for a, b in [(0.4, 4.2), (4.2, 8.0), (8.0, 11.8)]:
        arrow(ax, (a + bw, 5.92), (b, 5.92))

    box(ax, 4.2, 3.2, bw, bh, "Order swap (pairwise)", "run A,B then B,A;\nconsensus; flips recorded", edge=ORANGE)
    arrow(ax, (5.55, 5.3), (5.55, 3.2 + bh), "pairs", lx=0.4, ly=0.0)
    arrow(ax, (6.9, 4.2), (8.4, 5.3), "both orders", color=ORANGE, lx=-0.35, ly=0.25)

    box(ax, 8.0, 3.2, bw, bh, "Audit log (JSONL)", "every raw prompt + raw response,\nrole, tag, tokens, latency", edge=GREY)
    arrow(ax, (9.35, 5.3), (9.35, 3.2 + bh), "every call logged", lx=1.0, ly=0.0)

    box(ax, 11.8, 3.2, bw, bh, "Verdicts", "per-criterion scores + rationale;\nper-case pass / fail", edge=BLUE)
    arrow(ax, (13.15, 5.3), (13.15, 3.2 + bh))

    box(ax, 11.8, 1.0, bw, bh, "Report", "suite aggregates, bias probes,\nA/B sign test, CIs", edge=BLUE)
    arrow(ax, (13.15, 3.2), (13.15, 1.0 + bh))

    box(ax, 4.2, 1.0, bw + 1.0, bh, "Judge validation", "agreement vs labels, kappa,\ntest-retest, adversarial probes\n(run through the same pipeline)", edge=ORANGE)
    arrow(ax, (4.2 + bw + 1.0, 1.62), (11.8, 1.62), color=ORANGE)
    fig.savefig(OUT / "problem2_architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    problem1()
    problem2()
    print("wrote", OUT / "problem1_architecture.png", OUT / "problem2_architecture.png")
