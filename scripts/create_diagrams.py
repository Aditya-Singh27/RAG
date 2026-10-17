"""Generate high-resolution architecture diagrams for Problem 1 and Problem 2."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

def render_problem1_diagram():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    # Colors
    c_blue = "#1E40AF"      # Ingest / Retrieval
    c_teal = "#0D9488"      # Storage
    c_amber = "#D97706"     # Decision / Branch
    c_purple = "#6D28D9"    # Generation
    c_red = "#DC2626"       # Fallback / Refusal
    c_box = "#FFFFFF"

    # Title
    ax.text(6, 6.6, "Problem 1: End-to-End Cost-Efficient RAG Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(6, 6.25, "LanceDB On-Disk Vector Store + Cosine Thresholding + Grounded Generation",
            ha="center", va="center", fontsize=10, color="#475569")

    def draw_box(x, y, w, h, title, subtitle, color, bg=c_box, style="solid"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                      edgecolor=color, facecolor=bg, linewidth=2, linestyle=style)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)
        ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center", fontsize=8, color="#334155")

    def draw_arrow(x1, y1, x2, y2, label="", color="#475569", style="-|>", rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.8,
                                    connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.18, label, ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=color, bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CBD5E1", lw=0.8))

    # Top Row: Ingestion Pipeline
    draw_box(0.5, 4.8, 2.2, 1.0, "Corpus Ingestion", "PDF / HTML / Markdown\nHash & SHA-256", c_blue)
    draw_box(3.4, 4.8, 2.2, 1.0, "Recursive Chunking", "size=800, overlap=100\nBoundary-preserving", c_blue)
    draw_box(6.3, 4.8, 2.2, 1.0, "Gemini Embedder", "gemini-embedding-001\n768 dimensions", c_blue)
    draw_box(9.2, 4.8, 2.3, 1.0, "LanceDB Vector Store", "On-disk IVF-PQ / Flat\nScalar metadata filters", c_teal)

    draw_arrow(2.7, 5.3, 3.4, 5.3, "Raw Text")
    draw_arrow(5.6, 5.3, 6.3, 5.3, "Text Chunks")
    draw_arrow(8.5, 5.3, 9.2, 5.3, "768d Vectors")

    # Bottom Row: Query & Generation Pipeline
    draw_box(0.5, 2.0, 2.2, 1.0, "User Query (HTTP)", "FastAPI /query\nk=5, doc_id filter", c_purple)
    draw_box(3.4, 2.0, 2.2, 1.0, "Top-k Retrieval", "Cosine Vector Search\n+ Metadata Pre-filter", c_teal)
    draw_box(6.3, 2.0, 2.2, 1.0, "Similarity Gate", "Score >= MIN_SCORE (0.60)\nRelevance Threshold", c_amber)

    draw_arrow(2.7, 2.5, 3.4, 2.5, "Query Text")
    draw_arrow(3.4+1.1, 4.8, 3.4+1.1, 3.0, "Indexed Chunks", color=c_teal, style="<|-|>")
    draw_arrow(5.6, 2.5, 6.3, 2.5, "Top-5 Chunks")

    # Branches from similarity gate
    draw_box(9.2, 2.7, 2.3, 1.0, "Grounded Generation", "Gemini 3.1 Flash Lite\nStrict [Chunk X] Citations", c_purple)
    draw_box(9.2, 1.0, 2.3, 1.0, "Deterministic Refusal", "'no relevant context'\nZero Hallucination Path", c_red, bg="#FEF2F2")

    draw_arrow(8.5, 2.7, 9.2, 3.2, "Score >= 0.60", color="#16A34A")
    draw_arrow(8.5, 2.3, 9.2, 1.5, "Score < 0.60", color=c_red)

    # Output Arrow
    ax.text(10.35, 0.4, "FastAPI Response (/query)\n{ answer, cited_chunks, latency_ms }",
            ha="center", va="center", fontsize=8, color="#0F172A", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#F1F5F9", ec="#94A3B8"))
    draw_arrow(10.35, 2.7, 10.35, 0.8, color="#475569")
    draw_arrow(10.35, 1.0, 10.35, 0.8, color="#475569")

    plt.tight_layout()
    p = out_dir / "problem1_architecture.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved Problem 1 architecture diagram to {p}")

def render_problem2_diagram():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    c_blue = "#1E40AF"
    c_teal = "#0D9488"
    c_amber = "#D97706"
    c_purple = "#6D28D9"
    c_red = "#DC2626"
    c_box = "#FFFFFF"

    # Title
    ax.text(6, 7.1, "Problem 2: LLM-as-a-Judge Evaluation Pipeline Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(6, 6.75, "Cross-Family Decoupling + Symmetrical Pairwise + Few-Shot Anchors + Resilient JSON Parser",
            ha="center", va="center", fontsize=10, color="#475569")

    def draw_box(x, y, w, h, title, subtitle, color, bg=c_box, style="solid"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                      edgecolor=color, facecolor=bg, linewidth=2, linestyle=style)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)
        ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center", fontsize=8, color="#334155")

    def draw_arrow(x1, y1, x2, y2, label="", color="#475569", style="-|>", rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.8,
                                    connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.18, label, ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=color, bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CBD5E1", lw=0.8))

    # Stage 1: Input Suites
    draw_box(0.5, 5.0, 2.3, 1.2, "Test Suites (JSON)", "Pointwise / Pairwise / A/B\nHuman / Adversarial Probes", c_blue)

    # Stage 2: Prompt Construction & Bias Mitigations
    draw_box(3.5, 5.0, 2.4, 1.2, "Prompt Construction", "6-Dimensional Rubric\nFew-Shot Calibration Anchors", c_purple)
    draw_box(3.5, 3.2, 2.4, 1.2, "Position Bias Mitigation", "Symmetrical Permutation\nA_then_B & B_then_A", c_amber)

    draw_arrow(2.8, 5.6, 3.5, 5.6, "Pointwise Suite")
    draw_arrow(1.65, 5.0, 3.5, 3.8, "Pairwise / A/B", rad=-0.2)

    # Stage 3: Decoupled Judge Call
    draw_box(6.6, 4.1, 2.4, 1.2, "Decoupled LLM Judge", "Gemini 3.1 Flash Lite\n(Cross-family vs Llama 3.2)", c_teal)

    draw_arrow(5.9, 5.6, 6.6, 4.9, "Structured Prompt", rad=-0.15)
    draw_arrow(5.9, 3.8, 6.6, 4.5, "Dual Prompts", rad=0.15)

    # Stage 4: Resilient Parser & Malformed JSON Fallback
    draw_box(9.5, 4.1, 2.2, 1.2, "Resilient Parser", "Markdown / Regex / Fallback\n0% parse failure rate", c_purple)
    draw_arrow(9.0, 4.7, 9.5, 4.7, "Raw Text")

    # Stage 5: Aggregation & Reports
    draw_box(9.5, 1.5, 2.2, 1.2, "Statistical Report", "Cohen's Kappa + Wilson CI\nSign Test A/B Gating", c_blue)
    draw_box(5.0, 1.5, 2.4, 1.2, "Auditable JSONL Logs", "logs/llm_calls.jsonl\nEvery prompt + raw response", c_teal)

    draw_arrow(10.6, 4.1, 10.6, 2.7, "Structured Verdicts")
    draw_arrow(6.6, 4.1, 6.2, 2.7, "Full Audit Trail")
    draw_arrow(9.5, 2.1, 7.4, 2.1, "Verified Calls")

    plt.tight_layout()
    p = out_dir / "problem2_architecture.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved Problem 2 architecture diagram to {p}")

if __name__ == "__main__":
    render_problem1_diagram()
    render_problem2_diagram()
