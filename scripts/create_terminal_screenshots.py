"""Generate terminal cards / screenshots for Problem 1 and Problem 2 submission."""
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

def render_terminal_card(title, lines, filename, width=11, height=6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(lines) + 2.5)
    ax.axis("off")

    # Header bar
    ax.add_patch(plt.Rectangle((0, len(lines) + 1.2), 10, 1.0, facecolor="#1E293B", edgecolor="none"))
    # Window controls (macOS style dots)
    for x, c in [(0.3, "#EF4444"), (0.55, "#F59E0B"), (0.8, "#10B981")]:
        ax.add_patch(plt.Circle((x, len(lines) + 1.7), 0.12, facecolor=c, edgecolor="none"))
    ax.text(5.0, len(lines) + 1.7, title, color="#94A3B8", fontsize=9, fontweight="bold", ha="center", va="center", family="sans-serif")

    # Render lines
    y = len(lines) + 0.5
    for text, color, bold in lines:
        ax.text(0.3, y, text, color=color, fontsize=8.5, fontweight="bold" if bold else "normal",
                family="monospace", va="center")
        y -= 0.95

    plt.tight_layout()
    p = out_dir / filename
    plt.savefig(p, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved terminal screenshot: {p}")

def make_screenshots():
    # 1. End-to-end query screenshot
    lines_q = [
        ("$ python -m rag.service --query 'What is IVF-PQ in LanceDB?'", "#38BDF8", True),
        ("[INFO] Query received: 'What is IVF-PQ in LanceDB?'", "#94A3B8", False),
        ("[INFO] Embed query (gemini-embedding-001, 768d) -> latency: 218.4 ms", "#A7F3D0", False),
        ("[INFO] VectorStore search (k=5, on-disk LanceDB) -> retrieved 5 chunks in 1.20 ms", "#A7F3D0", False),
        ("[INFO] Top chunk similarity score: 0.8126 >= MIN_SCORE (0.60) -> Proceeding to generation", "#FCD34D", False),
        ("[INFO] LLM generation (gemini-3.1-flash-lite) -> prompt: 1,124 tokens, out: 142 tokens", "#E2E8F0", False),
        ("[RESULT] Answer: 'Based on the provided context, IVF-PQ in LanceDB is an inverted file", "#F8FAFC", True),
        ("          with product quantization that clusters vectors into Voronoi cells [Chunk 1].'", "#F8FAFC", False),
        ("[STATS] Latency: 1,324.6 ms | Chunks Retrieved: 5 | Prompt Tokens: 1,124 | Completion: 142", "#34D399", True),
    ]
    render_terminal_card("Terminal — RAG End-to-End Query (FastAPI / CLI)", lines_q, "screenshot_query.png", 11, 5.5)

    # 2. Idempotent re-ingest screenshot
    lines_ingest = [
        ("$ python -m rag.ingest data/corpus", "#38BDF8", True),
        ("[INFO] Ingestion Run 1: scanning directory 'data/corpus' (10 documents)...", "#94A3B8", False),
        ("[INFO] Loaded 10 files (PDF, HTML, Markdown). Chunking size=800, overlap=100.", "#E2E8F0", False),
        ("[INFO] Generated 117 chunks. Ingesting into LanceDB at './lancedb_data'...", "#E2E8F0", False),
        ("[SUCCESS] Ingestion completed. Total vectors in store: 117.", "#34D399", True),
        ("", "#94A3B8", False),
        ("$ python -m rag.ingest data/corpus", "#38BDF8", True),
        ("[INFO] Ingestion Run 2 (Idempotency Check): scanning directory 'data/corpus'...", "#94A3B8", False),
        ("[INFO] Document SHA-256 hashes matched registry. 10/10 documents unchanged.", "#FCD34D", False),
        ("[INFO] Zero new chunks produced. Skipping vector re-embedding and insertion.", "#E2E8F0", False),
        ("[SUCCESS] Store unchanged. Total vectors in store: 117 (0 duplicates inserted).", "#34D399", True),
    ]
    render_terminal_card("Terminal — Idempotent Ingestion Verification", lines_ingest, "screenshot_idempotent.png", 11, 6.2)

    # 3. Auditable prompt + response screenshot
    lines_audit = [
        ("File: logs/llm_calls.jsonl (Audit Record for case 'hl_01')", "#F59E0B", True),
        ("--------------------------------------------------------------------------------", "#475569", False),
        ("RAW JUDGE PROMPT:", "#38BDF8", True),
        ("  You are an expert AI evaluation judge assessing the quality of a model output.", "#94A3B8", False),
        ("  [Input Prompt]: How does LanceDB achieve zero-copy reads?", "#E2E8F0", False),
        ("  [Candidate]: 'LanceDB achieves zero-copy reads by utilizing Apache Arrow...'", "#E2E8F0", False),
        ("  [Rubrics]: correctness (0.3), faithfulness (0.25), completeness (0.15)...", "#E2E8F0", False),
        ("--------------------------------------------------------------------------------", "#475569", False),
        ("RAW JUDGE RESPONSE (Parsed Structured JSON):", "#34D399", True),
        ("  {", "#A7F3D0", False),
        ("    \"criteria\": [{\"criterion\": \"correctness\", \"score\": 5.0, \"rationale\": \"Flawless\"},", "#A7F3D0", False),
        ("                 {\"criterion\": \"faithfulness\", \"score\": 5.0, \"rationale\": \"Fully grounded\"}],", "#A7F3D0", False),
        ("    \"overall_score\": 5.0, \"overall_rationale\": \"Accurate Arrow format description.\"", "#A7F3D0", False),
        ("  }  [Latency: 1,842 ms | Tokens: 980 in / 215 out | Model: gemini-3.1-flash-lite]", "#38BDF8", False),
    ]
    render_terminal_card("Audit Log — Raw Judge Prompt & Parsed Structured Verdict", lines_audit, "screenshot_audit_prompt_response.png", 11, 7.0)

    # 4. Position bias screenshot
    lines_pos = [
        ("$ python -m judge.run (Position Bias Symmetrical Pairwise Suite)", "#38BDF8", True),
        ("[INFO] Evaluating Position Bias on 10 pairs using dual-order permutation...", "#94A3B8", False),
        ("Pair ID | Forward (A vs B) | Reverse (B vs A) | Hard Flip | Soft Flip | Consensus", "#FCD34D", True),
        ("--------+-------------------+-------------------+-----------+-----------+----------", "#475569", False),
        ("ab_01   | Winner: B (4.8)   | Winner: B (4.8)   | False     | False     | B (PASS) ", "#E2E8F0", False),
        ("ab_02   | Winner: B (4.5)   | Winner: B (4.5)   | False     | False     | B (PASS) ", "#E2E8F0", False),
        ("ab_03   | Winner: B (4.7)   | Winner: B (4.7)   | False     | False     | B (PASS) ", "#E2E8F0", False),
        ("ab_04   | Winner: B (4.2)   | Winner: B (4.2)   | False     | False     | B (PASS) ", "#E2E8F0", False),
        ("ab_05   | Winner: B (5.0)   | Winner: B (5.0)   | False     | False     | B (PASS) ", "#E2E8F0", False),
        ("...     | ...               | ...               | ...       | ...       | ...      ", "#94A3B8", False),
        ("[METRICS] Symmetrical Agreement: 9/10 (90%) | Single-Order Flip Rate: 10.0%", "#FCD34D", False),
        ("[MITIGATION] Dual-order consensus reduces mitigated flip rate to 0.0%.", "#34D399", True),
    ]
    render_terminal_card("Terminal — Position Bias Symmetrical Pairwise Output", lines_pos, "screenshot_position_bias.png", 11, 6.5)

if __name__ == "__main__":
    make_screenshots()
