"""Generate realistic technical corpus (PDF, HTML, MD) and 25 gold QA evaluation pairs.
"""
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)


def build_pdfs():
    # PDF 1: LanceDB and IVF-PQ indexing (2 pages)
    pdf1_path = CORPUS_DIR / "lancedb_indexing.pdf"
    c = canvas.Canvas(str(pdf1_path), pagesize=letter)
    
    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "LanceDB Architecture and Vector Indexing (Part 1)")
    c.setFont("Helvetica", 11)
    lines_p1 = [
        "LanceDB is an open-source, serverless vector database built on top of the Lance columnar data format.",
        "Unlike traditional vector databases that require always-on server clusters and keep entire indexes in RAM,",
        "LanceDB is designed to query disk-based vector indices with minimal memory overhead.",
        "",
        "The primary indexing mechanism in LanceDB is IVF-PQ (Inverted File with Product Quantization).",
        "IVF partitions the vector space into voronoi cells using k-means clustering.",
        "During query execution, only the centroids closest to the query vector are visited (controlled by the nprobes parameter).",
        "Product Quantization (PQ) compresses high-dimensional floating point vectors into compact byte codes.",
        "By dividing an original D-dimensional vector into M sub-vectors and quantizing each sub-vector into centroids,",
        "PQ reduces vector memory footprint by a factor of 16x to 32x while maintaining high cosine similarity fidelity.",
        "",
        "A critical feature of LanceDB is zero-copy reads via Apache Arrow. Data transfers between disk and RAM avoid",
        "serialization penalties, enabling sub-millisecond retrieval on local SSDs and cost-effective querying against Amazon S3.",
    ]
    y = 710
    for l in lines_p1:
        c.drawString(50, y, l)
        y -= 20

    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "LanceDB Disk-Based Search and Filtering (Part 2)")
    c.setFont("Helvetica", 11)
    lines_p2 = [
        "LanceDB supports both vector search and zero-cost scalar metadata filtering.",
        "Scalar filters can be executed as pre-filtering (filtering candidate rows before vector distance computation)",
        "or post-filtering (filtering after top-k vector retrieval).",
        "",
        "Pre-filtering prevents low recall caused by restrictive metadata constraints, which frequently degrades post-filtering.",
        "Lance's columnar layout allows selective column projection: when querying for similarities, the engine only reads",
        "the vector column and the filtered metadata columns from storage, skipping unused columns entirely.",
        "",
        "For index persistence, LanceDB stores immutable versioned manifests, supporting ACID transactions and snapshot isolation.",
        "When documents are updated or deleted, LanceDB uses tombstone markers and merge-on-read compaction,",
        "ensuring that writes never block concurrent read queries.",
    ]
    y = 710
    for l in lines_p2:
        c.drawString(50, y, l)
        y -= 20

    c.save()

    # PDF 2: Cost Analysis: Managed Vector DBs vs Serverless Storage (2 pages)
    pdf2_path = CORPUS_DIR / "vector_db_cost_analysis.pdf"
    c2 = canvas.Canvas(str(pdf2_path), pagesize=letter)

    # Page 1
    c2.setFont("Helvetica-Bold", 16)
    c2.drawString(50, 750, "Cost Economics of Vector Databases (Part 1: The Managed Trap)")
    c2.setFont("Helvetica", 11)
    lines_c1 = [
        "Fully managed vector databases like Pinecone (pod-based), Milvus managed, and AWS OpenSearch rely on always-on compute.",
        "For an enterprise index with 1,000,000 vectors of 768 dimensions (approx. 3.1 GB raw float32 embeddings),",
        "traditional HNSW graphs require holding both vectors and graph edges in memory 24/7.",
        "On Pinecone, 1M 768d vectors require at least two s1 or p1 pods for redundancy and performance.",
        "At approximately $70 per pod-month, an idle or lightly queried index costs $140 to $280 per month.",
        "At 10,000,000 vectors, memory requirements scale to 32 GB - 64 GB of RAM, costing $1,400 to $2,800+ monthly.",
        "",
        "This pricing model creates an economic mismatch for read-light workloads (e.g. internal knowledge bases with 500 queries/day).",
        "Organizations pay primarily for idle RAM allocations rather than actual search compute.",
    ]
    y = 710
    for l in lines_c1:
        c2.drawString(50, y, l)
        y -= 20

    c2.showPage()

    # Page 2
    c2.setFont("Helvetica-Bold", 16)
    c2.drawString(50, 750, "Cost Economics of Vector Databases (Part 2: Embedded & Serverless)")
    c2.setFont("Helvetica", 11)
    lines_c2 = [
        "Serverless and embedded vector engines like LanceDB, DuckDB-vss, and sqlite-vec decouple compute from storage.",
        "LanceDB stores compressed IVF-PQ vectors on block storage (local NVMe SSD or cloud object stores like AWS S3).",
        "At 100K vectors (300 MB storage): S3 storage costs roughly $0.01/month, and local disk cost is effectively $0.00.",
        "At 1M vectors (3 GB storage): S3 storage costs roughly $0.07/month.",
        "At 10M vectors (30 GB storage): S3 storage costs roughly $0.70/month.",
        "",
        "Compute costs in an embedded model are co-located within the application process or serverless container (e.g. AWS Lambda/ECS).",
        "During idle periods, compute cost is zero. Under query load, CPU cycles are consumed strictly per request.",
        "The accepted engineering trade-off for this 100x cost reduction is p95 retrieval latency: LanceDB achieves 5-15ms on NVMe SSD",
        "compared to 2-5ms for pure in-memory HNSW, but saves thousands of dollars per year.",
    ]
    y = 710
    for l in lines_c2:
        c2.drawString(50, y, l)
        y -= 20

    c2.save()


def build_markdown_files():
    (CORPUS_DIR / "rag_chunking_best_practices.md").write_text("""# RAG Chunking Strategies and Tradeoffs

Chunking is the foundational step of Retrieval-Augmented Generation that determines whether semantic meaning is preserved or fragmented.

## Fixed-Size vs Recursive Splitting
Fixed-size chunking divides text strictly by character or token counts. While simple, it often splits sentences midway, severing grammatical relationships.
In contrast, recursive character splitting uses a hierarchy of natural separators:
1. Double newlines (`\\n\\n`) to preserve paragraph boundaries.
2. Single newlines (`\\n`) to keep list items and short blocks intact.
3. Sentence terminators (`. `) to ensure complete thoughts.
4. Word boundaries (` `) as a fallback.

## Overlap Window Mechanics
Chunk overlap preserves context between adjacent chunks. If a query matches a concept described across a chunk boundary, an overlap of 10% to 20% (e.g., 100 characters on an 800-character chunk) ensures the relational context is present in at least one retrieved chunk.
However, excessive overlap wastes storage and introduces redundant tokens into the LLM generation context window.

## Multi-Format Ingestion Challenges
- **PDF Documents**: PDFs lack semantic structure. Text must be extracted page-by-page. To preserve exact citation metadata, chunking should occur per-page so chunks never straddle page boundaries.
- **HTML Documents**: Web markup contains navigation bars, footers, and scripts that inject noise. Effective loaders strip `<nav>`, `<footer>`, `<header>`, and `<script>` tags before chunking.
- **Markdown Documents**: Markdown headers (`#`, `##`) denote semantic topic changes and should guide structural chunk boundaries.
""", encoding="utf-8")

    (CORPUS_DIR / "retrieval_evaluation_metrics.md").write_text("""# Standard Information Retrieval Metrics for RAG

Retrieval evaluation assesses the ability of the vector store to place ground-truth relevant chunks at the top of the result list.

## 1. Hit Rate / Recall@k
Hit Rate measures whether at least one relevant document appears within the top-k retrieved results:
`HitRate@k = 1 if (relevant in top-k) else 0`
Recall@k measures the fraction of all relevant documents that were successfully retrieved in the top-k candidates.

## 2. Mean Reciprocal Rank (MRR)
MRR evaluates the rank position of the first relevant chunk. If the first relevant chunk appears at rank 1, reciprocal rank is 1.0; at rank 2, it is 0.5; at rank 3, 0.33:
`MRR = (1 / |Q|) * sum(1 / rank_i)`
MRR heavily penalizes search systems that bury the correct answer below irrelevant distractors.

## 3. Normalized Discounted Cumulative Gain (nDCG@k)
nDCG evaluates graded relevance and penalizes relevant results appearing lower in the ranking using logarithmic discounting:
`DCG@k = sum((2^rel_i - 1) / log2(i + 1))`
`nDCG@k = DCG@k / IDCG@k`
Where IDCG is the Ideal DCG obtained by sorting items by true relevance.

## 4. Context Precision
Context Precision evaluates whether all retrieved chunks in the context window are relevant to the query. High precision ensures that the generator LLM is not distracted by noisy or extraneous chunks, which prevents hallucinations and reduces token costs.
""", encoding="utf-8")

    (CORPUS_DIR / "python_asyncio_internals.md").write_text("""# Python AsyncIO Internal Architecture

Python's `asyncio` module provides single-threaded cooperative multitasking using an event loop, coroutines, and futures.

## The Event Loop Core
At the center of asyncio is the `AbstractEventLoop`. In Python 3.8+ on Windows, the default event loop implementation is `ProactorEventLoop` (backed by I/O Completion Ports - IOCP), whereas Unix systems default to `SelectorEventLoop` (backed by epoll or kqueue).
The event loop continuously monitors file descriptors, timers, and scheduled callbacks. When an `await` expression is encountered on a coroutine, control is yielded back to the event loop, allowing other tasks to execute.

## Tasks vs Futures vs Coroutines
- A **Coroutine** is a generator-like function declared with `async def`. It produces a coroutine object that cannot run by itself until driven by an event loop or awaited.
- A **Future** represents an eventual result of an asynchronous operation. It has states: PENDING, CANCELLED, or FINISHED.
- A **Task** is a subclass of Future that wraps a coroutine and registers it with the event loop for execution. `asyncio.create_task()` schedules concurrent execution.

## ThreadPoolExecutor Integration
CPU-bound tasks or blocking synchronous I/O (such as legacy filesystem reads or heavy crypto) will freeze the event loop. The recommended pattern is offloading blocking functions using `loop.run_in_executor(None, sync_func, *args)` or `asyncio.to_thread(sync_func, *args)`.
""", encoding="utf-8")

    (CORPUS_DIR / "pydantic_v2_architecture.md").write_text("""# Pydantic V2 Architecture and Performance

Pydantic V2 represents a complete rewrite of Python's most popular data validation library, achieving 5x to 50x performance improvements.

## Rust-Powered pydantic-core
The core validation engine was rewritten in Rust (`pydantic-core`). Python type hints are compiled at model declaration time into a static validation graph in Rust.
During data parsing:
1. Python objects (e.g. dicts, JSON strings) are passed directly into compiled Rust code.
2. Rust traverses the validation graph, coercing types and executing validators without Python interpreter overhead.
3. Errors are collected and formatted in a single pass.

## Serialization and JSON Handling
Pydantic V2 includes native JSON parsing and serialization powered by the `jiter` Rust crate. Calling `model.model_dump_json()` parses directly from C-level buffers, bypassing Python's intermediate dictionary creation. This dramatically lowers memory allocation overhead for high-throughput APIs.

## BaseModel Lifecycle
Models declare fields using standard Python type annotations. Validators are registered via `@field_validator` (for single attributes) and `@model_validator` (for cross-field validation, with `mode='before'` or `mode='after'`).
""", encoding="utf-8")


def build_html_files():
    (CORPUS_DIR / "llm_as_judge_biases.html").write_text("""<!DOCTYPE html>
<html>
<head><title>LLM-as-a-Judge Bias Taxonomies and Mitigations</title></head>
<body>
<nav><a href="/">Home</a></nav>
<main>
<h1>Biases in LLM-as-a-Judge Evaluation Systems</h1>
<p>Using large language models as automated judges offers scalable evaluation, but models exhibit well-documented systematic biases that distort quality scores if left unmitigated.</p>

<section>
<h2>1. Position Bias in Pairwise Evaluation</h2>
<p>When presenting two model outputs (Model A vs Model B) to a judge, judges frequently favor the first candidate (Option A) due to primacy effects, or occasionally the second candidate (Option B) due to recency effects.</p>
<p><strong>Mitigation:</strong> Run every evaluation pair twice with swapped order: Order 1 (A then B) and Order 2 (B then A). Measure the position flip rate. Require both evaluations to agree, or compute the average score across both orientations.</p>
</section>

<section>
<h2>2. Verbosity and Length Bias</h2>
<p>Judges inherently correlate response length with thoroughness and quality. A verbose response filled with padding or repetitive elaboration often scores higher than a concise, direct answer.</p>
<p><strong>Mitigation:</strong> Define explicit rubrics that penalize unsupported elaboration. Use adversarial probes (verbose-but-incorrect vs terse-and-correct) to validate that the judge rewards precision over token volume.</p>
</section>

<section>
<h2>3. Self-Enhancement Bias</h2>
<p>LLM judges consistently assign higher ratings to outputs produced by their own model family (e.g. GPT judging GPT, Gemini judging Gemini) due to shared token distributions and stylistic habits.</p>
<p><strong>Mitigation:</strong> Strictly decouple the judge model family from the generator model family (e.g. Gemini judging Llama 3.2, or an ensemble of independent models).</p>
</section>

<section>
<h2>4. Sycophancy and Style Bias</h2>
<p>Judges can be deceived by an authoritative, polished, or overly apologetic tone, overlooking factual falsehoods.</p>
<p><strong>Mitigation:</strong> Force structured per-criterion grounding with separate reasoning steps before assigning numerical scores.</p>
</section>
</main>
<footer>Copyright 2026 AI Evaluation Labs</footer>
</body>
</html>
""", encoding="utf-8")

    (CORPUS_DIR / "fastapi_concurrency_model.html").write_text("""<!DOCTYPE html>
<html>
<head><title>FastAPI Concurrency and Dependency Injection</title></head>
<body>
<header>API Documentation</header>
<article>
<h1>FastAPI Asynchronous Architecture</h1>
<p>FastAPI is a modern web framework built on top of Starlette and Pydantic that achieves performance on par with NodeJS and Go.</p>

<h2>Path Operation Functions: async def vs def</h2>
<p>FastAPI handles endpoints declared with <code>async def</code> differently from standard <code>def</code> endpoints:</p>
<ul>
<li><strong>async def</strong>: Executed directly on the main event loop thread. If an async endpoint executes blocking I/O (e.g. time.sleep or synchronous database calls), it blocks the entire event loop for all requests.</li>
<li><strong>def</strong>: Automatically dispatched by FastAPI to an external thread pool (anyio worker threads). This allows blocking synchronous code to run without halting the event loop.</li>
</ul>

<h2>Dependency Injection (Depends)</h2>
<p>FastAPI features a hierarchical Dependency Injection system. Dependencies can declare sub-dependencies, yield resources (enabling cleanup via context managers), and be shared across routers.</p>
</article>
</body>
</html>
""", encoding="utf-8")

    (CORPUS_DIR / "rag_hallucination_prevention.html").write_text("""<!DOCTYPE html>
<html>
<head><title>Grounded Generation and Hallucination Control</title></head>
<body>
<main>
<h1>Preventing Hallucinations in RAG Pipelines</h1>
<p>The primary vulnerability of RAG architectures is ungrounded generation: the model fabricates facts when retrieved chunks are insufficient or irrelevant.</p>

<h2>The Two-Stage Guardrail</h2>
<ol>
<li><strong>Similarity Score Thresholding (No-Context Branch)</strong>: If the top retrieved chunk has a cosine similarity score below a tuned threshold (e.g. 0.35), the pipeline must bypass the LLM generator entirely. It returns a deterministic refusal message ("I do not have enough relevant context in the provided documents to answer this question."). This eliminates 100% of hallucinations on out-of-domain queries and saves generation latency and API cost.</li>
<li><strong>Strict Citation Enforcement</strong>: The generator system prompt instructs the model to cite chunk markers [Chunk X] for every fact asserted. Claims lacking citations are penalized during automated evaluation.</li>
</ol>
</main>
</body>
</html>
""", encoding="utf-8")

    (CORPUS_DIR / "vector_index_ivf_pq_hnsw.html").write_text("""<!DOCTYPE html>
<html>
<head><title>Vector Index Algorithms: HNSW vs IVF-PQ</title></head>
<body>
<main>
<h1>Comparing Vector Index Algorithms</h1>
<p>Approximate Nearest Neighbor (ANN) search relies on specialized vector indexing structures.</p>

<h2>HNSW (Hierarchical Navigable Small World)</h2>
<p>HNSW builds a multi-layer proximity graph where upper layers have long-range skips and lower layers have dense local connections. HNSW provides exceptional recall (95-99%) and low query latency (1-3ms), but consumes immense RAM (typically 1.5x to 2x the raw vector size) and has slow index build times.</p>

<h2>IVF-PQ (Inverted File with Product Quantization)</h2>
<p>IVF-PQ clusters vectors into Voronoi cells and quantizes vectors into small codebooks. IVF-PQ can operate directly from disk or compressed memory, reducing RAM requirements by 80-95% at the expense of a modest reduction in recall (85-92%) and slightly higher latency (5-15ms).</p>
</main>
</body>
</html>
""", encoding="utf-8")


def build_evaluation_questions():
    questions = [
        # --- 18 In-Domain Questions ---
        {
            "id": "q01",
            "query": "What is the primary vector indexing mechanism used in LanceDB and how does it work?",
            "gold_answer": "LanceDB uses IVF-PQ (Inverted File with Product Quantization). IVF partitions the vector space into voronoi cells using k-means clustering, while Product Quantization divides high-dimensional vectors into sub-vectors and quantizes them into centroids, reducing memory by 16x to 32x.",
            "doc_id": "lancedb_indexing.pdf",
            "doc_type": "pdf",
            "category": "in_domain",
            "keywords": ["IVF-PQ", "Inverted File", "Product Quantization", "k-means", "voronoi"]
        },
        {
            "id": "q02",
            "query": "What are zero-copy reads in LanceDB and what technology powers them?",
            "gold_answer": "Zero-copy reads in LanceDB are powered by Apache Arrow, avoiding serialization penalties between disk and RAM and enabling sub-millisecond retrieval on local SSDs and cost-effective queries on S3.",
            "doc_id": "lancedb_indexing.pdf",
            "doc_type": "pdf",
            "category": "in_domain",
            "keywords": ["zero-copy", "Apache Arrow", "serialization", "SSD"]
        },
        {
            "id": "q03",
            "query": "Why does pre-filtering in LanceDB outperform post-filtering with metadata constraints?",
            "gold_answer": "Pre-filtering evaluates metadata constraints before vector distance computation, preventing low recall caused by restrictive metadata filters which often degrade post-filtering.",
            "doc_id": "lancedb_indexing.pdf",
            "doc_type": "pdf",
            "category": "in_domain",
            "keywords": ["pre-filtering", "post-filtering", "metadata constraints", "recall"]
        },
        {
            "id": "q04",
            "query": "What is the monthly cost comparison between Pinecone and LanceDB for storing 1M vectors?",
            "gold_answer": "For 1M 768d vectors, Pinecone requires at least two pods costing $140 to $280 per month because vectors must stay in memory 24/7. In contrast, LanceDB on S3 costs approximately $0.07 per month for 3 GB of storage, with zero idle compute cost.",
            "doc_id": "vector_db_cost_analysis.pdf",
            "doc_type": "pdf",
            "category": "in_domain",
            "keywords": ["Pinecone", "$140", "$280", "LanceDB", "S3", "$0.07", "1M"]
        },
        {
            "id": "q05",
            "query": "What is the engineering tradeoff of LanceDB compared to in-memory HNSW?",
            "gold_answer": "The tradeoff is retrieval latency: LanceDB achieves 5-15ms on NVMe SSD compared to 2-5ms for in-memory HNSW, in exchange for a roughly 100x reduction in infrastructure cost.",
            "doc_id": "vector_db_cost_analysis.pdf",
            "doc_type": "pdf",
            "category": "in_domain",
            "keywords": ["5-15ms", "2-5ms", "tradeoff", "latency", "HNSW"]
        },
        {
            "id": "q06",
            "query": "What is the hierarchy of separators used in recursive character splitting?",
            "gold_answer": "Recursive character splitting uses double newlines (\\n\\n) for paragraphs, single newlines (\\n) for lines/lists, period-space (. ) for sentences, and spaces as fallback.",
            "doc_id": "rag_chunking_best_practices.md",
            "doc_type": "md",
            "category": "in_domain",
            "keywords": ["\\n\\n", "\\n", ". ", "paragraphs", "sentences"]
        },
        {
            "id": "q07",
            "query": "Why should PDF documents be chunked per page rather than across page boundaries?",
            "gold_answer": "PDFs should be chunked per-page to preserve exact page citation metadata and ensure that chunks never straddle page boundaries.",
            "doc_id": "rag_chunking_best_practices.md",
            "doc_type": "md",
            "category": "in_domain",
            "keywords": ["per-page", "straddle", "page citation", "metadata"]
        },
        {
            "id": "q08",
            "query": "What does Mean Reciprocal Rank (MRR) evaluate in retrieval systems?",
            "gold_answer": "MRR evaluates the rank position of the first relevant chunk, calculated as the mean of the reciprocal ranks (1/rank) across all queries, heavily penalizing systems that rank the relevant item low.",
            "doc_id": "retrieval_evaluation_metrics.md",
            "doc_type": "md",
            "category": "in_domain",
            "keywords": ["MRR", "reciprocal rank", "first relevant", "rank position"]
        },
        {
            "id": "q09",
            "query": "How is nDCG calculated and what does it measure?",
            "gold_answer": "nDCG (Normalized Discounted Cumulative Gain) measures graded relevance using logarithmic discounting to penalize relevant items appearing lower in the ranking, normalized by the Ideal DCG (IDCG).",
            "doc_id": "retrieval_evaluation_metrics.md",
            "doc_type": "md",
            "category": "in_domain",
            "keywords": ["nDCG", "logarithmic discounting", "IDCG", "graded relevance"]
        },
        {
            "id": "q10",
            "query": "What is the difference between SelectorEventLoop and ProactorEventLoop in Python asyncio?",
            "gold_answer": "On Windows, Python 3.8+ defaults to ProactorEventLoop backed by I/O Completion Ports (IOCP), while Unix systems default to SelectorEventLoop backed by epoll or kqueue.",
            "doc_id": "python_asyncio_internals.md",
            "doc_type": "md",
            "category": "in_domain",
            "keywords": ["ProactorEventLoop", "SelectorEventLoop", "IOCP", "Windows", "epoll"]
        },
        {
            "id": "q11",
            "query": "How should blocking synchronous functions be executed inside an asyncio application?",
            "gold_answer": "Blocking synchronous functions should be offloaded to a thread pool using loop.run_in_executor(None, sync_func, *args) or asyncio.to_thread(sync_func, *args) so they do not freeze the event loop.",
            "doc_id": "python_asyncio_internals.md",
            "doc_type": "md",
            "category": "in_domain",
            "keywords": ["run_in_executor", "to_thread", "thread pool", "freeze the event loop"]
        },
        {
            "id": "q12",
            "query": "How does Pydantic V2 achieve its dramatic speed improvements over V1?",
            "gold_answer": "Pydantic V2 rewrote its core validation engine in Rust (pydantic-core), compiling Python type hints into a static validation graph in Rust that parses objects and JSON without Python interpreter overhead.",
            "doc_id": "pydantic_v2_architecture.md",
            "doc_type": "md",
            "category": "in_domain",
            "keywords": ["Rust", "pydantic-core", "validation graph", "jiter"]
        },
        {
            "id": "q13",
            "query": "What is position bias in LLM-as-a-judge and how can it be mitigated?",
            "gold_answer": "Position bias occurs when a judge systematically favors the first or second option in pairwise evaluation. It is mitigated by running every pair in both orders (A-B and B-A), calculating flip rate, and requiring agreement or averaging scores.",
            "doc_id": "llm_as_judge_biases.html",
            "doc_type": "html",
            "category": "in_domain",
            "keywords": ["position bias", "swapped order", "flip rate", "agreement", "both orders"]
        },
        {
            "id": "q14",
            "query": "How should self-enhancement bias be mitigated when using an LLM as an evaluation judge?",
            "gold_answer": "Self-enhancement bias is mitigated by strictly decoupling the judge model family from the generator model family (for example, using Gemini to judge Llama 3.2, or using an ensemble).",
            "doc_id": "llm_as_judge_biases.html",
            "doc_type": "html",
            "category": "in_domain",
            "keywords": ["self-enhancement", "different model family", "decouple", "Gemini", "Llama"]
        },
        {
            "id": "q15",
            "query": "How does FastAPI handle functions declared with 'def' versus 'async def'?",
            "gold_answer": "Endpoints declared with 'async def' run directly on the main event loop thread, while standard 'def' endpoints are dispatched to an external thread pool (anyio worker threads) to prevent blocking the event loop.",
            "doc_id": "fastapi_concurrency_model.html",
            "doc_type": "html",
            "category": "in_domain",
            "keywords": ["async def", "def", "thread pool", "event loop", "anyio"]
        },
        {
            "id": "q16",
            "query": "What is the two-stage guardrail for preventing hallucinations in RAG pipelines?",
            "gold_answer": "The two-stage guardrail consists of: 1) Similarity Score Thresholding (no-context branch) to return a deterministic refusal when cosine similarity is below a threshold, and 2) Strict Citation Enforcement requiring [Chunk X] citations for all asserted facts.",
            "doc_id": "rag_hallucination_prevention.html",
            "doc_type": "html",
            "category": "in_domain",
            "keywords": ["two-stage", "similarity score thresholding", "no-context", "strict citation"]
        },
        {
            "id": "q17",
            "query": "Compare memory usage and recall between HNSW and IVF-PQ vector index algorithms.",
            "gold_answer": "HNSW provides 95-99% recall and 1-3ms latency but requires 1.5x-2x vector size in RAM. IVF-PQ reduces RAM requirements by 80-95% by operating on disk or compressed codes, with 85-92% recall and 5-15ms latency.",
            "doc_id": "vector_index_ivf_pq_hnsw.html",
            "doc_type": "html",
            "category": "in_domain",
            "keywords": ["HNSW", "IVF-PQ", "95-99%", "80-95%", "RAM", "recall"]
        },
        {
            "id": "q18",
            "query": "What does Context Precision measure and why is it important for RAG generation cost?",
            "gold_answer": "Context Precision measures whether the retrieved chunks in the context window are relevant to the query. High precision prevents noisy distractors from causing hallucinations and reduces LLM input token consumption.",
            "doc_id": "retrieval_evaluation_metrics.md",
            "doc_type": "md",
            "category": "in_domain",
            "keywords": ["Context Precision", "distractors", "token consumption", "hallucinations"]
        },

        # --- 4 Cross-Document Synthesis Questions ---
        {
            "id": "q19",
            "query": "Why does LanceDB's IVF-PQ architecture directly enable lower infrastructure costs compared to managed HNSW databases?",
            "gold_answer": "LanceDB's IVF-PQ quantizes vectors and queries them directly from disk or S3 without requiring always-on in-memory HNSW graphs. This decouples storage from compute, allowing 1M vectors to cost cents per month on storage instead of hundreds of dollars in RAM pods.",
            "doc_id": "lancedb_indexing.pdf",
            "doc_type": "pdf",
            "category": "cross_doc",
            "keywords": ["IVF-PQ", "decouple compute", "disk", "S3", "RAM", "always-on"]
        },
        {
            "id": "q20",
            "query": "How do similarity score thresholding and context precision work together to ensure grounded RAG answers?",
            "gold_answer": "Similarity score thresholding acts as an upfront filter to reject queries with no matching context, while high context precision ensures that retrieved chunks passed to the LLM are relevant and free of distracting noise, enabling faithful citations.",
            "doc_id": "rag_hallucination_prevention.html",
            "doc_type": "html",
            "category": "cross_doc",
            "keywords": ["thresholding", "context precision", "grounded", "distracting noise", "citations"]
        },
        {
            "id": "q21",
            "query": "How does position bias interact with verbosity bias in pairwise LLM evaluation?",
            "gold_answer": "Judges may prefer the first candidate due to position bias and also prefer longer candidates due to verbosity bias. Swapping candidate order (A/B and B/A) and penalizing ungrounded length in the rubric prevents verbose answers from winning simply by being presented first.",
            "doc_id": "llm_as_judge_biases.html",
            "doc_type": "html",
            "category": "cross_doc",
            "keywords": ["position bias", "verbosity bias", "swapping order", "rubric"]
        },
        {
            "id": "q22",
            "query": "If an API endpoint in FastAPI performs vector retrieval with LanceDB, why should it be defined carefully with async def versus def?",
            "gold_answer": "If LanceDB disk retrieval performs blocking synchronous file I/O, declaring the endpoint as standard 'def' lets FastAPI run it on worker threads, whereas 'async def' without non-blocking I/O would block the main event loop.",
            "doc_id": "fastapi_concurrency_model.html",
            "doc_type": "html",
            "category": "cross_doc",
            "keywords": ["async def", "def", "blocking I/O", "event loop", "thread pool"]
        },

        # --- 3 Out-of-Domain Questions (Must trigger No-Context Branch!) ---
        {
            "id": "q23",
            "query": "What is the average lifespan of a Greenland shark in arctic waters?",
            "gold_answer": "I do not have enough relevant context in the provided documents to answer this question.",
            "doc_id": None,
            "doc_type": None,
            "category": "out_of_domain",
            "keywords": ["do not have enough relevant context"]
        },
        {
            "id": "q24",
            "query": "Who won the FIFA World Cup in 1998 and what was the final match score?",
            "gold_answer": "I do not have enough relevant context in the provided documents to answer this question.",
            "doc_id": None,
            "doc_type": None,
            "category": "out_of_domain",
            "keywords": ["do not have enough relevant context"]
        },
        {
            "id": "q25",
            "query": "What are the rules for castling in competitive chess tournaments?",
            "gold_answer": "I do not have enough relevant context in the provided documents to answer this question.",
            "doc_id": None,
            "doc_type": None,
            "category": "out_of_domain",
            "keywords": ["do not have enough relevant context"]
        },
    ]

    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/eval_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    print(f"Generated {len(questions)} evaluation questions in data/eval_questions.json")


if __name__ == "__main__":
    print("Building technical corpus...")
    build_pdfs()
    build_markdown_files()
    build_html_files()
    build_evaluation_questions()
    print("Corpus and evaluation set generated successfully!")
