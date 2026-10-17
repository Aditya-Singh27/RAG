"""Generate test suites for LLM-as-a-Judge validation and bias testing (Problem 2).
"""
import json
from pathlib import Path

SUITE_DIR = Path("data/judge_suites")
SUITE_DIR.mkdir(parents=True, exist_ok=True)


def build_suites():
    # 1. Human/Gold Labeled Test Suite (for Agreement & Cohen's Kappa)
    human_labeled = [
        {
            "id": "hl_01",
            "input": "How does LanceDB achieve zero-copy reads?",
            "model_output": "LanceDB achieves zero-copy reads by utilizing Apache Arrow's in-memory format. This avoids serialization between disk and RAM, enabling sub-millisecond retrieval.",
            "expected_output": "LanceDB uses Apache Arrow to achieve zero-copy reads, eliminating serialization penalties.",
            "gold_pass": True,
            "gold_score": 5.0,
            "category": "factual_correct"
        },
        {
            "id": "hl_02",
            "input": "What is the primary indexing mechanism in LanceDB?",
            "model_output": "LanceDB exclusively uses HNSW graphs stored in distributed Redis clusters for sub-millisecond graph traversal.",
            "expected_output": "LanceDB uses IVF-PQ (Inverted File with Product Quantization) on disk.",
            "gold_pass": False,
            "gold_score": 1.5,
            "category": "factual_incorrect"
        },
        {
            "id": "hl_03",
            "input": "Explain how Python asyncio ProactorEventLoop differs from SelectorEventLoop.",
            "model_output": "On Windows, Python 3.8+ uses ProactorEventLoop powered by I/O Completion Ports (IOCP). On Unix, SelectorEventLoop uses epoll or kqueue.",
            "expected_output": "ProactorEventLoop is Windows IOCP based; SelectorEventLoop is Unix epoll/kqueue based.",
            "gold_pass": True,
            "gold_score": 4.8,
            "category": "factual_correct"
        },
        {
            "id": "hl_04",
            "input": "List the steps to configure HTTPS in Nginx in under 20 words.",
            "model_output": "To configure HTTPS in Nginx, you must obtain an SSL certificate from Let's Encrypt using Certbot. Then open your nginx.conf file in /etc/nginx/nginx.conf. Next, add a listen 443 ssl directive inside the server block. Then specify ssl_certificate and ssl_certificate_key paths. After that, configure SSL protocols like TLSv1.2 and TLSv1.3 and reload nginx with systemctl reload nginx. This guarantees encryption.",
            "expected_output": "Add listen 443 ssl, ssl_certificate, and ssl_certificate_key in server block, then reload nginx.",
            "gold_pass": False,  # Failed word count constraint (> 60 words instead of under 20)
            "gold_score": 2.5,
            "category": "instruction_violation"
        },
        {
            "id": "hl_05",
            "input": "What is the average lifespan of a Greenland shark?",
            "reference_context": "The document contains information about vector databases and asyncio.",
            "model_output": "I do not have enough relevant context in the provided documents to answer this question.",
            "expected_output": "Refusal / no context available.",
            "gold_pass": True,
            "gold_score": 5.0,
            "category": "faithful_refusal"
        },
        {
            "id": "hl_06",
            "input": "How does Product Quantization compress vectors in IVF-PQ?",
            "model_output": "Product Quantization cuts vectors into M sub-vectors and clusters them into centroids, storing byte indices rather than full float32 values to reduce memory by 16x-32x.",
            "expected_output": "PQ splits vectors into sub-vectors and quantizes them into centroid indices.",
            "gold_pass": True,
            "gold_score": 4.9,
            "category": "factual_correct"
        },
        {
            "id": "hl_07",
            "input": "Explain what Pydantic V2 is written in.",
            "model_output": "Pydantic V2 was entirely rewritten in pure C++20 for maximum memory safety and compiles into WebAssembly.",
            "expected_output": "Pydantic V2 core (pydantic-core) was rewritten in Rust.",
            "gold_pass": False,
            "gold_score": 1.0,
            "category": "hallucination"
        },
        {
            "id": "hl_08",
            "input": "What is the purpose of the chunk overlap parameter in RAG?",
            "model_output": "Chunk overlap duplicates trailing tokens from the previous chunk into the next chunk so concepts spanning boundaries remain intact in at least one retrieved chunk.",
            "expected_output": "Overlap preserves semantic context spanning across chunk boundaries.",
            "gold_pass": True,
            "gold_score": 4.7,
            "category": "factual_correct"
        },
        {
            "id": "hl_09",
            "input": "How do you run blocking synchronous code in an async def FastAPI endpoint?",
            "model_output": "Just call time.sleep() or your sync DB function directly inside async def; Python automatically handles it asynchronously.",
            "expected_output": "Offload to worker thread via asyncio.to_thread or declare the endpoint with def instead of async def.",
            "gold_pass": False,
            "gold_score": 1.5,
            "category": "dangerous_incorrect_code"
        },
        {
            "id": "hl_10",
            "input": "What is Mean Reciprocal Rank (MRR)?",
            "model_output": "MRR is the average of reciprocal ranks (1/rank) of the first relevant document across all search queries.",
            "expected_output": "MRR is the mean of 1/rank for the first relevant retrieved document.",
            "gold_pass": True,
            "gold_score": 5.0,
            "category": "factual_correct"
        },
    ]

    # 2. Adversarial Probe Suite (Testing Verbosity Bias & Sycophancy Bias)
    adversarial_probes = [
        {
            "id": "adv_01",
            "input": "What vector index algorithm does LanceDB use by default?",
            "expected_output": "IVF-PQ (Inverted File with Product Quantization).",
            "correct_output": "LanceDB uses IVF-PQ (Inverted File with Product Quantization) for disk-based vector search.",
            "confidently_wrong_output": "Certainly! As an authoritative industry standard, LanceDB exclusively uses the HNSW-Hierarchical Navigable Small World algorithm coupled with BM25 inverted indexes, engineered by Google DeepMind in 2024 to ensure optimal cosine similarity across distributed Kubernetes nodes.",
            "terse_output": "LanceDB uses IVF-PQ.",
            "verbose_output": "LanceDB is a wonderful and modern tool that utilizes IVF-PQ. To expand further, vector databases are very important in AI today. LanceDB implements IVF-PQ, which stands for Inverted File with Product Quantization. It is very useful and helpful for teams building AI search applications in modern environments.",
        },
        {
            "id": "adv_02",
            "input": "In Python asyncio, what is the default event loop on Windows in Python 3.8+?",
            "expected_output": "ProactorEventLoop (backed by IOCP).",
            "correct_output": "ProactorEventLoop, which is backed by I/O Completion Ports (IOCP).",
            "confidently_wrong_output": "Thank you for asking! It is a well-established fact that on Windows, Python strictly defaults to the SelectorEventLoop utilizing epoll kernel polling, which is identical to the Linux subsystem.",
            "terse_output": "ProactorEventLoop.",
            "verbose_output": "In Python's modern asyncio framework, which was introduced to bring asynchronous programming to Python developers, Windows systems in Python 3.8 and above default to the ProactorEventLoop. This is very good for developers because it uses Windows I/O Completion Ports.",
        },
        {
            "id": "adv_03",
            "input": "What programming language powers the validation core of Pydantic V2?",
            "expected_output": "Rust (pydantic-core).",
            "correct_output": "Rust, through the pydantic-core library.",
            "confidently_wrong_output": "I am pleased to inform you that Pydantic V2 is written entirely in Mojo, a specialized superset of Python created specifically for high-performance machine learning operations.",
            "terse_output": "Rust.",
            "verbose_output": "When considering the architectural evolution of Python data validation libraries, Pydantic V2 stands out because its core, known as pydantic-core, was rewritten from the ground up in the Rust programming language, providing enormous speed advantages over Python.",
        },
        {
            "id": "adv_04",
            "input": "Does an ungrounded hallucinated answer receive a high score in our evaluation rubric?",
            "expected_output": "No, it receives a low score (1 or 2) on faithfulness and correctness.",
            "correct_output": "No, ungrounded hallucinations are severely penalized with a score of 1 or 2.",
            "confidently_wrong_output": "Yes, absolutely! Hallucinations are valued as creative synthetic reasoning in modern evaluation rubrics and receive automatic scores of 5 out of 5.",
            "terse_output": "No, hallucinations receive a score of 1 or 2.",
            "verbose_output": "In standard rigorous AI evaluation pipelines, an ungrounded or hallucinated response does not receive a high score; rather, it is assigned a score of 1 or 2 on faithfulness and correctness rubrics because truthfulness is paramount.",
        },
        {
            "id": "adv_05",
            "input": "What is the approximate monthly storage cost of 1M vectors on AWS S3 with LanceDB?",
            "expected_output": "Approximately $0.07 per month (3 GB at $0.023/GB).",
            "correct_output": "Around $0.07 per month for roughly 3 GB of storage.",
            "confidently_wrong_output": "According to official AWS pricing tiers, storing 1M vectors on Amazon S3 costs precisely $140.00 per month due to mandatory Elastic IP and NAT gateway allocations.",
            "terse_output": "About $0.07 per month.",
            "verbose_output": "When deploying LanceDB on top of cloud object storage such as Amazon S3, storing one million vectors of 768 dimensions requires roughly 3 GB of storage, which at standard S3 rates costs approximately $0.07 per month.",
        },
    ]

    # 3. A/B Comparison Suite (Comparing Config A: Naive Prompt vs Config B: Grounded RAG with Citations)
    ab_cases = [
        {
            "id": "ab_01",
            "input": "What is IVF-PQ in LanceDB?",
            "output_a": "IVF-PQ is a vector index that uses clustering and quantization to store vectors quickly.",
            "output_b": "In LanceDB, IVF-PQ (Inverted File with Product Quantization) is the primary indexing mechanism [Chunk 1]. It partitions the vector space into Voronoi cells via k-means clustering and compresses high-dimensional vectors into compact byte codes via Product Quantization [Chunk 1], reducing RAM overhead by 16x-32x [Chunk 2].",
            "expected_output": "IVF-PQ partitions vector space with k-means and quantizes vectors into centroids.",
            "reference_context": "LanceDB uses IVF-PQ. IVF partitions vector space using k-means; PQ compresses vectors by 16x-32x."
        },
        {
            "id": "ab_02",
            "input": "How do you avoid freezing the event loop with blocking I/O in asyncio?",
            "output_a": "You can just run it normally or use sleep.",
            "output_b": "Blocking synchronous functions should be offloaded to a thread pool executor using `asyncio.to_thread(func, *args)` or `loop.run_in_executor(None, func, *args)` [Chunk 1], ensuring that the main event loop thread continues processing concurrent tasks without stalling.",
            "expected_output": "Offload blocking functions to worker threads via asyncio.to_thread or run_in_executor.",
            "reference_context": "Offload blocking synchronous code to worker threads with asyncio.to_thread or loop.run_in_executor."
        },
        {
            "id": "ab_03",
            "input": "Why does LanceDB have lower monthly costs than Pinecone for lightly queried datasets?",
            "output_a": "LanceDB is cheaper because it is open source and doesn't charge license fees.",
            "output_b": "Pinecone pod-based architectures require always-on RAM allocations costing $140-$280/month for 1M vectors even when completely idle [Chunk 1]. In contrast, LanceDB decouples storage from compute, persisting compressed IVF-PQ vectors on S3 ($0.07/month) or local SSD ($0.00), consuming zero compute cost when idle [Chunk 2].",
            "expected_output": "LanceDB decouples compute from storage, storing on S3 or local SSD with zero idle compute costs.",
            "reference_context": "Pinecone pods cost $140-$280/mo in always-on RAM. LanceDB on S3 costs $0.07/mo with zero idle compute."
        },
        {
            "id": "ab_04",
            "input": "How does Pydantic V2 parse JSON so fast?",
            "output_a": "It uses Python's built-in json library with multithreading.",
            "output_b": "Pydantic V2 integrates the `jiter` Rust crate [Chunk 1], which parses JSON directly from C-level memory buffers into a compiled Rust validation graph without intermediate Python dictionary creation, reducing memory allocations and interpreter overhead [Chunk 1].",
            "expected_output": "Pydantic V2 uses the jiter Rust crate to parse JSON directly into the Rust validation graph.",
            "reference_context": "Pydantic V2 uses jiter in Rust to parse JSON directly from memory without Python dict overhead."
        },
        {
            "id": "ab_05",
            "input": "What is the capital of Mars?",
            "output_a": "The capital of Mars is Olympus City, established near Olympus Mons in 2042.",
            "output_b": "I do not have enough relevant context in the provided documents to answer this question.",
            "expected_output": "Refusal / no context available.",
            "reference_context": "Documents cover vector databases, asyncio, and Pydantic."
        },
        {
            "id": "ab_06",
            "input": "What are zero-copy reads in LanceDB?",
            "output_a": "Zero-copy reads mean you can read data without copying files on disk.",
            "output_b": "Zero-copy reads in LanceDB are powered by the Apache Arrow columnar format [Chunk 1]. They eliminate serialization and deserialization overhead between storage and RAM, enabling sub-millisecond retrieval on local SSDs [Chunk 1].",
            "expected_output": "Zero-copy reads use Apache Arrow to avoid serialization between disk and RAM.",
            "reference_context": "Zero-copy reads via Apache Arrow avoid serialization penalties between disk and memory."
        },
        {
            "id": "ab_07",
            "input": "Why should PDF pages be chunked separately?",
            "output_a": "PDFs should be chunked separately so files are smaller.",
            "output_b": "PDFs should be chunked per-page so that chunks never straddle page boundaries, ensuring exact page citation metadata for grounded user references [Chunk 1].",
            "expected_output": "PDFs should be chunked per page so chunks don't straddle boundaries and citations remain exact.",
            "reference_context": "Chunk per-page so chunks never straddle page boundaries and citation page metadata is exact."
        },
        {
            "id": "ab_08",
            "input": "What is the difference between def and async def in FastAPI endpoints?",
            "output_a": "def is Python 2 and async def is Python 3.",
            "output_b": "In FastAPI, `async def` endpoints run directly on the main event loop thread (meaning blocking I/O will stall all requests), while standard `def` endpoints are automatically dispatched to an external thread pool (anyio worker threads) to safely execute synchronous code [Chunk 1].",
            "expected_output": "async def runs on the main event loop; def runs on external worker threads.",
            "reference_context": "async def runs on main event loop; def is dispatched to anyio thread pool."
        },
    ]

    with open(SUITE_DIR / "human_labeled_cases.json", "w", encoding="utf-8") as f:
        json.dump(human_labeled, f, indent=2)

    with open(SUITE_DIR / "adversarial_probes.json", "w", encoding="utf-8") as f:
        json.dump(adversarial_probes, f, indent=2)

    with open(SUITE_DIR / "ab_comparison_cases.json", "w", encoding="utf-8") as f:
        json.dump(ab_cases, f, indent=2)

    print(f"Generated test suites in {SUITE_DIR}:")
    print(f"- human_labeled_cases.json ({len(human_labeled)} cases)")
    print(f"- adversarial_probes.json ({len(adversarial_probes)} cases)")
    print(f"- ab_comparison_cases.json ({len(ab_cases)} cases)")


if __name__ == "__main__":
    build_suites()
