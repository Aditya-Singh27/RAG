# RAG Chunking Strategies and Tradeoffs

Chunking is the foundational step of Retrieval-Augmented Generation that determines whether semantic meaning is preserved or fragmented.

## Fixed-Size vs Recursive Splitting
Fixed-size chunking divides text strictly by character or token counts. While simple, it often splits sentences midway, severing grammatical relationships.
In contrast, recursive character splitting uses a hierarchy of natural separators:
1. Double newlines (`\n\n`) to preserve paragraph boundaries.
2. Single newlines (`\n`) to keep list items and short blocks intact.
3. Sentence terminators (`. `) to ensure complete thoughts.
4. Word boundaries (` `) as a fallback.

## Overlap Window Mechanics
Chunk overlap preserves context between adjacent chunks. If a query matches a concept described across a chunk boundary, an overlap of 10% to 20% (e.g., 100 characters on an 800-character chunk) ensures the relational context is present in at least one retrieved chunk.
However, excessive overlap wastes storage and introduces redundant tokens into the LLM generation context window.

## Multi-Format Ingestion Challenges
- **PDF Documents**: PDFs lack semantic structure. Text must be extracted page-by-page. To preserve exact citation metadata, chunking should occur per-page so chunks never straddle page boundaries.
- **HTML Documents**: Web markup contains navigation bars, footers, and scripts that inject noise. Effective loaders strip `<nav>`, `<footer>`, `<header>`, and `<script>` tags before chunking.
- **Markdown Documents**: Markdown headers (`#`, `##`) denote semantic topic changes and should guide structural chunk boundaries.
