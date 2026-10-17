# Standard Information Retrieval Metrics for RAG

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
