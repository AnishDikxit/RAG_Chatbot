# RAG_Chatbot

<!-- ============================================================
     NOTES FOR README POLISH (not user-facing yet)
     ============================================================ -->

## Chunking Strategy — Tradeoffs (TODO: polish)

We use **Semantic Chunking** (`SemanticChunker` from `langchain_experimental`) with `breakpoint_threshold_type="percentile"`.

### Why semantic chunking over fixed-size?

- Fixed-size (e.g., 1000 chars) splits arbitrarily — often mid-sentence or mid-thought. This hurts retrieval quality because a chunk might contain two unrelated topics, or half of an important explanation.
- Semantic chunking measures embedding similarity between consecutive sentences and splits where meaning shifts. Each chunk becomes a coherent "thought unit."

### Alternatives considered

| Strategy | Embed | Return to LLM | Pros | Cons | Best for |
|---|---|---|---|---|---|
| **Fixed-size** (`RecursiveCharacterTextSplitter`) | Fixed chunks | Same | Simple, predictable size | Cuts mid-thought, merges unrelated content | Quick prototypes |
| **Semantic** (`SemanticChunker`) | Variable chunks at topic boundaries | Same | Coherent chunks, better retrieval | Slower index build (embeds every sentence), variable sizes | Transcripts, articles, conversational content |
| **Sentence-Window** | Single sentences | Sentence + N surrounding sentences | High precision for factoid queries | Needs large window for context, complex setup | QA over structured docs |
| **Parent-Document** | Small child chunks | Full parent document | Broad context for LLM | Returns too much irrelevant text, needs doc hierarchy | Long structured documents (papers, legal) |

### Why semantic chunking fits this project

- YouTube transcripts are conversational with natural topic shifts
- Speakers don't respect character boundaries — thoughts span variable lengths
- The cross-encoder re-ranker downstream benefits from coherent chunks (easier to score relevance)

### Configuration

- `breakpoint_threshold_type="percentile"` — splits where similarity drops below the 95th percentile
- Alternatives: `"standard_deviation"` (fewer, larger chunks), `"interquartile"` (balanced)

---

## Enhancements Implemented (TODO: expand each)

- **Streaming responses** — `.stream()` with token-by-token output
- **Logging & error handling** — `logging` module + `tenacity` retries with exponential backoff
- **Hybrid retrieval** — BM25 (sparse) + FAISS (dense) via `EnsembleRetriever` (weights: 0.3/0.7)
- **Cross-encoder re-ranking** — `ms-marco-MiniLM-L-6-v2`, top-10 → top-3
- **Semantic chunking** — `SemanticChunker` with percentile breakpoints
- **Evaluation framework** — Ragas metrics with Anthropic as judge LLM

---

## Evaluation Results

Evaluated on 14 hand-crafted Q&A pairs across 3 YouTube source videos.
Judge LLM: Claude Haiku 4.5 | Embeddings: all-MiniLM-L6-v2

| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------|-------------|-----------------|-------------------|----------------|
| **Baseline** (semantic + hybrid 0.3/0.7 + rerank top-3) | 0.8719 | 0.6802 | 0.7500 | 0.8333 |
| No re-ranking | 0.8199 | 0.6453 | 0.6071 | 0.8333 |
| FAISS-only (no BM25) | 0.8737 | 0.7702 | 0.7619 | 0.8333 |
| Hybrid 0.5/0.5 | 0.8671 | 0.7048 | 0.7500 | 0.8333 |
| Retrieval k=20, rerank top-3 | 0.8857 | 0.6336 | 0.7619 | 0.8333 |
| **Retrieval k=10, rerank top-5** | **0.8916** | **0.8139** | **0.7677** | **0.9405** |

### Key Findings

1. **Re-ranking is critical** — removing it drops context precision by 14 points (0.75 → 0.61) and answer relevancy by 3.5 points. The cross-encoder is doing real work.

2. **More context to the LLM helps significantly** — `k=10, rerank top-5` is the best config across all metrics. Giving the LLM 5 re-ranked chunks instead of 3 boosted context recall from 0.83 → 0.94 and answer relevancy from 0.68 → 0.81.

3. **BM25 adds marginal value here** — FAISS-only actually scored slightly better on answer relevancy (0.77 vs 0.68). For these transcript-style queries, semantic similarity alone is sufficient. BM25 would matter more with keyword-heavy queries (codes, acronyms).

4. **Wider candidate pool (k=20) helps faithfulness but hurts relevancy** — more candidates give the re-ranker better options (faithfulness 0.87 → 0.89), but the answers become less focused.

5. **Ensemble weights are not sensitive** — 0.3/0.7 vs 0.5/0.5 produced nearly identical results, suggesting the re-ranker dominates downstream quality regardless of initial retrieval ordering.