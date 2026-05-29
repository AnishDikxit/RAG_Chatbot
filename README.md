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