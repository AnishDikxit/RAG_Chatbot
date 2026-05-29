"""
Configuration constants for the RAG Chatbot pipeline.
"""

# === Source Videos ===
SOURCES = ["LPZh9BOjkQs", "iUU4O1sWtJA", "cUfLrn3TM3M"]

# === Index Settings ===
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# === Chunking ===
CHUNKING_STRATEGY = "semantic"
BREAKPOINT_THRESHOLD_TYPE = "percentile"

# === Retrieval ===
RETRIEVAL_K = 10
BM25_WEIGHT = 0.3
FAISS_WEIGHT = 0.7
TOP_N_RERANK = 5

# === LLM ===
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_TEMPERATURE = 0.2

# === Prompts ===
RAG_PROMPT_TEMPLATE = """
{chat_history}
You are a helpful assistant.
Answer ONLY from the provided transcript context.
When answering, cite the source video and timestamp for each piece of information you use.
If the context is insufficient, just say you don't know.
{context}
Question: {question}
"""

REWRITE_PROMPT_TEMPLATE = """
{chat_history}
Given this conversation history and a follow-up question, rewrite the follow-up into a standalone question. Do NOT answer it — only rewrite.
{question}
"""
