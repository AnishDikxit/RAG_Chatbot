"""
RAG Pipeline — core retrieval and generation logic.

This module handles:
- Index building and caching (FAISS + BM25)
- Hybrid retrieval (dense + sparse)
- Question rewriting for multi-turn conversations
- Answer generation with streaming support

Importable by both main.py (interactive chat) and eval.py (batch evaluation).
"""

import os
import hashlib
import json
import pickle
import logging

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from config import (
    SOURCES, FAISS_INDEX_PATH, EMBEDDING_MODEL,
    CHUNKING_STRATEGY, BREAKPOINT_THRESHOLD_TYPE,
    RETRIEVAL_K, BM25_WEIGHT, FAISS_WEIGHT,
    LLM_MODEL, LLM_TEMPERATURE,
    RAG_PROMPT_TEMPLATE, REWRITE_PROMPT_TEMPLATE,
)
from ingest import ingest_youtube
from reranker import rerank_docs

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# Index Management
# =============================================================================

def get_source_hash() -> str:
    """Generate a hash from the source config for cache invalidation."""
    config = json.dumps({
        "video_id": SOURCES,
        "chunking_strategy": CHUNKING_STRATEGY,
        "breakpoint_threshold_type": BREAKPOINT_THRESHOLD_TYPE,
    }, sort_keys=True)
    return hashlib.md5(config.encode()).hexdigest()


def should_rebuild_index(index_path: str, current_hash: str) -> bool:
    """Check if the index exists and matches the current source config."""
    hash_file = os.path.join(index_path, "source_hash.txt")
    if not os.path.exists(index_path):
        return True
    if not os.path.exists(hash_file):
        return True
    with open(hash_file, "r") as f:
        stored_hash = f.read().strip()
    return stored_hash != current_hash


def save_source_hash(index_path: str, current_hash: str):
    """Save the hash alongside the FAISS index."""
    hash_file = os.path.join(index_path, "source_hash.txt")
    with open(hash_file, "w") as f:
        f.write(current_hash)


def build_or_load_index():
    """Build the FAISS index from scratch or load from cache.

    Returns:
        Tuple of (vector_store, chunks)
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    current_hash = get_source_hash()

    if should_rebuild_index(FAISS_INDEX_PATH, current_hash):
        logger.info("Source changed or no index found. Building FAISS index...")
        docs = []
        for video_id in SOURCES:
            docs.extend(ingest_youtube(video_id))

        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
        )
        chunks = semantic_splitter.split_documents(docs)
        logger.info(f"Semantic chunking produced {len(chunks)} chunks")

        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)

        # Persist chunks for BM25 retriever on subsequent loads
        with open(os.path.join(FAISS_INDEX_PATH, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)

        save_source_hash(FAISS_INDEX_PATH, current_hash)
        logger.info(f"FAISS index saved to '{FAISS_INDEX_PATH}/'")
    else:
        logger.info("Source unchanged. Loading FAISS index from disk...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        with open(os.path.join(FAISS_INDEX_PATH, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)

    return vector_store, chunks


# =============================================================================
# Retriever Setup
# =============================================================================

def build_retriever(vector_store, chunks):
    """Build the hybrid retriever (BM25 + FAISS with EnsembleRetriever)."""
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVAL_K}
    )
    bm25_retriever = BM25Retriever.from_documents(chunks, k=RETRIEVAL_K)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[BM25_WEIGHT, FAISS_WEIGHT],
    )
    return retriever


# =============================================================================
# LLM and Chains
# =============================================================================

def build_llm():
    """Initialize the LLM model."""
    return ChatAnthropic(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def build_prompts():
    """Build the RAG and rewrite prompt templates."""
    rag_prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question", "chat_history"],
    )
    rewrite_prompt = PromptTemplate(
        template=REWRITE_PROMPT_TEMPLATE,
        input_variables=["chat_history", "question"],
    )
    return rag_prompt, rewrite_prompt


# =============================================================================
# Retry-Wrapped Helpers
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def invoke_rewrite_chain(rewrite_chain, chat_history_str: str, question: str) -> str:
    """Rewrite a follow-up question into a standalone question, with retries."""
    return rewrite_chain.invoke({"chat_history": chat_history_str, "question": question})


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def invoke_retriever(retriever, question: str):
    """Retrieve relevant documents, with retries."""
    return retriever.invoke(question)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def stream_llm_response(chain, inputs: dict):
    """Return the stream iterator for the LLM response, with retries."""
    return chain.stream(inputs)


# =============================================================================
# High-Level Pipeline Functions
# =============================================================================

def format_chat_history(history: list) -> str:
    """Format conversation history into a string for the prompt."""
    if not history:
        return ""
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in history)


def format_docs(retrieved_docs) -> str:
    """Format retrieved documents into a context string with source citations."""
    return "\n\n".join(
        f"{doc.page_content}\n[Source: {doc.metadata['source']}, Timestamp: {doc.metadata['timestamp']}s]"
        for doc in retrieved_docs
    )


def answer_question(question: str, chat_history: list = None, *,
                    retriever=None, rewrite_chain=None, rag_prompt=None, model=None) -> dict:
    """Run the full RAG pipeline and return question, answer, and contexts.

    Args:
        question: The user's question.
        chat_history: Optional conversation history for multi-turn.
        retriever: Pre-built retriever (pass to avoid rebuilding).
        rewrite_chain: Pre-built rewrite chain.
        rag_prompt: Pre-built RAG prompt template.
        model: Pre-built LLM model.

    Returns:
        dict with keys: "question", "answer", "contexts"
    """
    chat_history_str = format_chat_history(chat_history) if chat_history else ""

    # Rewrite if there's conversation history
    if chat_history_str and rewrite_chain:
        standalone_question = invoke_rewrite_chain(rewrite_chain, chat_history_str, question)
    else:
        standalone_question = question

    # Retrieve and re-rank
    retrieved_docs = invoke_retriever(retriever, standalone_question)
    reranked_docs = rerank_docs(standalone_question, retrieved_docs)
    context_text = format_docs(reranked_docs)

    # Generate answer
    chain = rag_prompt | model | StrOutputParser()
    answer = chain.invoke({
        "chat_history": chat_history_str,
        "context": context_text,
        "question": question,
    })

    return {
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in reranked_docs],
    }


# =============================================================================
# Pipeline Initialization (lazy — only runs when this module is used)
# =============================================================================

class RAGPipeline:
    """Encapsulates the full RAG pipeline state for reuse."""

    def __init__(self):
        self.vector_store, self.chunks = build_or_load_index()
        self.retriever = build_retriever(self.vector_store, self.chunks)
        self.model = build_llm()
        self.rag_prompt, self.rewrite_prompt = build_prompts()
        self.rewrite_chain = self.rewrite_prompt | self.model | StrOutputParser()

    def answer(self, question: str, chat_history: list = None) -> dict:
        """Answer a question using the full pipeline."""
        return answer_question(
            question,
            chat_history=chat_history,
            retriever=self.retriever,
            rewrite_chain=self.rewrite_chain,
            rag_prompt=self.rag_prompt,
            model=self.model,
        )

    def stream_answer(self, question: str, chat_history: list = None):
        """Stream an answer token-by-token. Yields (token, full_result) on completion.

        Yields string tokens. After the last token, returns the full result dict.
        """
        chat_history_str = format_chat_history(chat_history) if chat_history else ""

        # Rewrite if multi-turn
        if chat_history_str:
            standalone_question = invoke_rewrite_chain(
                self.rewrite_chain, chat_history_str, question
            )
        else:
            standalone_question = question

        # Retrieve and re-rank
        retrieved_docs = invoke_retriever(self.retriever, standalone_question)
        reranked_docs = rerank_docs(standalone_question, retrieved_docs)
        context_text = format_docs(reranked_docs)

        # Stream LLM response
        chain = self.rag_prompt | self.model | StrOutputParser()
        stream = stream_llm_response(chain, {
            "chat_history": chat_history_str,
            "context": context_text,
            "question": question,
        })

        result = ""
        for token in stream:
            result += token
            yield token

        # Store the final result for the caller to access
        self._last_result = {
            "question": question,
            "answer": result,
            "contexts": [doc.page_content for doc in reranked_docs],
        }

    @property
    def last_result(self) -> dict:
        """Access the result from the last stream_answer call."""
        return getattr(self, "_last_result", {})
