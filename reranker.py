"""
Cross-encoder re-ranking module.
"""

import logging

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from config import CROSS_ENCODER_MODEL, TOP_N_RERANK

logger = logging.getLogger(__name__)

# Load cross-encoder once at module level
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)


def rerank_docs(query: str, retrieved_docs: list[Document], top_n: int = TOP_N_RERANK) -> list[Document]:
    """Re-rank retrieved documents using a cross-encoder model.

    Args:
        query: The user's question.
        retrieved_docs: Documents from the initial retrieval step.
        top_n: Number of top documents to return after re-ranking.

    Returns:
        Top-N documents sorted by cross-encoder relevance score.
    """
    if not retrieved_docs:
        return []

    logger.info(f"Re-ranking {len(retrieved_docs)} candidates for query: '{query[:80]}...'")
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)

    # Log all scores at DEBUG level
    for i, (score, doc) in enumerate(zip(scores, retrieved_docs)):
        logger.debug(
            f"  Candidate {i+1}: score={score:.4f} | "
            f"source={doc.metadata['source']} @ {doc.metadata['timestamp']}s"
        )

    # Sort by score descending, take top-N
    scored_docs = sorted(zip(scores, retrieved_docs), reverse=True)
    result = [doc for _, doc in scored_docs[:top_n]]

    # Log selected documents
    logger.info(f"Re-ranking complete. Top-{top_n} selected:")
    for i, (score, doc) in enumerate(scored_docs[:top_n]):
        logger.info(
            f"  #{i+1}: score={score:.4f} | "
            f"source={doc.metadata['source']} @ {doc.metadata['timestamp']}s"
        )

    return result
