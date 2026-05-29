"""
Data ingestion utilities for YouTube transcripts.
"""

import logging

from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def ingest_youtube(video_id: str, group_size: int = 15) -> list[Document]:
    """Fetch YouTube transcript and convert to LangChain Documents.

    Groups consecutive transcript segments together for more coherent chunks.

    Args:
        video_id: YouTube video ID.
        group_size: Number of transcript segments to combine per document.

    Returns:
        List of Document objects with page_content and metadata.
    """
    logger.info(f"Fetching transcript for video: {video_id}")
    yt_api = YouTubeTranscriptApi()
    fetched_transcript = yt_api.fetch(video_id)

    docs = []
    segments = list(fetched_transcript)

    for i in range(0, len(segments), group_size):
        group = segments[i:i + group_size]
        combined_text = " ".join(seg.text for seg in group)
        docs.append(Document(
            page_content=combined_text,
            metadata={
                "source": video_id,
                "timestamp": group[0].start,
            }
        ))

    logger.info(f"  Ingested {len(docs)} document groups from {video_id}")
    return docs
