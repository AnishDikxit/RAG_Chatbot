import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sentence_transformers import CrossEncoder
import logging
logging.basicConfig(
    level = logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for i, txt in enumerate(texts):
            while True:
                try:
                    results.append(self.embed_query(txt))
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"Rate limited at chunk {i}. Waiting 35s...")
                        time.sleep(35)
                    else:
                        raise
            time.sleep(1)
        return results
    def embed_query(self, text: str) -> list[float]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return super().embed_query(text)
            except Exception as e:
                if ("500" in str(e) or "INTERNAL" in str(e) or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                    print(f"Server error on embed_query (attempt {attempt + 1}/{max_retries}). Waiting 35s...")
                    time.sleep(35)
                else:
                    raise
        raise Exception("Gemini embedding API failed after all retries.")

def format_docs(retrieved_docs):
    context_text = "\n\n".join(f"{doc.page_content}\n[Source: {doc.metadata['source']}, Timestamp: {doc.metadata['timestamp']}s]"
 for doc in retrieved_docs)
    return context_text
@retry(stop = stop_after_attempt(3), wait = wait_exponential(multiplier =1, min = 2, max =10), retry = retry_if_exception_type(Exception))
def ingest_youtube(video_id: str) -> list[Document]:
    yt_api = YouTubeTranscriptApi()
    fetched_transcript = yt_api.fetch(video_id)
    
    docs = []
    group_size = 15
    segments = list(fetched_transcript)
    
    for i in range(0, len(segments), group_size):
        group = segments[i:i + group_size]
        combined_text = " ".join(seg.text for seg in group)
        docs.append(Document(
            page_content=combined_text,
            metadata={
                "source": video_id,
                "timestamp": group[0].start  # timestamp of first segment in group
            }
        ))
    return docs
def format_chat_history(history)->str:
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in history)

def rerank_docs(query:str, retrieved_docs):
    logger.info(f"Re-ranking {len(retrieved_docs)} candidates for query: '{query[:80]}...'")
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)

    # Log scores before sorting
    for i, (score, doc) in enumerate(zip(scores, retrieved_docs)):
        logger.debug(f"  Candidate {i+1}: score={score:.4f} | source={doc.metadata['source']} @ {doc.metadata['timestamp']}s")

    temp = sorted(zip(scores, retrieved_docs), reverse = True)
    result = []
    for i in range(3):
        score, doc = temp[i]
        result.append(doc)

    # Log the top-3 after re-ranking
    logger.info("Re-ranking complete. Top-3 selected:")
    for i, (score, doc) in enumerate(temp[:3]):
        logger.info(f"  #{i+1}: score={score:.4f} | source={doc.metadata['source']} @ {doc.metadata['timestamp']}s")

    return result