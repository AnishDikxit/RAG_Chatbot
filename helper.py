import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

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
def format_docs(retrieved_docs):
    context_text = "\n\n".join(f"{doc.page_content}\n[Source: {doc.metadata['source']}, Timestamp: {doc.metadata['timestamp']}s]"
 for doc in retrieved_docs)
    return context_text

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
