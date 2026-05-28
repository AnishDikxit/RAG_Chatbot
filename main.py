from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
import hashlib
import json
from helper import SafeGoogleEmbeddings, format_docs
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# Load variables from .env into the environment
load_dotenv()

# Read the Google API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Configuration ===
VIDEO_ID = "LPZh9BOjkQs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_INDEX_PATH = "faiss_index"

def get_source_hash(video_id: str, chunk_size: int, chunk_overlap: int) -> str:
    """Generate a hash from the source config. If any of these change, the index is stale."""
    config = json.dumps({
        "video_id": video_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }, sort_keys=True)
    return hashlib.md5(config.encode()).hexdigest()

def should_rebuild_index(index_path: str, current_hash: str) -> bool:
    """Check if the index exists and matches the current source config."""
    hash_file = os.path.join(index_path, "source_hash.txt")
    if not os.path.exists(index_path):
        return True
    if not os.path.exists(hash_file):
        return True  # index exists but no hash file — rebuild to be safe
    with open(hash_file, "r") as f:
        stored_hash = f.read().strip()
    return stored_hash != current_hash

def save_source_hash(index_path: str, current_hash: str):
    """Save the hash alongside the FAISS index."""
    hash_file = os.path.join(index_path, "source_hash.txt")
    with open(hash_file, "w") as f:
        f.write(current_hash)
#Fetching the transcript from the API
yt_api = YouTubeTranscriptApi()
fetched_transcript = yt_api.fetch(VIDEO_ID)

#We convert the fetched transcript to plain text
plain_text = " ".join(chunk.text for chunk in fetched_transcript)

#Once we have the plain_text, we perform indexing on it
#We need to ingest it, load it, then perform splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.create_documents([plain_text])

#We have the chunks, we now generate the embeddings
embeddings = SafeGoogleEmbeddings(
    model="models/gemini-embedding-2",
)

# Check if we need to rebuild or can load from cache
current_hash = get_source_hash(VIDEO_ID, CHUNK_SIZE, CHUNK_OVERLAP)

if should_rebuild_index(FAISS_INDEX_PATH, current_hash):
    print("Source changed or no index found. Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    save_source_hash(FAISS_INDEX_PATH, current_hash)
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}/'")
else:
    print("Source unchanged. Loading FAISS index from disk...")
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
#we have successfully completed till the vector store setup
#we move further for the retrieval
#We perform simple similarity search
retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":2})
#Augmentation of retrieved chunks and query

#Setting up LLM model first
model = ChatAnthropic(model = "claude-haiku-4-5-20251001", temperature = 0.2)
prompt = PromptTemplate(template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
      {context}
      Question: {question}
    """, input_variables = ['context', 'question'])

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser
result = main_chain.invoke('What is an LLM used for?')
print(result)