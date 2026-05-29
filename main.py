from youtube_transcript_api import YouTubeTranscriptApi
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
import hashlib
import json
import pickle
from helper import format_docs, ingest_youtube, format_chat_history, rerank_docs
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# Load variables from .env into the environment
load_dotenv()

# Read the Google API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Configuration ===
SOURCES = ["LPZh9BOjkQs", "iUU4O1sWtJA", "cUfLrn3TM3M"]

FAISS_INDEX_PATH = "faiss_index"

def get_source_hash() -> str:
    """Generate a hash from the source config. If any of these change, the index is stale."""
    config = json.dumps({
        "video_id": SOURCES,
        "chunking_strategy": "semantic",
        "breakpoint_threshold_type": "percentile"
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
#We have the chunks, we now generate the embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Check if we need to rebuild or can load from cache
# Check if we need to rebuild or can load from cache
current_hash = get_source_hash()

if should_rebuild_index(FAISS_INDEX_PATH, current_hash):
    logger.info("Source changed or no index found. Building FAISS index...")
    docs = []
    for VIDEO_ID in SOURCES:
        docs.extend(ingest_youtube(VIDEO_ID))
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    )
    chunks = semantic_splitter.split_documents(docs)

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
    # Load persisted chunks for BM25
    with open(os.path.join(FAISS_INDEX_PATH, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

# === Hybrid Retrieval: BM25 (sparse) + FAISS (dense) ===
faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(chunks, k=10)
retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])

#Setting up LLM model first
model = ChatAnthropic(model = "claude-haiku-4-5-20251001", temperature = 0.2)
prompt = PromptTemplate(template="""
      {chat_history}
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      When answering, cite the source video and timestamp for each piece of information you use.
      If the context is insufficient, just say you don't know.
      {context}
      Question: {question}
      
    """, input_variables = ['context', 'question', 'chat_history'])

conversation_history = []


rewrite_prompt = PromptTemplate(
    template = """
    {chat_history}
    Given this conversation history and a follow-up question, rewrite the follow-up into a standalone question. Do NOT answer it — only rewrite.
    {question}
    """, input_variables = ['chat_history', 'question']
)
rewrite_chain = rewrite_prompt | model | StrOutputParser()


# === Retry-wrapped helpers for flaky network calls ===

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def rewrite_question(chat_history_str: str, question: str) -> str:
    """Rewrite a follow-up question into a standalone question, with retries."""
    return rewrite_chain.invoke({"chat_history": chat_history_str, "question": question})


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def retrieve_docs(question: str):
    """Retrieve relevant documents from the vector store, with retries."""
    return retriever.invoke(question)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def stream_llm_response(chain, inputs: dict):
    """Return the stream iterator for the LLM response, with retries.
    
    Note: retries apply to establishing the stream connection.
    If the stream breaks mid-way, the outer try/except in chat() handles it.
    """
    return chain.stream(inputs)


def chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        try:
            chat_history_str = format_chat_history(conversation_history)

            # Rewrite question (retries on transient failures)
            standalone_question = rewrite_question(chat_history_str, user_input)

            # Retrieve context (retries on transient failures)
            retrieved_docs = retrieve_docs(standalone_question)
            reranked_docs = rerank_docs(standalone_question, retrieved_docs)
            context_text = format_docs(reranked_docs)

            # Stream LLM response (retries on connection failure)
            chain = prompt | model | StrOutputParser()
            stream = stream_llm_response(chain, {
                "chat_history": chat_history_str,
                "context": context_text,
                "question": user_input,
            })

            result = ""
            for token in stream:
                result += token
                print(token, end="", flush=True)

            print('\n\n')
            conversation_history.append({"role": "Human", "content": user_input})
            conversation_history.append({"role": "Assistant", "content": result})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.exception("An error occurred while processing your question.")
            print(f"\n[Error] Something went wrong: {e}\n")
if __name__ == "__main__":
    chat()