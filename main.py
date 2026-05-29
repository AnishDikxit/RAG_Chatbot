from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
import hashlib
import json
from helper import format_docs, ingest_youtube, format_chat_history
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import time

# Load variables from .env into the environment
load_dotenv()

# Read the Google API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Configuration ===
SOURCES = ["LPZh9BOjkQs", "iUU4O1sWtJA", "cUfLrn3TM3M"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_INDEX_PATH = "faiss_index"

def get_source_hash(chunk_size: int, chunk_overlap: int) -> str:
    """Generate a hash from the source config. If any of these change, the index is stale."""
    config = json.dumps({
        "video_id": SOURCES,
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
#We have the chunks, we now generate the embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Check if we need to rebuild or can load from cache
# Check if we need to rebuild or can load from cache
current_hash = get_source_hash(CHUNK_SIZE, CHUNK_OVERLAP)

if should_rebuild_index(FAISS_INDEX_PATH, current_hash):
    print("Source changed or no index found. Building FAISS index...")
    docs = []
    for VIDEO_ID in SOURCES:
        docs.extend(ingest_youtube(VIDEO_ID))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
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
retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":4})
#Augmentation of retrieved chunks and query

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

def chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        chat_history_str = format_chat_history(conversation_history)
        standalone_question = rewrite_chain.invoke({"chat_history":chat_history_str, "question":user_input})
        retrieved_docs = retriever.invoke(standalone_question)
        context_text = format_docs(retrieved_docs)
        result = ""
        for token in (prompt | model | StrOutputParser()).stream({
            "chat_history": chat_history_str,
            "context": context_text,
            "question": user_input
        }):
            result+=token
            print(token, end="", flush = True)
        
        print('\n\n')
        conversation_history.append({"role":"Human", "content":user_input})
        conversation_history.append({"role":"Assistant", "content":result})
if __name__ == "__main__":
    chat()