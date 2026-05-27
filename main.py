from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import time
# Load variables from .env into the environment
load_dotenv()

# Read the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#Fetching the transcript from the API
yt_api = YouTubeTranscriptApi()
fetched_transcript = yt_api.fetch("PqVbypvxDto")

#We convert the fetched transcript to plain text
plain_text = " ".join(chunk.text for chunk in fetched_transcript)

#Once we have the plain_text, we perform indexing on it
#We need to ingest it, load it, then perform splitting
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.create_documents([plain_text])

#We have the chunks, we now generate the embeddings
#Initialize the Gemini Embedding Model
class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for i, txt in enumerate(texts):
            results.append(self.embed_query(txt))
            
            # Sleep for 1 second after every request. 
            # This ensures you only make 60 requests per minute, 
            # safely keeping you below your 100 RPM limit!
            time.sleep(1) 
            
        return results
# 2. Use this new class instead of the standard one
embeddings = SafeGoogleEmbeddings(model="models/gemini-embedding-2")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
#storing in the vector store
vector_store = FAISS.from_documents(chunks, embeddings)
#we have successfully completed till the vector store setup
#we move further for the retrieval
#We perform simple similarity search
retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":4})
# 1. Perform the search and store the results in a variable
results = retriever.invoke('What is the mathematical version of AlphaGo?')

# 2. Print out the results so you can actually read them!
print(f"\n--- Found {len(results)} relevant chunks ---")

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)