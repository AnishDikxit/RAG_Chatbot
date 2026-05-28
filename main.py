from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from helper import SafeGoogleEmbeddings, format_docs
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# Load variables from .env into the environment
load_dotenv()

# Read the Google API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#Fetching the transcript from the API
yt_api = YouTubeTranscriptApi()
fetched_transcript = yt_api.fetch("LPZh9BOjkQs")

#We convert the fetched transcript to plain text
plain_text = " ".join(chunk.text for chunk in fetched_transcript)

#Once we have the plain_text, we perform indexing on it
#We need to ingest it, load it, then perform splitting
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.create_documents([plain_text])

#We have the chunks, we now generate the embeddings
embeddings = SafeGoogleEmbeddings(
    model="models/gemini-embedding-2",
)

#storing in the vector store
vector_store = FAISS.from_documents(chunks, embeddings)
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