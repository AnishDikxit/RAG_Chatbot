import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text