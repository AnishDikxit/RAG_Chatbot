"""
RAG Chatbot — Interactive Chat Interface

Provides a streaming conversational interface over YouTube transcript content.
Uses the RAG pipeline from pipeline.py for retrieval and generation.

Usage:
    python main.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from pipeline import RAGPipeline, format_chat_history


def chat():
    """Interactive chat loop with streaming responses."""
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    print("Ready! Type your question (or 'quit' to exit).\n")

    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        try:
            # Stream the response token-by-token
            for token in pipeline.stream_answer(user_input, chat_history=conversation_history):
                print(token, end="", flush=True)

            print("\n\n")

            # Update conversation history
            result = pipeline.last_result
            conversation_history.append({"role": "Human", "content": user_input})
            conversation_history.append({"role": "Assistant", "content": result["answer"]})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.exception("An error occurred while processing your question.")
            print(f"\n[Error] Something went wrong: {e}\n")


if __name__ == "__main__":
    chat()
