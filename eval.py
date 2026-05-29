"""
RAG Evaluation Script

Runs the RAG pipeline against a ground-truth dataset and evaluates
using Ragas metrics (faithfulness, answer relevancy, context precision, context recall).

Usage:
    python eval.py
"""

import json
import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from pipeline import RAGPipeline
from config import LLM_MODEL, EMBEDDING_MODEL


def main():
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # Load eval dataset
    with open("eval_dataset.json", "r") as f:
        eval_data = json.load(f)

    # Run pipeline on each question
    print(f"Running pipeline on {len(eval_data)} questions...")
    results = []
    for i, item in enumerate(eval_data):
        print(f"  [{i+1}/{len(eval_data)}] {item['question'][:60]}...")
        output = pipeline.answer(item["question"])
        results.append({
            "question": output["question"],
            "answer": output["answer"],
            "contexts": output["contexts"],
            "ground_truth": item["ground_truth"],
        })

    # Build Ragas dataset
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    # Evaluate
    print("Running Ragas evaluation...")
    judge_llm = LangchainLLMWrapper(ChatAnthropic(model=LLM_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, max_retries=15, max_wait=90),
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
