"""
Evaluation Experiments Script

Runs the RAG pipeline with different configurations and records results.
Each experiment changes one variable from the baseline to measure its impact.

Usage:
    python eval_experiments.py --experiment baseline
    python eval_experiments.py --experiment no_rerank
    python eval_experiments.py --experiment faiss_only
    python eval_experiments.py --experiment equal_weights
    python eval_experiments.py --experiment k20_top3
    python eval_experiments.py --experiment k10_top5
    python eval_experiments.py --experiment all
"""

import os
import json
import pickle
import argparse
import logging
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from config import (
    FAISS_INDEX_PATH, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE,
    RAG_PROMPT_TEMPLATE,
)
from reranker import rerank_docs
from pipeline import format_docs


# =============================================================================
# Experiment Configurations
# =============================================================================

EXPERIMENTS = {
    "baseline": {
        "description": "Semantic + Hybrid (BM25 0.3 / FAISS 0.7) + Rerank top-3",
        "use_bm25": True,
        "bm25_weight": 0.3,
        "faiss_weight": 0.7,
        "retrieval_k": 10,
        "use_reranking": True,
        "top_n_after_rerank": 3,
    },
    "no_rerank": {
        "description": "Semantic + Hybrid (BM25 0.3 / FAISS 0.7) + NO reranking (top-3 from retriever)",
        "use_bm25": True,
        "bm25_weight": 0.3,
        "faiss_weight": 0.7,
        "retrieval_k": 3,
        "use_reranking": False,
        "top_n_after_rerank": 3,
    },
    "faiss_only": {
        "description": "Semantic + FAISS-only (no BM25) + Rerank top-3",
        "use_bm25": False,
        "bm25_weight": 0.0,
        "faiss_weight": 1.0,
        "retrieval_k": 10,
        "use_reranking": True,
        "top_n_after_rerank": 3,
    },
    "equal_weights": {
        "description": "Semantic + Hybrid (BM25 0.5 / FAISS 0.5) + Rerank top-3",
        "use_bm25": True,
        "bm25_weight": 0.5,
        "faiss_weight": 0.5,
        "retrieval_k": 10,
        "use_reranking": True,
        "top_n_after_rerank": 3,
    },
    "k20_top3": {
        "description": "Semantic + Hybrid (BM25 0.3 / FAISS 0.7) + k=20 + Rerank top-3",
        "use_bm25": True,
        "bm25_weight": 0.3,
        "faiss_weight": 0.7,
        "retrieval_k": 20,
        "use_reranking": True,
        "top_n_after_rerank": 3,
    },
    "k10_top5": {
        "description": "Semantic + Hybrid (BM25 0.3 / FAISS 0.7) + k=10 + Rerank top-5",
        "use_bm25": True,
        "bm25_weight": 0.3,
        "faiss_weight": 0.7,
        "retrieval_k": 10,
        "use_reranking": True,
        "top_n_after_rerank": 5,
    },
}


# =============================================================================
# Experiment Helpers
# =============================================================================

def build_retriever(config, vector_store, chunks):
    """Build a retriever based on experiment config."""
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": config["retrieval_k"]}
    )

    if config["use_bm25"]:
        bm25_retriever = BM25Retriever.from_documents(chunks, k=config["retrieval_k"])
        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[config["bm25_weight"], config["faiss_weight"]],
        )
    return faiss_retriever


def answer_question_with_config(question, retriever, config, model, prompt):
    """Run the RAG pipeline with a specific experiment config."""
    retrieved_docs = retriever.invoke(question)

    if config["use_reranking"]:
        final_docs = rerank_docs(question, retrieved_docs, top_n=config["top_n_after_rerank"])
    else:
        final_docs = retrieved_docs[:config["top_n_after_rerank"]]

    context_text = format_docs(final_docs)

    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({
        "chat_history": "",
        "context": context_text,
        "question": question,
    })

    return {
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in final_docs],
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(experiment_name):
    """Run a single experiment and return results."""
    config = EXPERIMENTS[experiment_name]
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Config: {config['description']}")
    print(f"{'='*60}\n")

    # Load shared resources
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    with open(os.path.join(FAISS_INDEX_PATH, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    # Build retriever for this experiment
    retriever = build_retriever(config, vector_store, chunks)

    # Setup LLM and prompt
    model = ChatAnthropic(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question", "chat_history"],
    )

    # Load eval dataset
    with open("eval_dataset.json", "r") as f:
        eval_data = json.load(f)

    # Run pipeline on each question
    results = []
    for i, item in enumerate(eval_data):
        print(f"  [{i+1}/{len(eval_data)}] {item['question'][:60]}...")
        output = answer_question_with_config(
            item["question"], retriever, config, model, prompt
        )
        results.append({
            "question": output["question"],
            "answer": output["answer"],
            "contexts": output["contexts"],
            "ground_truth": item["ground_truth"],
        })

    # Run Ragas evaluation
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    judge_llm = LangchainLLMWrapper(ChatAnthropic(model=LLM_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    eval_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, max_retries=15, max_wait=90),
    )

    # Extract scores
    scores = {}
    for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
        val = eval_result[m.name]
        scores[m.name] = float(np.nanmean(val)) if isinstance(val, list) else float(val)

    # Save results
    os.makedirs("eval_results", exist_ok=True)
    output_path = f"eval_results/{experiment_name}.json"
    output_data = {
        "experiment": experiment_name,
        "description": config["description"],
        "config": config,
        "scores": scores,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results for: {experiment_name}")
    print(f"{'='*60}")
    for metric_name, score in scores.items():
        print(f"  {metric_name}: {score:.4f}")
    print(f"\nSaved to: {output_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation experiments")
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        required=True,
        help="Which experiment to run",
    )
    args = parser.parse_args()

    if args.experiment == "all":
        all_results = []
        for name in EXPERIMENTS:
            result = run_experiment(name)
            all_results.append(result)

        # Print summary table
        print(f"\n\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Experiment':<20} {'Faithful':>10} {'Relevancy':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 60)
        for r in all_results:
            s = r["scores"]
            print(
                f"{r['experiment']:<20} "
                f"{s.get('faithfulness', 0):>10.4f} "
                f"{s.get('answer_relevancy', 0):>10.4f} "
                f"{s.get('context_precision', 0):>10.4f} "
                f"{s.get('context_recall', 0):>10.4f}"
            )
    else:
        run_experiment(args.experiment)
