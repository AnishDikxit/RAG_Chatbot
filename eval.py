import sys
import types
import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from main import answer_question
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Use Anthropic as the judge LLM instead of OpenAI
judge_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# Load eval dataset
with open("eval_dataset.json", "r") as f:
    eval_data = json.load(f)

# Run pipeline on each question
results = []
for item in eval_data:
    output = answer_question(item["question"])  # no chat_history → standalone
    results.append({
        "question": output["question"],
        "answer": output["answer"],
        "contexts": output["contexts"],
        "ground_truth": item["ground_truth"],
    })

dataset = Dataset.from_dict({
    "question": [r["question"] for r in results],
    "answer": [r["answer"] for r in results],
    "contexts": [r["contexts"] for r in results],
    "ground_truth": [r["ground_truth"] for r in results],
})

from ragas.run_config import RunConfig

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=RunConfig(max_workers=2, max_retries=15, max_wait=90),
)
print(result)