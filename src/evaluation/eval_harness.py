"""
evaluation/eval_harness.py

Runs RAGAS evaluation over a test question set.
Measures: faithfulness, answer_relevancy, context_precision, context_recall. #MG Updated 04/20/2026
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.generation.qa_chain import FinRAGPipeline, RAGResponse


DEFAULT_EVAL_QUESTIONS = [
    {
        "question": "What were JPMorgan's total net revenues in fiscal year 2024?",
        "ground_truth": "JPMorgan's total net revenues in fiscal year 2024 were $177.6 billion.",
    },
    {
        "question": "What was JPMorgan's return on equity in 2024?",
        "ground_truth": "JPMorgan's return on equity in 2024 was 18%.",
    },
    {
        "question": "What were Goldman Sachs net revenues in 2024?",
        "ground_truth": "Goldman Sachs reported net revenues of $53.51 billion in 2024, a 16% increase compared to 2023.",
    },
    {
        "question": "What was JPMorgan's provision for credit losses in 2024 and how did it compare to 2023?",
        "ground_truth": "JPMorgan reported a provision for credit losses of $10.7 billion in 2024, up from $9.3 billion in 2023. The provision included $8.6 billion in net charge-offs, driven primarily by Card Services consumer loans.",
    },
    {
        "question": "What is Goldman Sachs strategy for Asset and Wealth Management?",
        "ground_truth": "Goldman Sachs Asset and Wealth Management strategy focuses on managing client assets across equity, fixed income, and alternative investments, offering customized portfolio solutions to ultra-high-net-worth individuals, families, foundations, and endowments globally, combining proprietary and third-party investment offerings with personalized wealth advisory services.",
    },
]

class RAGEvaluator:
    def __init__(self, pipeline: FinRAGPipeline):
        self.pipeline = pipeline

    def run(
        self,
        questions: List[dict] = None,
        output_path: str = "evaluation/results.json",
    ) -> pd.DataFrame:
        """
        Run RAGAS evaluation over a list of {question, ground_truth} dicts.

        Returns a DataFrame with per-question metric scores.
        Saves results to output_path as JSON.
        """
        questions = questions or DEFAULT_EVAL_QUESTIONS

        print(f"Running RAGAS eval on {len(questions)} questions...\n")

        eval_rows = []

        for item in questions:
            q = item["question"]
            gt = item.get("ground_truth", "")

            try:
                resp: RAGResponse = self.pipeline.query(q)

                eval_rows.append({
                    "question": q,
                    "answer": resp.answer,
                    "contexts": [n.node.text for n in resp.reranked_nodes],
                    "ground_truth": gt,
                })
                print(f"  ✓ '{q[:60]}...'")
            except Exception as e:
                print(f"  ✗ Failed: {q[:60]} — {e}")

        dataset = Dataset.from_list(eval_rows)

        metrics = [faithfulness, answer_relevancy, context_precision]
        if any(r["ground_truth"] for r in eval_rows):
            metrics.append(context_recall)

        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        df = result.to_pandas()

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"\nResults saved to {output_path}")

        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        metric_cols = [c for c in df.columns if c not in ("question", "answer", "contexts", "ground_truth")]
        print("\n=== RAGAS Evaluation Summary ===")
        print(df[metric_cols].describe().loc[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, help="Path to JSON file with eval questions")
    parser.add_argument("--output", type=str, default="evaluation/results.json")
    args = parser.parse_args()

    questions = None
    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)

    print("Load your pipeline first via the notebook or main.py, then call evaluator.run()")
