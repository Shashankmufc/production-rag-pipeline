"""
LangSmithBridge: pushes evaluation/eval_dataset.json to LangSmith once
(idempotent -- skips if the dataset already exists), then runs
langsmith.evaluate() against the traced RAGPipeline.answer function.

This is what gives you the hosted trace-tree view per query -- the local
EvalHarness gives you the same scores, but only LangSmith gives you the
per-step drill-down (which chunk was retrieved, at what latency) when a
specific query scores badly.
"""

import os
from pathlib import Path
from evaluation.harness import DEFAULT_DATASET_PATH
from evaluation.metrics import BaseEvaluator
from rag_pipeline import RAGPipeline
import json


class LangSmithBridge:
    def __init__(
        self,
        pipeline: RAGPipeline,
        evaluators: list[BaseEvaluator],
        dataset_name: str = "rag-eval-v1",
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
    ):
        if not os.environ.get("LANGCHAIN_API_KEY"):
            raise RuntimeError(
                "LANGCHAIN_API_KEY not set. Set it (and LANGCHAIN_TRACING_V2=true) "
                "in .env before using LangSmithBridge -- or use evaluation.harness."
                "EvalHarness for local-only scoring without LangSmith."
            )
        from langsmith import Client
        self.client = Client()
        self.pipeline = pipeline
        self.evaluators = evaluators
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def ensure_dataset(self) -> None:
        """Idempotent: only creates the LangSmith dataset if it doesn't exist yet."""
        if self.client.has_dataset(dataset_name=self.dataset_name):
            print(f"Dataset '{self.dataset_name}' already exists on LangSmith, skipping upload.")
            return

        with open(self.dataset_path, "r") as f:
            items = json.load(f)

        dataset = self.client.create_dataset(
            dataset_name=self.dataset_name,
            description="Pricing/RM + RAG/MLOps domain eval set (synthetic + adversarial).",
        )
        self.client.create_examples(
            inputs=[{"query": item["query"]} for item in items],
            outputs=[{}] * len(items),  # ground truth lives in metadata, not outputs
            metadata=[
                {
                    "expected_answer": item["expected_answer"],
                    "expected_source_docs": item["expected_source_docs"],
                    "category": item["category"],
                    "difficulty": item["difficulty"],
                }
                for item in items
            ],
            dataset_id=dataset.id,
        )
        print(f"Uploaded {len(items)} examples to LangSmith dataset '{self.dataset_name}'.")

    def run_evaluation(self, experiment_prefix: str):
        from langsmith.evaluation import evaluate

        self.ensure_dataset()

        # LangSmith calls this with a plain dict of example.inputs
        def target(inputs: dict) -> dict:
            return self.pipeline.answer(inputs["query"])

        return evaluate(
            target,
            data=self.dataset_name,
            evaluators=self.evaluators,
            experiment_prefix=experiment_prefix,
        )
