"""
EvalHarness: runs a RAGPipeline against evaluation/eval_dataset.json and
scores each query with a list of BaseEvaluator instances.

Deliberately dependency-free (no LangSmith, no MLflow) so you can sanity
check a pipeline config locally in seconds. LangSmithBridge (below) wraps
this same harness for hosted tracing + dataset versioning; MLflowRunner
(in mlflow_bridge.py) wraps it again for experiment comparison. Same
core loop, three different destinations for the results.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from rag_pipeline import RAGPipeline
from evaluation.metrics import BaseEvaluator

DEFAULT_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


@dataclass
class EvalResult:
    query_id: str
    query: str
    category: str
    scores: dict[str, float]


@dataclass
class EvalSummary:
    results: list[EvalResult]
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    scores_by_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def print_report(self) -> None:
        print(f"\n{'='*60}\nEVAL SUMMARY  ({len(self.results)} queries)\n{'='*60}")
        for metric, value in self.aggregate_scores.items():
            print(f"  {metric:<25} {value:.3f}")
        print(f"\n{'-'*60}\nBy category:\n{'-'*60}")
        for category, scores in self.scores_by_category.items():
            print(f"  {category}:")
            for metric, value in scores.items():
                print(f"    {metric:<23} {value:.3f}")


class EvalHarness:
    def __init__(
        self,
        pipeline: RAGPipeline,
        evaluators: list[BaseEvaluator],
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        top_k: int = 5,
    ):
        self.pipeline = pipeline
        self.evaluators = evaluators
        self.top_k = top_k
        self.dataset = self._load_dataset(dataset_path)

    @staticmethod
    def _load_dataset(path: Path | str) -> list[dict[str, Any]]:
        with open(path, "r") as f:
            return json.load(f)

    def run(self) -> EvalSummary:
        results: list[EvalResult] = []
        for item in self.dataset:
            run_outputs = self.pipeline.answer(item["query"], top_k=self.top_k)
            example_inputs = {"query": item["query"]}
            example_metadata = {
                "expected_answer": item["expected_answer"],
                "expected_source_docs": item["expected_source_docs"],
            }
            scores = {
                ev.key: ev.score(run_outputs, example_inputs, example_metadata)
                for ev in self.evaluators
            }
            results.append(
                EvalResult(
                    query_id=item["id"],
                    query=item["query"],
                    category=item["category"],
                    scores=scores,
                )
            )
        return self._summarize(results)

    def _summarize(self, results: list[EvalResult]) -> EvalSummary:
        metric_keys = self.evaluators[0].key and [ev.key for ev in self.evaluators]

        aggregate = {
            key: mean(r.scores[key] for r in results) for key in metric_keys
        }

        by_category: dict[str, dict[str, float]] = {}
        categories = {r.category for r in results}
        for category in categories:
            subset = [r for r in results if r.category == category]
            by_category[category] = {
                key: mean(r.scores[key] for r in subset) for key in metric_keys
            }

        return EvalSummary(results=results, aggregate_scores=aggregate, scores_by_category=by_category)
