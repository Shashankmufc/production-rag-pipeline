"""
MLflowRunner: runs EvalHarness for one pipeline configuration and logs
the result as an MLflow run -- params (what config was tested) + metrics
(how it scored). This is the piece that turns "I tried BGE and it felt
better" into an actual comparable, versioned record.

Division of ownership (carried over from the design discussion):
  - MLflow owns: which config/model version scored best, across experiments
  - LangSmith owns: why a specific query failed, via trace drill-down
  - This module is the bridge: it logs LangSmith/local eval SCORES into
    MLflow, tagged with enough params to reconstruct which pipeline
    variant produced them.
"""

import mlflow
from evaluation.harness import EvalHarness, EvalSummary
from evaluation.metrics import BaseEvaluator
from rag_pipeline import RAGPipeline


class MLflowRunner:
    def __init__(self, experiment_name: str = "rag-pipeline-experiments"):
        mlflow.set_experiment(experiment_name)

    def run(
        self,
        pipeline: RAGPipeline,
        evaluators: list[BaseEvaluator],
        run_name: str,
        params: dict,
        langsmith_experiment_id: str | None = None,
    ) -> EvalSummary:
        """
        params: whatever distinguishes this run -- e.g.
            {"chunker": "RecursiveChunker", "chunk_size": 500,
             "embedder": "all-MiniLM-L6-v2", "top_k": 5}
        so the MLflow run list is filterable/sortable by exactly the
        variables you're experimenting with in Phases 3-6.
        """
        harness = EvalHarness(pipeline=pipeline, evaluators=evaluators)
        summary = harness.run()

        with mlflow.start_run(run_name=run_name):
            for key, value in params.items():
                mlflow.log_param(key, value)

            for metric, value in summary.aggregate_scores.items():
                mlflow.log_metric(metric, value)

            for category, scores in summary.scores_by_category.items():
                for metric, value in scores.items():
                    mlflow.log_metric(f"{category}__{metric}", value)

            if langsmith_experiment_id:
                mlflow.set_tag("langsmith_experiment", langsmith_experiment_id)

        summary.print_report()
        return summary
