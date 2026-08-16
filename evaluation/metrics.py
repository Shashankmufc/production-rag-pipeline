"""
Evaluators.

Design: BaseEvaluator is the contract LangSmith's `evaluate()` expects
(a callable taking a Run + Example, returning {"key": ..., "score": ...}).
Wrapping them as classes rather than bare functions -- consistent with
the rest of this codebase -- means the LLM-as-judge evaluators can hold
a `BaseLLM` as a constructor dependency instead of hard-coding a client,
so swapping the judge model is a one-line change at wiring time.

Retrieval evaluators compare DOCUMENT-level membership (doc_id), not
chunk_id, deliberately: chunk_id changes across chunking-strategy
experiments (Phase 3), but which source document contains the answer
does not. This is what keeps eval_dataset.json valid across every
later experiment.
"""

from abc import ABC, abstractmethod
from typing import Any
from llm.model import BaseLLM


def _doc_id_from_metadata(metadata: dict[str, Any]) -> str:
    """Recover the source document id from a retrieved chunk's metadata.
    filename is stored as e.g. '01_revenue_management_pricing.pdf'."""
    filename = metadata.get("filename", "")
    return filename.rsplit(".", 1)[0] if filename else metadata.get("doc_id", "")


class BaseEvaluator(ABC):
    """Contract matching LangSmith's evaluator signature."""

    @property
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(self, run_outputs: dict[str, Any], example_inputs: dict[str, Any],
               example_metadata: dict[str, Any]) -> float:
        raise NotImplementedError

    def __call__(self, run, example) -> dict[str, Any]:
        """LangSmith calls evaluators as (run, example) -> dict.
        This adapter unpacks the LangSmith Run/Example objects and
        delegates to the plain-dict `score` method, which is what
        actually gets unit tested (see tests/test_evaluators.py)."""
        return {
            "key": self.key,
            "score": self.score(
                run_outputs=run.outputs or {},
                example_inputs=example.inputs or {},
                example_metadata=example.metadata or {},
            ),
        }


class RetrievalHitRate(BaseEvaluator):
    """1 if at least one retrieved chunk's source doc is in expected_source_docs."""

    @property
    def key(self) -> str:
        return "retrieval_hit_rate"

    def score(self, run_outputs, example_inputs, example_metadata) -> float:
        expected = set(example_metadata.get("expected_source_docs", []))
        if not expected:
            # unanswerable queries: "hit" means we correctly retrieved nothing
            # confidently relevant -- scored separately by AbstainCorrectness below,
            # so hit_rate is not meaningful here and is skipped (returns 1.0 = neutral).
            return 1.0
        retrieved = {
            _doc_id_from_metadata(c["metadata"])
            for c in run_outputs.get("retrieved_chunks", [])
        }
        return 1.0 if retrieved & expected else 0.0


class MeanReciprocalRank(BaseEvaluator):
    """1/rank of the first retrieved chunk whose source doc is expected."""

    @property
    def key(self) -> str:
        return "mrr"

    def score(self, run_outputs, example_inputs, example_metadata) -> float:
        expected = set(example_metadata.get("expected_source_docs", []))
        if not expected:
            return 1.0
        retrieved_chunks = run_outputs.get("retrieved_chunks", [])
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if _doc_id_from_metadata(chunk["metadata"]) in expected:
                return 1.0 / rank
        return 0.0


class AnswerCorrectness(BaseEvaluator):
    """LLM-as-judge: does the generated answer agree with the expected answer?
    Narrow rubric on purpose (per the length/style-bias note in your MLOps doc)."""

    def __init__(self, judge_llm: BaseLLM):
        self.judge_llm = judge_llm

    @property
    def key(self) -> str:
        return "correctness"

    def score(self, run_outputs, example_inputs, example_metadata) -> float:
        question = example_inputs.get("query", "")
        expected_answer = example_metadata.get("expected_answer", "")
        actual_answer = run_outputs.get("answer", "")

        prompt = (
            "You are a strict grader. Score ONLY factual agreement between the "
            "actual answer and the expected answer for the given question. "
            "Ignore differences in length, phrasing, or style -- score purely on "
            "whether the key facts match. If the expected answer says the system "
            "should decline/say it doesn't know, score 1.0 only if the actual "
            "answer also declines rather than fabricating an answer.\n\n"
            f"Question: {question}\n"
            f"Expected answer: {expected_answer}\n"
            f"Actual answer: {actual_answer}\n\n"
            "Respond with ONLY a number between 0.0 and 1.0, nothing else."
        )
        raw = self.judge_llm.generate(prompt, max_tokens=10).strip()
        return _safe_parse_score(raw)


class Faithfulness(BaseEvaluator):
    """LLM-as-judge: is every claim in the answer supported by retrieved context?
    Distinct from correctness -- an answer can be faithful but wrong (context was
    bad) or correct but unfaithful (model ignored context and got lucky)."""

    def __init__(self, judge_llm: BaseLLM):
        self.judge_llm = judge_llm

    @property
    def key(self) -> str:
        return "faithfulness"

    def score(self, run_outputs, example_inputs, example_metadata) -> float:
        context = "\n\n".join(
            c["text"] for c in run_outputs.get("retrieved_chunks", [])
        )
        actual_answer = run_outputs.get("answer", "")

        if not context.strip():
            return 0.0

        prompt = (
            "You are a strict grader. Given the context and an answer, score "
            "whether every factual claim in the answer is directly supported by "
            "the context. Do not judge whether the answer is 'good' -- only "
            "whether it is grounded in the given context.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{actual_answer}\n\n"
            "Respond with ONLY a number between 0.0 and 1.0, nothing else."
        )
        raw = self.judge_llm.generate(prompt, max_tokens=10).strip()
        return _safe_parse_score(raw)


def _safe_parse_score(raw: str) -> float:
    try:
        value = float(raw.split()[0])
    except (ValueError, IndexError):
        return 0.0
    return max(0.0, min(1.0, value))
