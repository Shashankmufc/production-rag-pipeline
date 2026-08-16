from llm.model import BaseLLM
from evaluation.metrics import (
    RetrievalHitRate, MeanReciprocalRank, AnswerCorrectness, Faithfulness, _safe_parse_score,
)


class FakeJudgeLLM(BaseLLM):
    """Returns a fixed score regardless of prompt -- lets us test the
    evaluator's parsing/plumbing without a real judge call."""

    def __init__(self, fixed_score: str = "0.8"):
        self.fixed_score = fixed_score

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.fixed_score


def make_run_outputs(doc_ids: list[str]) -> dict:
    return {
        "answer": "some answer",
        "retrieved_chunks": [
            {"chunk_id": f"c{i}", "text": "text", "score": 1.0,
             "metadata": {"filename": f"{doc_id}.pdf"}}
            for i, doc_id in enumerate(doc_ids)
        ],
    }


def test_hit_rate_scores_1_when_expected_doc_present():
    evaluator = RetrievalHitRate()
    run_outputs = make_run_outputs(["01_revenue_management_pricing", "02_rag_system_architecture"])
    score = evaluator.score(run_outputs, {}, {"expected_source_docs": ["01_revenue_management_pricing"]})
    assert score == 1.0


def test_hit_rate_scores_0_when_expected_doc_absent():
    evaluator = RetrievalHitRate()
    run_outputs = make_run_outputs(["02_rag_system_architecture"])
    score = evaluator.score(run_outputs, {}, {"expected_source_docs": ["01_revenue_management_pricing"]})
    assert score == 0.0


def test_hit_rate_neutral_for_unanswerable_queries():
    evaluator = RetrievalHitRate()
    run_outputs = make_run_outputs(["01_revenue_management_pricing"])
    score = evaluator.score(run_outputs, {}, {"expected_source_docs": []})
    assert score == 1.0  # neutral -- correctness/faithfulness handle unanswerable cases


def test_mrr_rewards_higher_rank():
    evaluator = MeanReciprocalRank()
    # expected doc is 2nd in the list -> MRR should be 0.5
    run_outputs = make_run_outputs(["02_rag_system_architecture", "01_revenue_management_pricing"])
    score = evaluator.score(run_outputs, {}, {"expected_source_docs": ["01_revenue_management_pricing"]})
    assert score == 0.5


def test_mrr_zero_when_never_found():
    evaluator = MeanReciprocalRank()
    run_outputs = make_run_outputs(["02_rag_system_architecture"])
    score = evaluator.score(run_outputs, {}, {"expected_source_docs": ["03_mlops_eval_observability"]})
    assert score == 0.0


def test_answer_correctness_parses_judge_score():
    evaluator = AnswerCorrectness(judge_llm=FakeJudgeLLM(fixed_score="0.75"))
    run_outputs = {"answer": "alpha is the max price multiplier"}
    score = evaluator.score(run_outputs, {"query": "what is alpha?"},
                             {"expected_answer": "alpha is the max growth multiplier"})
    assert score == 0.75


def test_faithfulness_zero_when_no_context_retrieved():
    evaluator = Faithfulness(judge_llm=FakeJudgeLLM(fixed_score="0.9"))
    run_outputs = {"answer": "some answer", "retrieved_chunks": []}
    score = evaluator.score(run_outputs, {}, {})
    assert score == 0.0  # no context to be faithful to -> can't be faithful


def test_safe_parse_score_clamps_out_of_range_values():
    assert _safe_parse_score("1.5") == 1.0
    assert _safe_parse_score("-0.3") == 0.0
    assert _safe_parse_score("not a number") == 0.0
    assert _safe_parse_score("0.42") == 0.42
