from pathlib import Path

from reportlab.pdfgen import canvas

root = Path(__file__).resolve().parent / "download"
root.mkdir(exist_ok=True)

corpus = {
    "01_revenue_management_pricing.pdf": """Revenue Management and Pricing

Alpha and tau in a spiral-up pricing control mechanism
Alpha is the maximum price growth multiplier, representing how much price movement demand can absorb.
Tau is the alert sensitivity threshold, describing the business tolerance for risk and operator intervention.

The materialization rate is the share of shopped or quoted transactions that convert into bookings. Drift in materialization rate by price band indicates the choice set construction or customer preference model may be miscalibrated.

Mixed logit allows taste parameters to vary across the population using a distribution, which makes the choice model more realistic but removes the closed form expression used by standard MNL. This requires simulation-based estimation.

Forecast Value Added (FVA) is used to determine whether a forecasting step improves accuracy over a naive baseline. If it does not, the step should be removed.

Composite criticality scoring combines multiple weak signals such as price delta, competitor deviation, inventory scarcity, and time to departure into a single severity score. This is more robust than relying on a single z-score anomaly.
""",
    "02_rag_system_architecture.pdf": """RAG System Architecture

Recursive splitting uses separators such as paragraphs, lines, and spaces to break texts into chunks. Semantic chunking instead embeds sentences and splits where adjacent embeddings become topically distant.

A cross-encoder reranker is powerful but expensive. It is usually applied only after a fast first-stage retriever produces a shortlist.

HyDE (Hypothetical Document Embeddings) creates a synthetic answer and embeds that instead of the raw query to improve retrieval when the original query is short or vague.

ColBERT stores per-token vectors and uses late interaction scoring, which improves fidelity but increases index size.

Hybrid retrieval combines dense and sparse signals. Sparse retrieval is especially useful for exact matches such as product codes or technical identifiers.
""",
    "03_mlops_eval_observability.pdf": """MLOps, Evaluation, and Observability

Hit rate is binary: did a relevant document appear anywhere in the retrieved results? Mean Reciprocal Rank (MRR) additionally rewards documents ranked higher in the result list.

Faithfulness measures whether each claim in an answer is supported by the retrieved context. Correctness measures whether the answer agrees with the expected answer, regardless of retrieval evidence.

LLM-as-judge systems are susceptible to length and style bias. A narrow rubric and explicit instructions reduce the risk of inflated scores.

Experiment tracking answers 'which configuration performed best', while trace debugging answers 'why did this request fail'. The two systems solve different observability problems.

Retrieval content is untrusted. Prompt injection can cause instructions in retrieved content to override the system prompt, so retrieved text should be clearly delimited and monitored.

PII leakage can be mitigated with redaction at ingestion time and by enforcing document-level access rules during retrieval.
""",
}

for name, text in corpus.items():
    path = root / name
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    c.setTitle(name)
    c.setFont("Helvetica", 12)
    y = 760
    for line in text.splitlines():
        if not line.strip():
            y -= 18
            continue
        c.drawString(50, y, line)
        y -= 18
    c.save()
    print(f"Created: {path}")
