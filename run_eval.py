"""
Run this after Phase 1's requirements are installed and .env has both
ANTHROPIC_API_KEY and (optionally) LANGCHAIN_API_KEY set:

    python run_eval.py

Without LANGCHAIN_API_KEY set, this still runs fully -- LangSmithBridge
is simply skipped, and you get local scoring + MLflow logging. This is
the intended "start here" script for Phase 3 onward: change ONE config
line below (chunker, embedder, etc.), rerun, and compare the new MLflow
run against the baseline.
"""

import os
from dotenv import load_dotenv

from ingestion.loader import PDFLoader
from chunking.chunker import RecursiveChunker
from embeddings.embed import SentenceTransformerEmbedder
from vector_store.chromaDBstore import ChromaDBStore
from retrieval.retriever import VectorRetriever
from prompt.prompt_temp import RAGPromptTemplate
from llm.model import OpenAILLM
from rag_pipeline import RAGPipeline

from evaluation.metrics import RetrievalHitRate, MeanReciprocalRank, AnswerCorrectness, Faithfulness
from evaluation.mlflow_bridge import MLflowRunner

load_dotenv()

SAMPLE_CORPUS = [
    "download/01_revenue_management_pricing.pdf",
    "download/02_rag_system_architecture.pdf",
    "download/03_mlops_eval_observability.pdf",
]

# ---- Phase 3+ knobs: change these, rerun, compare in `mlflow ui` ----
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
RUN_NAME = "baseline-recursive-500-minilm"
# ----------------------------------------------------------------------


def build_pipeline() -> RAGPipeline:
    embedder = SentenceTransformerEmbedder(model_name=EMBEDDING_MODEL)
    vector_store = ChromaDBStore(collection_name=RUN_NAME, persist_dir="./chroma_db_eval")
    return RAGPipeline(
        loader=PDFLoader(),
        chunker=RecursiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
        embedder=embedder,
        vector_store=vector_store,
        retriever=VectorRetriever(embedder=embedder, vector_store=vector_store),
        prompt_template=RAGPromptTemplate(),
        llm=OpenAILLM(),
    )


def main():
    pipeline = build_pipeline()

    if pipeline.vector_store.count() == 0:
        print(f"Ingesting {len(SAMPLE_CORPUS)} documents...")
        num_chunks = pipeline.ingest(SAMPLE_CORPUS)
        print(f"Ingested {num_chunks} chunks.")
    else:
        print(f"Vector store already has {pipeline.vector_store.count()} chunks, skipping ingest.")

    judge_llm = OpenAILLM()  # same model as generation here; fine to use a cheaper model as judge later
    evaluators = [
        RetrievalHitRate(),
        MeanReciprocalRank(),
        AnswerCorrectness(judge_llm=judge_llm),
        Faithfulness(judge_llm=judge_llm),
    ]

    runner = MLflowRunner(experiment_name="rag-pipeline-experiments")
    runner.run(
        pipeline=pipeline,
        evaluators=evaluators,
        run_name=RUN_NAME,
        params={
            "chunker": "RecursiveChunker",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedder": EMBEDDING_MODEL,
            "top_k": TOP_K,
        },
    )

    if os.environ.get("LANGCHAIN_API_KEY"):
        from evaluation.langsmith_bridge import LangSmithBridge
        print("\nLANGCHAIN_API_KEY found -- also running hosted LangSmith evaluation...")
        bridge = LangSmithBridge(pipeline=pipeline, evaluators=evaluators)
        bridge.run_evaluation(experiment_prefix=RUN_NAME)
    else:
        print("\nLANGCHAIN_API_KEY not set -- skipping LangSmith trace upload "
              "(local + MLflow results above are still complete).")


if __name__ == "__main__":
    main()
