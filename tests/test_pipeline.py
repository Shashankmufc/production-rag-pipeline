"""
Tests use fake in-memory implementations of each interface instead of
real models/DBs -- this is the payoff of the ABC design: any class that
satisfies BaseEmbedder/BaseVectorStore/BaseLLM can be dropped in, so
tests run in milliseconds with no network/API calls.
"""

from ingestion.loader import BaseLoader, Document
from chunking.chunker import RecursiveChunker
from embeddings.embed import BaseEmbedder
from vector_store.chromaDBstore import BaseVectorStore, RetrievedChunk
from retrieval.retriever import VectorRetriever
from prompt.prompt_temp import RAGPromptTemplate
from llm.model import BaseLLM
from rag_pipeline import RAGPipeline


class FakeLoader(BaseLoader):
    def load(self, source: str) -> Document:
        return Document(doc_id="doc1", text=source, metadata={"filename": "fake.txt"})


class FakeEmbedder(BaseEmbedder):
    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts):
        return [[float(len(t)), 0.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0, 0.0]


class FakeVectorStore(BaseVectorStore):
    def __init__(self):
        self._chunks = []

    def add(self, chunks, embeddings):
        self._chunks.extend(chunks)

    def query(self, query_embedding, top_k=5, filters=None):
        return [
            RetrievedChunk(chunk_id=c.chunk_id, text=c.text, metadata=c.metadata, score=1.0)
            for c in self._chunks[:top_k]
        ]

    def count(self):
        return len(self._chunks)


class FakeLLM(BaseLLM):
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return "fake answer"


def build_test_pipeline() -> RAGPipeline:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    return RAGPipeline(
        loader=FakeLoader(),
        chunker=RecursiveChunker(chunk_size=50, chunk_overlap=5),
        embedder=embedder,
        vector_store=store,
        retriever=VectorRetriever(embedder=embedder, vector_store=store),
        prompt_template=RAGPromptTemplate(),
        llm=FakeLLM(),
    )


def test_ingest_adds_chunks():
    pipeline = build_test_pipeline()
    text = "Revenue management optimizes price and availability. " * 5
    num_chunks = pipeline.ingest([text])
    assert num_chunks > 0
    assert pipeline.vector_store.count() == num_chunks


def test_answer_returns_dict_for_eval_harness():
    """pipeline.answer() returns a plain dict -- this is the shape the
    eval harness and LangSmith evaluators consume directly."""
    pipeline = build_test_pipeline()
    pipeline.ingest(["Alpha is the max price growth multiplier. " * 5])
    result = pipeline.answer("What is alpha?")
    assert result["answer"] == "fake answer"
    assert result["query"] == "What is alpha?"
    assert len(result["retrieved_chunks"]) > 0
    assert "metadata" in result["retrieved_chunks"][0]


def test_answer_typed_returns_dataclass_for_api_callers():
    pipeline = build_test_pipeline()
    pipeline.ingest(["Alpha is the max price growth multiplier. " * 5])
    result = pipeline.answer_typed("What is alpha?")
    assert result.answer == "fake answer"
    assert result.query == "What is alpha?"
    assert len(result.retrieved_chunks) > 0


def test_recursive_chunker_respects_size():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
    doc = Document(doc_id="d1", text="word " * 100, metadata={})
    chunks = chunker.chunk(doc)
    assert all(len(c.text) <= 60 for c in chunks)  # small slack for overlap merge
    assert len(chunks) > 1
