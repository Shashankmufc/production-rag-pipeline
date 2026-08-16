"""
RAGPipeline: composes every layer behind one class. Every dependency is
injected in __init__, never constructed inside methods -- that's what
makes each layer swappable and independently testable (see tests/).

Phase 2 adds tracing: `ingest` and `answer` are wrapped with LangSmith's
`@traceable`. This is a no-op decorator when LANGCHAIN_TRACING_V2 is unset
(no network calls, no crash if langsmith isn't installed) -- so the same
pipeline runs identically whether tracing is on or off, which is exactly
why tests/test_pipeline.py still passes without any LangSmith config.
"""

from dataclasses import dataclass
from ingestion.loader import BaseLoader, Document
from chunking.chunker import BaseChunker, Chunk
from embeddings.embed import BaseEmbedder
from vector_store.chromaDBstore import BaseVectorStore, RetrievedChunk
from retrieval.retriever import BaseRetriever
from prompt.prompt_temp import BasePromptTemplate
from llm.model import BaseLLM

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - keeps the pipeline runnable without langsmith installed
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not (args and callable(args[0])) else args[0]


@dataclass
class RAGResponse:
    answer: str
    query: str
    retrieved_chunks: list[RetrievedChunk]


class RAGPipeline:
    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        retriever: BaseRetriever,
        prompt_template: BasePromptTemplate,
        llm: BaseLLM,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.llm = llm

    @traceable(name="rag_ingest", run_type="chain")
    def ingest(self, source_paths: list[str]) -> int:
        """Load -> chunk -> embed -> store. Returns number of chunks added."""
        documents: list[Document] = self.loader.load_many(source_paths)
        chunks: list[Chunk] = self.chunker.chunk_many(documents)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.add(chunks, embeddings)
        return len(chunks)

    @traceable(name="rag_answer", run_type="chain")
    def answer(self, query: str, top_k: int = 5) -> dict:
        """Retrieve -> build prompt -> generate.

        Returns a plain dict (not the RAGResponse dataclass) because this
        is the function the eval harness and LangSmith's `evaluate()` call
        directly -- evaluators in evaluation/metrics.py read `run.outputs`
        as a dict, so the traced function's return shape needs to already
        be JSON-friendly rather than requiring a dataclass-to-dict step
        at every call site.
        """
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        prompt = self.prompt_template.build(query, retrieved)
        answer_text = self.llm.generate(prompt)
        return {
            "answer": answer_text,
            "query": query,
            "retrieved_chunks": [
                {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata, "score": c.score}
                for c in retrieved
            ],
        }

    def answer_typed(self, query: str, top_k: int = 5) -> RAGResponse:
        """Convenience wrapper for callers (e.g. main.py's FastAPI route)
        that want the typed dataclass instead of a raw dict."""
        result = self.answer(query, top_k=top_k)
        chunks = [
            RetrievedChunk(chunk_id=c["chunk_id"], text=c["text"],
                            metadata=c["metadata"], score=c["score"])
            for c in result["retrieved_chunks"]
        ]
        return RAGResponse(answer=result["answer"], query=result["query"], retrieved_chunks=chunks)
