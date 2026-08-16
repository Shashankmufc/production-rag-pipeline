"""
Retrieval layer.

Design: Retriever composes an embedder + a vector store (dependency
injection, not inheritance) so Phase 4's reranker and Phase 6's hybrid
search wrap this class rather than modifying it -- see RerankingRetriever
stub below, which follows the Decorator pattern: same interface, added
behavior.
"""

from abc import ABC, abstractmethod
from typing import Any
from embeddings.embed import BaseEmbedder
from vector_store.chromaDBstore import BaseVectorStore, RetrievedChunk


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError


class VectorRetriever(BaseRetriever):
    """Straightforward dense retrieval: embed the query, search the store."""

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.query(query_embedding, top_k=top_k, filters=filters)


class RerankingRetriever(BaseRetriever):
    """
    Decorator around any BaseRetriever: over-fetches candidates, then
    reorders them. Phase 4 will fill in a real cross-encoder in `_rerank`;
    for now this is a pass-through so the pipeline is wired end-to-end
    ahead of the actual reranking model being added.
    """

    def __init__(self, base_retriever: BaseRetriever, overfetch_multiplier: int = 3):
        self.base_retriever = base_retriever
        self.overfetch_multiplier = overfetch_multiplier

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        candidates = self.base_retriever.retrieve(query, top_k=top_k * self.overfetch_multiplier)
        reranked = self._rerank(query, candidates)
        return reranked[:top_k]

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # Phase 4 TODO: replace with a cross-encoder model
        # (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) scoring (query, candidate.text) pairs.
        return sorted(candidates, key=lambda c: c.score, reverse=True)
