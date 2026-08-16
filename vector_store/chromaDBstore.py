"""
Vector store layer.

Design: BaseVectorStore is the contract. Phase 6 (metadata filtering +
hybrid search) extends this interface with a `filters` argument and a
sparse-score fusion step -- the retrieval layer calls the same `query`
method regardless, so swapping Chroma for Qdrant/Weaviate later is a
one-class change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import chromadb
from chunking.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float  # similarity score, higher = more relevant


class BaseVectorStore(ABC):
    """Contract for all vector store backends."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError


class ChromaDBStore(BaseVectorStore):
    """Persistent local ChromaDB-backed vector store."""

    def __init__(self, collection_name: str = "rag_collection", persist_dir: str = "./chroma_db"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[self._sanitize_metadata(c.metadata) for c in chunks],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,  # Chroma's native metadata pre-filter; wired up properly in Phase 6
        )

        retrieved = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]  # cosine distance, lower = closer

        for chunk_id, text, meta, distance in zip(ids, docs, metadatas, distances):
            similarity = 1 - distance  # convert distance -> similarity score
            retrieved.append(
                RetrievedChunk(chunk_id=chunk_id, text=text, metadata=meta, score=similarity)
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Chroma only accepts str/int/float/bool metadata values."""
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v if v is not None else ""
            else:
                clean[k] = str(v)
        return clean
