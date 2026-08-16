"""
Embeddings layer.

Design: BaseEmbedder is the contract. Phase 5 (embedding model bake-off)
adds OpenAIEmbedder, VoyageEmbedder, etc. as siblings -- vector_store and
retrieval code depend only on `embed_documents` / `embed_query`, never on
a specific model.

Note the deliberate query/passage asymmetry: some models (E5, BGE) expect
different prefixes for queries vs. documents. Baking that distinction into
the interface now avoids a silent retrieval-quality bug later.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """Contract for all embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local, free, CPU-friendly embedder using sentence-transformers.
    Default model (all-MiniLM-L6-v2) is deliberately small so Phase 1
    runs fast without a GPU or API key -- swap for BGE/E5 in Phase 5.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: np.ndarray = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding: np.ndarray = self._model.encode(
            [text], show_progress_bar=False, convert_to_numpy=True
        )[0]
        return embedding.tolist()
