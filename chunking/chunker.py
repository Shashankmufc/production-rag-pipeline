"""
Chunking layer.

Design: BaseChunker is the contract. Phase 3 of the roadmap swaps in
SemanticChunker / MarkdownChunker as siblings implementing the same
interface -- retrieval and embedding code never needs to know which
chunker produced a Chunk.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from ingestion.loader import Document


@dataclass
class Chunk:
    """A single chunk of a document, ready for embedding."""
    chunk_id: str
    text: str
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Contract for all chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Splits text using a prioritized separator list, recursing into any
    piece still larger than chunk_size using the next separator.
    Mirrors LangChain's RecursiveCharacterTextSplitter behavior, implemented
    directly here so the project has no hard LangChain dependency for
    something this core.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, document: Document) -> list[Chunk]:
        raw_pieces = self._split(document.text, self.separators)
        merged = self._merge_with_overlap(raw_pieces)

        chunks = []
        for i, text in enumerate(merged):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{i}",
                    text=text.strip(),
                    doc_id=document.doc_id,
                    metadata={**document.metadata, "chunk_index": i},
                )
            )
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        sep, rest = separators[0], separators[1:]
        pieces = text.split(sep) if sep else list(text)

        result: list[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                result.append(piece)
            else:
                result.extend(self._split(piece, rest))
        return result

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedily pack small pieces back together up to chunk_size,
        carrying `chunk_overlap` characters forward into the next chunk."""
        merged: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                overlap_tail = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = f"{overlap_tail} {piece}".strip()
        if current:
            merged.append(current)
        return merged


class FixedSizeChunker(BaseChunker):
    """Naive baseline chunker -- kept around deliberately so Phase 2's
    eval harness has a 'bad' baseline to show measurable improvement against."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        i = 0
        idx = 0
        while i < len(text):
            piece = text[i:i + self.chunk_size]
            if piece.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{idx}",
                        text=piece.strip(),
                        doc_id=document.doc_id,
                        metadata={**document.metadata, "chunk_index": idx},
                    )
                )
                idx += 1
            i += step
        return chunks
