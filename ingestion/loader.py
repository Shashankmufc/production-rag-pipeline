"""
Ingestion layer.

Design: BaseLoader defines the contract every document loader must satisfy.
Adding a new source (DOCX, web page, Confluence, S3 bucket) later means
writing one new class here -- nothing downstream (chunker, embedder,
retriever) needs to change, because they only ever depend on `Document`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pypdf


@dataclass
class Document:
    """A single loaded document, prior to chunking."""
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Contract for all document loaders."""

    @abstractmethod
    def load(self, source: str) -> Document:
        """Load a single document from `source` (e.g. a file path)."""
        raise NotImplementedError

    def load_many(self, sources: list[str]) -> list[Document]:
        """Default batch implementation -- subclasses can override
        if their source supports a more efficient batch call."""
        return [self.load(s) for s in sources]


class PDFLoader(BaseLoader):
    """Loads text content from a PDF file using pypdf."""

    def load(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {source}")

        reader = pypdf.PdfReader(str(path))
        pages_text = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(page_text)

        full_text = "\n\n".join(pages_text)
        meta = reader.metadata or {}

        return Document(
            doc_id=path.stem,
            text=full_text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "title": getattr(meta, "title", None) or path.stem,
                "num_pages": len(reader.pages),
            },
        )


class TextLoader(BaseLoader):
    """Loads a plain .txt or .md file. Useful for quick tests without a PDF."""

    def load(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {source}")
        text = path.read_text(encoding="utf-8")
        return Document(
            doc_id=path.stem,
            text=text,
            metadata={"source": str(path), "filename": path.name, "title": path.stem},
        )
