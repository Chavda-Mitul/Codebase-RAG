"""Parse markdown and plain text documents into chunks."""
from __future__ import annotations
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from src.config import settings


def _chunk_id(source: str, idx: int) -> str:
    raw = f"chunk:{source}:{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16] + "_chunk"


def parse_doc(doc: Document) -> list[Document]:
    """Split a document into chunks with enriched metadata."""
    language = doc.metadata.get("language", "text")
    path = doc.metadata.get("path", "unknown")

    if language == "markdown":
        splitter = MarkdownTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    chunks = splitter.create_documents(
        [doc.page_content],
        metadatas=[doc.metadata],
    )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = _chunk_id(path, i)
        chunk.metadata["doc_type"] = language
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks
