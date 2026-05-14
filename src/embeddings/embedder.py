from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings


@lru_cache(maxsize=1)
def _get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using local sentence-transformers. Free, no API key."""
    if not texts:
        return []
    return _get_embedder().embed_documents(texts)


def embed_text(text: str) -> list[float]:
    return _get_embedder().embed_query(text)
