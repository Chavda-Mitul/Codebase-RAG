"""Langfuse tracing integration. Tracing is no-op when keys are not configured."""
from __future__ import annotations
from functools import lru_cache
from src.config import settings


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Return a Langfuse client, or None if keys are not configured."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        return None


def get_langchain_callback():
    """Return a LangChain-compatible Langfuse callback handler, or None."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        return None


def is_tracing_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)
