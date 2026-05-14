"""Shared data models for retrieval results."""
from __future__ import annotations
from pydantic import BaseModel


class RetrievedNode(BaseModel):
    node_id: str
    label: str
    name: str
    text: str           # human-readable representation used for LLM context
    score: float = 0.0
    source: str = ""    # "vector", "graph", "bm25"
    metadata: dict = {}


def node_dict_to_text(node: dict) -> str:
    """Convert a raw Neo4j node dict to a readable text string."""
    label = node.get("label", "Node")
    name = node.get("name", node.get("path", node.get("id", "unknown")))
    parts = [f"[{label}] {name}"]

    for key in ("docstring", "description", "signature", "file_path", "path"):
        val = node.get(key)
        if val:
            parts.append(f"{key}: {val}")

    return " | ".join(parts)


def node_dict_to_retrieved(node: dict, score: float = 0.0, source: str = "") -> RetrievedNode:
    label = (node.get("label") or "Node")
    name = node.get("name") or node.get("path") or node.get("id") or "unknown"
    return RetrievedNode(
        node_id=node.get("id", ""),
        label=label,
        name=name,
        text=node_dict_to_text(node),
        score=score,
        source=source,
        metadata={k: v for k, v in node.items() if k not in ("embedding", "id")},
    )
