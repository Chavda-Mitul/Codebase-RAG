"""LLM-based summarization of node communities into CommunityNode objects."""
from __future__ import annotations
import hashlib
from langchain_groq import ChatGroq
from src.graph.neo4j_client import Neo4jClient
from src.graph.schema import CommunityNode
from src.embeddings.embedder import embed_text

_SUMMARY_PROMPT = """You are a software architect analysing a codebase knowledge graph.

Below is a group of related code entities that form a connected component in the graph.
Summarise what this group of entities does together as a coherent module or component.
Be concise (2-4 sentences). Include: the primary purpose, key classes/functions involved,
and any notable patterns or responsibilities.

Entities:
{entity_list}

Summary:"""

_NAME_PROMPT = """Given this community summary, provide a short descriptive name (3-6 words, title case):

{summary}

Name:"""


def _community_id(member_ids: list[str]) -> str:
    digest = hashlib.md5(",".join(sorted(member_ids)).encode()).hexdigest()[:12]
    return f"community_{digest}"


def summarize_community(
    member_ids: list[str],
    client: Neo4jClient,
    llm: ChatGroq,
) -> CommunityNode | None:
    """Fetch node details for community members, summarise with LLM, return CommunityNode."""
    if not member_ids:
        return None

    rows = client.run_read_query(
        """
        UNWIND $ids AS nid
        MATCH (n {id: nid})
        RETURN labels(n)[0] AS label, n.name AS name,
               n.docstring AS docstring, n.description AS description,
               n.signature AS sig
        """,
        {"ids": member_ids},
    )

    if not rows:
        return None

    entity_lines = []
    for r in rows:
        parts = [f"[{r.get('label') or 'Node'}] {r.get('name') or '(unnamed)'}"]
        for key in ("docstring", "description", "sig"):
            val = r.get(key)
            if val:
                parts.append(str(val)[:120])
        entity_lines.append(" | ".join(parts))

    entity_list = "\n".join(entity_lines[:40])  # cap for LLM context

    try:
        summary_resp = llm.invoke([
            {"role": "user", "content": _SUMMARY_PROMPT.format(entity_list=entity_list)},
        ])
        summary = summary_resp.content.strip()

        name_resp = llm.invoke([
            {"role": "user", "content": _NAME_PROMPT.format(summary=summary)},
        ])
        name = name_resp.content.strip().strip('"').strip("'")[:80]
    except Exception:
        # Fallback: use member count as name
        summary = f"A group of {len(member_ids)} related code entities."
        name = f"Module ({len(member_ids)} nodes)"

    node_id = _community_id(member_ids)
    text = f"[Community] {name} | {summary}"
    embedding = embed_text(text)

    return CommunityNode(
        id=node_id,
        name=name,
        members=member_ids,
        summary=summary,
        embedding=embedding,
    )
