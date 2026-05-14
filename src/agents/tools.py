"""Agentic tool implementations for the Code-RAG agent pipeline."""
from __future__ import annotations
import subprocess
import sys
from langchain_groq import ChatGroq
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.retrieval.models import RetrievedNode
from src.qa.chain import _format_context


def search_codebase(query: str, client: Neo4jClient, top_k: int = 5) -> tuple[str, list[RetrievedNode]]:
    """Search the knowledge graph and return (formatted_text, nodes)."""
    nodes = hybrid_retrieve(query, client, top_k=top_k)
    return _format_context(nodes), nodes


def run_code_snippet(code: str, timeout: int = 5) -> str:
    """Execute a Python snippet in a subprocess with a timeout. Returns stdout+stderr."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()
        if errors:
            output = f"{output}\n[stderr]\n{errors}".strip()
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[Error] Code execution timed out after {timeout}s"
    except Exception as e:
        return f"[Error] {e}"


def generate_mermaid_diagram(entity_names_str: str, client: Neo4jClient) -> str:
    """Build a Mermaid diagram for named entities and their direct relationships."""
    names = [n.strip() for n in entity_names_str.split(",") if n.strip()]
    if not names:
        return "graph LR\n  (no entities provided)"

    rows = client.run_read_query(
        """
        UNWIND $names AS nm
        MATCH (a) WHERE toLower(a.name) CONTAINS toLower(nm)
        OPTIONAL MATCH (a)-[r]-(b)
        WHERE b.name IS NOT NULL
        RETURN DISTINCT a.name AS src, type(r) AS rel, b.name AS tgt, labels(a)[0] AS src_type
        LIMIT 40
        """,
        {"names": names},
    )

    lines: list[str] = ["graph LR"]
    seen: set[str] = set()
    for row in rows:
        src = str(row.get("src") or "").replace('"', "'")
        tgt = str(row.get("tgt") or "").replace('"', "'")
        rel = str(row.get("rel") or "").replace('"', "'")
        if src and tgt and rel:
            edge = f'  {_mermaid_id(src)}["{src}"] --{rel}--> {_mermaid_id(tgt)}["{tgt}"]'
            if edge not in seen:
                seen.add(edge)
                lines.append(edge)

    if len(lines) == 1:
        # No relationships found — just show the nodes
        for name in names:
            lines.append(f'  {_mermaid_id(name)}["{name}"]')

    return "\n".join(lines)


def suggest_refactors(function_name: str, client: Neo4jClient, llm: ChatGroq) -> str:
    """Retrieve a function and its callers, then ask the LLM for refactoring suggestions."""
    rows = client.run_read_query(
        """
        MATCH (f:Function) WHERE toLower(f.name) CONTAINS toLower($name)
        OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
        RETURN f.name AS fn_name, f.docstring AS docstring, f.signature AS sig,
               f.file_path AS fpath, collect(caller.name) AS callers
        LIMIT 1
        """,
        {"name": function_name},
    )
    if not rows:
        return f"No function matching '{function_name}' found in the knowledge graph."

    row = rows[0]
    fn_context = (
        f"Function: {row.get('fn_name')}\n"
        f"File: {row.get('fpath') or 'unknown'}\n"
        f"Signature: {row.get('sig') or 'N/A'}\n"
        f"Docstring: {row.get('docstring') or 'N/A'}\n"
        f"Called by: {', '.join(row.get('callers') or []) or 'nobody'}"
    )

    prompt = (
        f"You are a senior software engineer reviewing a codebase.\n\n"
        f"{fn_context}\n\n"
        "Suggest 3-5 concrete, actionable refactoring improvements for this function. "
        "Focus on: naming clarity, single responsibility, testability, error handling, and code smells."
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


def _mermaid_id(name: str) -> str:
    """Create a safe Mermaid node ID from a name."""
    return "".join(c if c.isalnum() else "_" for c in name)[:30]
