"""Orchestrates the full ingestion pipeline: load → parse → extract → graph."""
from __future__ import annotations
import time
from tqdm import tqdm
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from src.config import settings
from src.ingestion.loaders import clone_repo, discover_files
from src.ingestion.parsers.code_parser import parse_python_file
from src.ingestion.parsers.doc_parser import parse_doc
from src.ingestion.extractors.llm_extractor import extract_concepts_from_file
from src.graph.schema import BaseNode, BaseRelationship
from src.graph.builder import build_graph
from src.graph.neo4j_client import Neo4jClient
from src.graph.community_detection import detect_communities
from src.graph.community_summarizer import summarize_community


def run_ingestion(repo_url: str | None = None, local_path: str | None = None) -> dict:
    """Run the full ingestion pipeline. Provide either repo_url or local_path."""
    assert repo_url or local_path, "Provide repo_url or local_path"

    start = time.time()

    # Step 1: Load files
    if repo_url:
        print(f"Cloning {repo_url}...")
        root = clone_repo(repo_url)
        print(f"Cloned to {root}")
    else:
        root = local_path

    print("Discovering files...")
    all_docs = discover_files(root)
    py_docs = [d for d in all_docs if d.metadata["language"] == "python"]
    doc_docs = [d for d in all_docs if d.metadata["language"] != "python"]
    print(f"Found {len(py_docs)} Python files, {len(doc_docs)} doc files")

    all_nodes: list[BaseNode] = []
    all_rels: list[BaseRelationship] = []
    seen_node_ids: set[str] = set()

    def add_nodes(nodes: list[BaseNode], rels: list[BaseRelationship]) -> None:
        for n in nodes:
            if n.id not in seen_node_ids:
                all_nodes.append(n)
                seen_node_ids.add(n.id)
        all_rels.extend(rels)

    # Step 2: Parse Python files (structural)
    print("\nParsing Python files (tree-sitter)...")
    for doc in tqdm(py_docs, desc="Parsing"):
        nodes, rels = parse_python_file(
            doc.page_content,
            doc.metadata["path"],
            repo=repo_url or local_path or "",
        )
        add_nodes(nodes, rels)

    # Step 3: Parse doc files
    print("\nParsing documentation files...")
    for doc in tqdm(doc_docs, desc="Docs"):
        chunks = parse_doc(doc)
        # Doc chunks are stored as metadata; not yet adding as graph nodes in Phase 1

    # Step 4: LLM semantic extraction (Python files only, in batches)
    print(f"\nRunning LLM extraction on {len(py_docs)} Python files...")
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    batch_size = settings.llm_batch_size

    for i in tqdm(range(0, len(py_docs), batch_size), desc="LLM extraction"):
        batch = py_docs[i : i + batch_size]
        for doc in batch:
            nodes, rels = extract_concepts_from_file(doc, llm=llm)
            add_nodes(nodes, rels)

    # Step 5: Write to Neo4j
    print(f"\nWriting {len(all_nodes)} nodes and {len(all_rels)} relationships to Neo4j...")
    with Neo4jClient() as client:
        client.verify_connectivity()
        stats = build_graph(all_nodes, all_rels, client)

        # Step 6: Community detection + summarization
        print("\nDetecting communities...")
        communities = detect_communities(client)
        print(f"Found {len(communities)} communities. Summarising...")
        community_count = 0
        for group in tqdm(communities, desc="Communities"):
            node = summarize_community(group, client, llm)
            if node:
                client.store_community(node)
                community_count += 1

        node_counts = client.node_count()
        rel_counts = client.relationship_count()

    elapsed = time.time() - start
    summary = {
        **stats,
        "communities_written": community_count,
        "node_distribution": node_counts,
        "relationship_distribution": rel_counts,
        "elapsed_seconds": round(elapsed, 1),
        "python_files": len(py_docs),
        "doc_files": len(doc_docs),
    }
    return summary
