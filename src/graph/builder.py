from tqdm import tqdm
from src.graph.neo4j_client import Neo4jClient
from src.graph.schema import BaseNode, BaseRelationship
from src.embeddings.embedder import embed_texts
from src.config import settings


_VECTOR_LABELS = ["File", "Class", "Function", "Module", "Concept", "Community"]


def _node_text(node: BaseNode) -> str:
    """Produce a human-readable text representation of a node for embedding."""
    data = node.model_dump(exclude={"id", "label", "embedding"})
    parts = [f"{node.label}"]
    for k, v in data.items():
        if v:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


def build_graph(
    nodes: list[BaseNode],
    relationships: list[BaseRelationship],
    client: Neo4jClient,
) -> dict:
    # Create vector indexes
    for label in _VECTOR_LABELS:
        client.create_vector_index(label, dimensions=settings.embedding_dimensions)

    # Embed nodes in batches
    texts = [_node_text(n) for n in nodes]
    batch_size = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(embed_texts(batch))

    for node, emb in zip(nodes, all_embeddings):
        node.embedding = emb

    # Write nodes
    for node in tqdm(nodes, desc="Writing nodes"):
        client.create_node(node)

    # Write relationships
    for rel in tqdm(relationships, desc="Writing relationships"):
        client.create_relationship(rel)

    return {
        "nodes_written": len(nodes),
        "relationships_written": len(relationships),
    }
