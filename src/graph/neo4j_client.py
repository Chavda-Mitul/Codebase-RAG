from neo4j import GraphDatabase
from src.config import settings
from src.graph.schema import BaseNode, BaseRelationship


class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def create_node(self, node: BaseNode) -> None:
        label = node.label
        props = node.model_dump(exclude={"label", "embedding"})
        props = {k: v for k, v in props.items() if v is not None}
        query = (
            f"MERGE (n:{label} {{id: $id}}) "
            "SET n += $props"
        )
        if node.embedding:
            query = (
                f"MERGE (n:{label} {{id: $id}}) "
                "SET n += $props "
                "SET n.embedding = $embedding"
            )
            with self.driver.session() as session:
                session.run(query, id=node.id, props=props, embedding=node.embedding)
        else:
            with self.driver.session() as session:
                session.run(query, id=node.id, props=props)

    def create_relationship(self, rel: BaseRelationship) -> None:
        query = (
            "MATCH (a {id: $src}), (b {id: $tgt}) "
            f"MERGE (a)-[r:{rel.type}]->(b) "
            "SET r += $props"
        )
        with self.driver.session() as session:
            session.run(query, src=rel.source_id, tgt=rel.target_id, props=rel.properties)

    def create_vector_index(self, label: str, property: str = "embedding", dimensions: int = 3072) -> None:
        index_name = f"{label.lower()}_vector_idx"
        query = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{property}) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}"
        )
        with self.driver.session() as session:
            session.run(query)

    def vector_search(self, query_embedding: list[float], label: str, top_k: int = 5) -> list[dict]:
        index_name = f"{label.lower()}_vector_idx"
        query = (
            f"CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding) "
            "YIELD node, score "
            "RETURN node, score"
        )
        with self.driver.session() as session:
            result = session.run(query, top_k=top_k, embedding=query_embedding)
            return [{"node": dict(r["node"]), "score": r["score"]} for r in result]

    def node_count(self) -> dict[str, int]:
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
            return {r["label"]: r["cnt"] for r in result}

    def relationship_count(self) -> dict[str, int]:
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt")
            return {r["type"]: r["cnt"] for r in result}

    def store_community(self, node) -> None:
        """Upsert a CommunityNode (creates Community vector index on first call)."""
        self.create_vector_index("Community", dimensions=384)
        self.create_node(node)

    def get_all_communities(self) -> list[dict]:
        """Return all Community nodes as plain dicts, ordered by name."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (c:Community) RETURN c ORDER BY c.name"
            )
            rows = []
            for r in result:
                node = dict(r["c"])
                node.pop("embedding", None)  # don't serialize large vectors
                rows.append(node)
            return rows

    def run_read_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute an arbitrary read-only Cypher query and return rows as plain dicts."""
        with self.driver.session() as session:
            result = session.run(cypher, **(params or {}))
            return [dict(r) for r in result]

    def get_subgraph(self, node_ids: list[str]) -> tuple[list[dict], list[dict]]:
        """Return (nodes, edges) for a 1-hop subgraph around the given node IDs."""
        if not node_ids:
            return [], []
        query = """
        UNWIND $ids AS seed_id
        MATCH (n {id: seed_id})
        OPTIONAL MATCH (n)-[r]-(neighbor)
        WHERE neighbor.id IS NOT NULL
        RETURN DISTINCT
            n.id AS n_id, labels(n)[0] AS n_label, n.name AS n_name,
            n.docstring AS n_doc, n.path AS n_path,
            neighbor.id AS nb_id, labels(neighbor)[0] AS nb_label,
            neighbor.name AS nb_name, neighbor.docstring AS nb_doc,
            neighbor.path AS nb_path, type(r) AS rel_type,
            startNode(r).id AS rel_src, endNode(r).id AS rel_tgt
        LIMIT 80
        """
        with self.driver.session() as session:
            rows = list(session.run(query, ids=node_ids))

        seen_nodes: dict[str, dict] = {}
        edges: list[dict] = []
        seen_edges: set[str] = set()

        for row in rows:
            for prefix in ("n", "nb"):
                nid = row[f"{prefix}_id"]
                if nid and nid not in seen_nodes:
                    seen_nodes[nid] = {
                        "id": nid,
                        "label": row[f"{prefix}_label"] or "Node",
                        "name": row[f"{prefix}_name"] or nid,
                        "docstring": row[f"{prefix}_doc"] or "",
                        "path": row[f"{prefix}_path"] or "",
                    }
            if row["rel_type"] and row["rel_src"] and row["rel_tgt"]:
                eid = f"{row['rel_src']}__{row['rel_type']}__{row['rel_tgt']}"
                if eid not in seen_edges:
                    seen_edges.add(eid)
                    edges.append({
                        "id": eid,
                        "source": row["rel_src"],
                        "target": row["rel_tgt"],
                        "label": row["rel_type"],
                    })

        return list(seen_nodes.values()), edges
