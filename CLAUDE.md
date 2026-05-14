# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Code-RAG** is an Adaptive GraphRAG + CRAG (Corrective RAG) pipeline for codebase Q&A. It ingests source repositories, builds a Neo4j knowledge graph, and answers questions via hybrid retrieval + LLM generation with automatic quality correction loops.

## Common Commands

```bash
# Environment setup
uv sync
docker-compose up -d          # starts Neo4j
cp .env.example .env          # then add GROQ_API_KEY

# Ingestion
uv run python scripts/ingest.py --repo https://github.com/owner/repo
uv run python scripts/ingest.py --local /path/to/repo

# Querying
uv run python scripts/ask.py -q "Your question"
uv run python scripts/ask.py   # interactive REPL

# API server
uv run python scripts/serve.py --reload

# Evaluation
uv run python scripts/eval.py --pipeline crag --limit 5
uv run python scripts/eval.py --compare

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_crag.py -v        # single test file
uv run pytest -k "test_hybrid" -v          # single test by name
```

## Architecture

### Data Flow

```
Git/Local Repo
  → Ingestion (tree-sitter AST + LLM extraction)
  → Neo4j Knowledge Graph (nodes: File, Class, Function, Module, Concept, Community)
  → Query Routing (simple / complex / conceptual)
  → Hybrid Retrieval (Vector + BM25 + Graph, merged via RRF)
  → CRAG Loop (grade → rewrite if needed → generate → hallucination check)
  → Answer
```

### Key Modules

| Path | Purpose |
|------|---------|
| `src/ingestion/` | Repo loading, tree-sitter parsing, LLM semantic extraction, graph writing |
| `src/graph/` | Neo4j client, schema definitions, community detection (Louvain), community summarization |
| `src/retrieval/` | Vector, BM25, graph retrievers + hybrid RRF merger + cross-encoder reranker |
| `src/crag/` | LangGraph state machine: retrieve → grade → rewrite → generate → check |
| `src/routing/` | Query classifier, sub-question decomposer, step-back abstraction, orchestration |
| `src/agents/` | Tool-calling agentic pipeline (search, run code, diagram, refactor) |
| `src/eval/` | Golden QA pairs, RAGAS-style metrics, pipeline comparison |
| `src/config.py` | Pydantic Settings (single source of truth for all config) |
| `server/` | FastAPI app with `/ask`, `/agent`, `/graph`, `/eval/*` endpoints |
| `scripts/` | CLI entry points for ingest, ask, serve, eval |

### Retrieval Design

- **RRF (Reciprocal Rank Fusion)** with `k=60` merges vector + BM25 + graph results — equal weight prevents single-retriever dominance.
- **CRAG corrective loop**: if >50% of retrieved docs fail LLM relevance grading, the query is rewritten and retrieval repeats. Max 3 iterations.
- **Reranker**: optional cross-encoder (`ms-marco-MiniLM-L6`) via `USE_RERANKER=true`.

### Query Routing Paths

- **Simple**: direct hybrid retrieval → generate
- **Complex**: decompose into sub-questions → answer each → synthesize
- **Conceptual**: step-back abstraction → enrich context → generate

### Tech Stack

- **LLM**: Groq (`llama-3.3-70b-versatile`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, local)
- **Graph DB**: Neo4j 5.x with APOC plugin (Docker)
- **Code Parsing**: `tree-sitter` + `tree-sitter-python`
- **Orchestration**: LangChain + LangGraph
- **API**: FastAPI 0.115+
- **Package manager**: `uv`

## Environment Variables

Required in `.env` (see `.env.example`):

```
GROQ_API_KEY=           # Required
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
LLM_MODEL=llama-3.3-70b-versatile
USE_RERANKER=true
# Optional observability:
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

All config is accessed via `src/config.py` (Pydantic Settings) — never read env vars directly.

## Testing

Tests in `tests/` use `pytest` + `pytest-asyncio`. Key test files:
- `test_crag.py` — CRAG state machine and routing logic
- `test_retrieval.py` — hybrid retrieval and RRF
- `test_routing.py` — router, decomposer, step-back
- `test_code_parser.py` — tree-sitter AST parsing

Neo4j must be running for integration tests. Unit tests mock the DB client.
