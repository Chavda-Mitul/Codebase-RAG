# Code-RAG — Adaptive GraphRAG + Corrective Agentic Pipeline

A production-grade RAG system for codebase analysis and architecture decision support.  
Combines **GraphRAG** (knowledge graph), **CRAG** (corrective agentic loop), **hybrid retrieval** (vector + BM25 + graph), and **smart query routing** (simple / complex / conceptual).

## Architecture

```
Question
  → Router (simple | complex | conceptual)
     ├─ simple     → CRAG pipeline
     ├─ complex    → Decompose → CRAG × N → Synthesize
     └─ conceptual → Step-back → CRAG with enriched context

CRAG pipeline:
  retrieve (hybrid: vector + BM25 + graph expansion)
  → grade documents (relevance check)
  → [correction loop if low relevance]
  → generate answer
  → check hallucination
  → check answer quality
  → [rewrite + retry if not useful]
  → final answer
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Graph DB | Neo4j 5.x |
| Orchestration | LangChain + LangGraph |
| Code Parsing | tree-sitter + tree-sitter-python |
| Eval | Custom RAGAS-style metrics (faithfulness, relevance, recall, precision) |
| Observability | Langfuse |
| API | FastAPI |
| Frontend | Next.js 15 + React Flow |

## Setup

### 1. Prerequisites
- Python 3.11+, `uv`, Docker Desktop, Node.js 18+

### 2. Install Python dependencies
```bash
uv sync
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env — add GROQ_API_KEY (required)
# LANGFUSE keys are optional (observability)
```

### 4. Start Neo4j
```bash
docker-compose up -d
# Neo4j Browser: http://localhost:7474  (user: neo4j, pass: password123)
```

### 5. Ingest a codebase
```bash
uv run python scripts/ingest.py --repo https://github.com/Chavda-Mitul/invoice-anomaly-detection
# Or use a local path:
uv run python scripts/ingest.py --local /path/to/repo
```

### 6. Run the frontend
```bash
# Terminal 1 — Python API
uv run python scripts/serve.py --reload

# Terminal 2 — Next.js UI
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

## CLI Usage

```bash
# Smart routing (recommended)
uv run python scripts/ask.py --question "What are the main sources of technical debt?"

# Interactive REPL
uv run python scripts/ask.py

# CRAG pipeline only
uv run python scripts/crag_query.py --question "What does AnomalyDetector do?"

# Run evaluation
uv run python scripts/eval.py --pipeline crag --limit 5
uv run python scripts/eval.py --compare   # naive vs crag vs routed
```

## Project Structure

```
src/
├── ingestion/      # Repo loader, tree-sitter parser, doc parser, LLM extractor
├── graph/          # Neo4j client, schema, graph builder
├── embeddings/     # HuggingFace sentence-transformers
├── retrieval/      # Vector, BM25, graph expansion, hybrid RRF merger
├── qa/             # Q&A chain, prompts
├── crag/           # LangGraph CRAG pipeline (state, nodes, graders, graph)
├── routing/        # Router, decomposer, step-back, routing pipeline
├── eval/           # Golden QA pairs, RAGAS metrics, eval runner
└── observability/  # Langfuse tracing

server/             # FastAPI server
frontend/           # Next.js 15 + React Flow UI
scripts/            # CLI entrypoints
tests/              # 77 unit tests
```

## Running Tests
```bash
uv run pytest tests/ -v
```
