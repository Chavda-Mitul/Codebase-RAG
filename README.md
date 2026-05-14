# Code-RAG — Adaptive GraphRAG + Corrective Agentic Pipeline

> A production-grade AI system that ingests any GitHub repository or local codebase, builds a semantic knowledge graph in Neo4j, and answers natural-language questions about code with automatic self-correction, hallucination detection, and explainability.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=nextdotjs)](https://nextjs.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat&logo=neo4j)](https://neo4j.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Loop-1C3C3C?style=flat)](https://langchain-ai.github.io/langgraph/)

---

## What It Does

Traditional RAG systems retrieve text chunks and generate answers — they have no awareness of code structure, no way to verify retrieval quality, and no mechanism to fix bad answers. **Code-RAG** solves all three:

| Problem | How Code-RAG Solves It |
|---|---|
| Flat chunk retrieval loses code structure | Builds a **Neo4j knowledge graph** of files, classes, functions, and their relationships |
| Single retrieval strategy misses context | **Hybrid RRF** merges vector search + BM25 keyword + graph traversal |
| Bad retrievals produce hallucinated answers | **CRAG corrective loop** grades each document and rewrites the query if retrieval quality is low |
| One-size-fits-all querying | **Smart router** classifies queries and picks simple / complex / conceptual pipelines |
| No way to measure answer quality | **RAGAS-style eval** scores faithfulness, relevance, recall, and precision per pipeline |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
│                                                                     │
│  Git / Local Repo                                                   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │  Repo Loader│───▶│ tree-sitter AST  │───▶│  LLM Extractor    │  │
│  │  (Git/Local)│    │  Code Parser     │    │  (Groq LLaMA-3.3) │  │
│  └─────────────┘    └──────────────────┘    └────────┬──────────┘  │
│                                                      │              │
│                                                      ▼              │
│                                          ┌───────────────────────┐ │
│                                          │  Neo4j Knowledge Graph│ │
│                                          │  File ─ Class         │ │
│                                          │   │       │           │ │
│                                          │  Func  Module         │ │
│                                          │   │       │           │ │
│                                          │  Concept─Community    │ │
│                                          └───────────┬───────────┘ │
└──────────────────────────────────────────────────────│─────────────┘
                                                       │
┌──────────────────────────────────────────────────────│─────────────┐
│                         QUERY PIPELINE                │             │
│                                                       ▼             │
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     SMART QUERY ROUTER                       │  │
│  │              (LLM-based query classification)                 │  │
│  └──────┬──────────────────┬──────────────────────┬─────────────┘  │
│         │                  │                      │                 │
│         ▼                  ▼                      ▼                 │
│    [SIMPLE]           [COMPLEX]             [CONCEPTUAL]           │
│         │           Sub-question            Step-back              │
│         │           Decomposer              Abstraction            │
│         │                  │                      │                │
│         └──────────────────┴──────────────────────┘               │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   HYBRID RETRIEVAL (RRF)                     │   │
│  │                                                              │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │   │
│  │   │Vector Search │  │ BM25 Keyword │  │  Graph Traversal │  │   │
│  │   │(MiniLM-L6-v2)│  │   Search     │  │  (Neo4j Cypher)  │  │   │
│  │   └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │   │
│  │          └─────────────────┴───────────────────┘            │   │
│  │                            │                                 │   │
│  │                    Reciprocal Rank Fusion                    │   │
│  │                         (k = 60)                             │   │
│  │                            │                                 │   │
│  │                   Cross-Encoder Reranker                     │   │
│  │                   (ms-marco-MiniLM-L6)                       │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    CRAG CORRECTIVE LOOP                      │   │
│  │                                                              │   │
│  │  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐  │   │
│  │  │Retrieve │───▶│  Grade   │───▶│ Generate │───▶│ Check  │  │   │
│  │  │  Docs   │    │  Docs    │    │  Answer  │    │Halluc. │  │   │
│  │  └─────────┘    └────┬─────┘    └──────────┘    └───┬────┘  │   │
│  │                      │ <50% relevant                 │ fail  │   │
│  │                      ▼                               ▼       │   │
│  │               ┌─────────────┐               ┌──────────────┐ │   │
│  │               │Query Rewrite│               │ Answer Check │ │   │
│  │               │  + Retry    │               │  + Rewrite   │ │   │
│  │               └─────────────┘               └──────────────┘ │   │
│  │                   max 3 iterations                            │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│                         Final Answer                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
         ┌──────────┐
         │ Git Repo │
         └────┬─────┘
              │  clone / read
              ▼
    ┌──────────────────┐        ┌────────────────────────┐
    │  tree-sitter AST │        │   LLM Semantic Extract │
    │  (Python parser) │───────▶│   - Concepts           │
    │  - Files         │        │   - Summaries          │
    │  - Classes       │        │   - Relationships      │
    │  - Functions     │        └───────────┬────────────┘
    │  - Imports       │                    │
    └──────────────────┘                    │
              │                             │
              └─────────────┬───────────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │      Neo4j Graph DB      │
             │                          │
             │  (File)──CONTAINS──►(Class)
             │     │                    │
             │  IMPORTS            HAS_METHOD
             │     │                    │
             │  (Module)        (Function)
             │     │                    │
             │  RELATES_TO       USES_CONCEPT
             │     │                    │
             │  (Concept)◄──────(Community)
             └────────────┬─────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌──────────────────┐   ┌─────────────────────┐
    │  Vector Index    │   │   BM25 Index         │
    │  (384-dim embed) │   │   (keyword search)   │
    └──────────────────┘   └─────────────────────┘
```

---

## Query Routing Logic

```
                    ┌──────────────┐
                    │ User Question│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LLM Router  │
                    │  Classifier  │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   ┌─────────────┐  ┌────────────┐  ┌──────────────┐
   │   SIMPLE    │  │  COMPLEX   │  │  CONCEPTUAL  │
   │             │  │            │  │              │
   │ "What does  │  │ "Compare   │  │ "Why is this │
   │  foo() do?" │  │  X vs Y"   │  │  designed    │
   │             │  │            │  │  this way?"  │
   └──────┬──────┘  └─────┬──────┘  └──────┬───────┘
          │               │                │
          │        ┌──────▼──────┐  ┌──────▼───────┐
          │        │Sub-question │  │  Step-back   │
          │        │ Decomposer  │  │ Abstraction  │
          │        │ Q1,Q2,Q3..  │  │ (higher-level│
          │        └──────┬──────┘  │  context)    │
          │               │         └──────┬───────┘
          │        ┌──────▼──────┐         │
          │        │ Answer each │         │
          │        │ sub-question│         │
          │        │ individually│         │
          │        └──────┬──────┘         │
          │        ┌──────▼──────┐         │
          │        │ Synthesize  │         │
          │        │  answers    │         │
          │        └──────┬──────┘         │
          └───────────────┴────────────────┘
                          │
                    Final Answer
```

---

## CRAG Corrective Loop (State Machine)

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    LangGraph State Machine                   │
  │                                                             │
  │   START                                                     │
  │     │                                                       │
  │     ▼                                                       │
  │  ┌──────────┐    docs retrieved                             │
  │  │ retrieve │─────────────────────────────────────────┐    │
  │  └──────────┘                                         │    │
  │                                                       ▼    │
  │                                              ┌─────────────┐│
  │                                              │grade_docs   ││
  │                                              │(LLM grades  ││
  │                                              │ each doc    ││
  │                                              │ yes/no)     ││
  │                                              └──────┬──────┘│
  │                                                     │        │
  │                              ┌──────────────────────┤        │
  │                              │                      │        │
  │                    >50% relevant           <50% relevant     │
  │                              │                      │        │
  │                              ▼                      ▼        │
  │                       ┌──────────┐         ┌──────────────┐ │
  │                       │ generate │         │query_rewrite │ │
  │                       │  answer  │         │ (rewrite →   │ │
  │                       └────┬─────┘         │  retrieve)   │ │
  │                            │               └──────────────┘ │
  │                            ▼                   max 3 iters  │
  │                   ┌─────────────────┐                       │
  │                   │hallucination_   │                       │
  │                   │check (grounded?)│                       │
  │                   └────────┬────────┘                       │
  │                            │                                │
  │                ┌───────────┴───────────┐                   │
  │                │                       │                   │
  │           grounded                not grounded             │
  │                │                       │                   │
  │                ▼                       ▼                   │
  │       ┌─────────────┐         ┌─────────────────┐         │
  │       │answer_check │         │   regenerate    │         │
  │       │ (useful?)   │         │  (back to gen.) │         │
  │       └──────┬──────┘         └─────────────────┘         │
  │              │                                             │
  │       ┌──────┴──────┐                                     │
  │       │             │                                      │
  │    useful        not useful                                │
  │       │             │                                      │
  │       ▼             ▼                                      │
  │     END       rewrite_query                                │
  │               (loop again)                                 │
  └─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | Groq `llama-3.3-70b-versatile` | Generation, grading, routing, extraction |
| Embeddings | `all-MiniLM-L6-v2` (local) | 384-dim semantic vectors, no API cost |
| Graph DB | Neo4j 5.x + APOC | Knowledge graph storage and traversal |
| Orchestration | LangChain + LangGraph | CRAG state machine and chain composition |
| Code Parsing | tree-sitter + tree-sitter-python | AST-level code structure extraction |
| Reranker | `ms-marco-MiniLM-L6` cross-encoder | Precision reranking of retrieved docs |
| Community Detection | Louvain algorithm (networkx) | Groups related code concepts together |
| Eval | Custom RAGAS-style metrics | Faithfulness, relevance, recall, precision |
| Observability | Langfuse | LLM call tracing and cost tracking |
| API | FastAPI 0.115+ | REST endpoints for query, eval, graph |
| Frontend | Next.js 15 + React Flow + Recharts | Chat UI, graph viz, eval dashboard |
| Package Manager | `uv` | Fast Python dependency management |

---

## Key Design Decisions

**Why Neo4j over pure vector search?**
Code has explicit structure — functions call other functions, classes inherit from each other, files import modules. A graph DB captures these relationships. Graph traversal retrieval (2-hop expansion from matched nodes) surfaces context that embedding similarity alone would miss.

**Why Reciprocal Rank Fusion (RRF)?**
No single retriever dominates for all query types. RRF with `k=60` merges ranked lists from vector, BM25, and graph retrievers without needing to tune relative weights — it naturally promotes documents that rank well across multiple strategies.

**Why CRAG instead of naive RAG?**
Naive RAG blindly passes retrieved docs to the LLM. CRAG grades each retrieved document for relevance before generation. If quality is low, the query is rewritten and retrieval repeats (max 3 iterations). After generation, a hallucination check verifies the answer is grounded in the retrieved context.

**Why query routing?**
"What does `foo()` do?" and "Compare the authentication approaches across all services" require fundamentally different retrieval strategies. Routing to simple / complex / conceptual pipelines prevents over-engineering simple queries and under-serving complex ones.

---

## Project Structure

```
Code-RAG/
├── src/
│   ├── config.py               # Pydantic Settings — single source of truth
│   ├── ingestion/
│   │   ├── loaders.py          # Git clone + local repo reader
│   │   ├── pipeline.py         # Orchestrates full ingestion
│   │   ├── parsers/
│   │   │   ├── code_parser.py  # tree-sitter AST → structured nodes
│   │   │   └── doc_parser.py   # Markdown / docstring parser
│   │   └── extractors/
│   │       └── llm_extractor.py # LLM-based semantic concept extraction
│   ├── graph/
│   │   ├── neo4j_client.py     # Neo4j driver wrapper
│   │   ├── schema.py           # Node/relationship type definitions
│   │   ├── builder.py          # Writes parsed nodes into graph
│   │   ├── community_detection.py  # Louvain community detection
│   │   └── community_summarizer.py # LLM summaries for communities
│   ├── embeddings/
│   │   └── embedder.py         # HuggingFace sentence-transformers
│   ├── retrieval/
│   │   ├── vector_retriever.py # FAISS / Neo4j vector index search
│   │   ├── bm25_retriever.py   # BM25 keyword retriever
│   │   ├── graph_retriever.py  # Neo4j Cypher graph traversal
│   │   ├── hybrid_retriever.py # RRF merger across all retrievers
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   └── models.py           # Shared retrieval data models
│   ├── crag/
│   │   ├── state.py            # LangGraph state schema
│   │   ├── nodes.py            # retrieve, grade, generate, check nodes
│   │   ├── graders.py          # LLM-based relevance / hallucination graders
│   │   └── graph.py            # LangGraph compiled state machine
│   ├── routing/
│   │   ├── router.py           # Query type classifier (simple/complex/conceptual)
│   │   ├── decomposer.py       # Sub-question decomposition for complex queries
│   │   ├── step_back.py        # Step-back abstraction for conceptual queries
│   │   └── pipeline.py         # Routing orchestration
│   ├── qa/
│   │   ├── chain.py            # LangChain Q&A chain
│   │   └── prompts.py          # All prompt templates
│   ├── agents/
│   │   ├── agent_graph.py      # Tool-calling agentic pipeline
│   │   ├── tools.py            # Search, run code, diagram, refactor tools
│   │   ├── state.py            # Agent state schema
│   │   └── memory.py           # Conversation memory
│   ├── eval/
│   │   ├── golden_qa.py        # Golden QA pair definitions
│   │   ├── metrics.py          # RAGAS-style scoring
│   │   ├── runner.py           # Eval pipeline runner
│   │   └── store.py            # SQLite result persistence
│   └── observability/
│       ├── tracer.py           # Langfuse tracing wrapper
│       └── langfuse_handler.py # LangChain callback handler
├── server/
│   ├── main.py                 # FastAPI app + /ask /agent /graph endpoints
│   ├── eval_routes.py          # /eval/* endpoints
│   └── models.py               # Pydantic request/response models
├── frontend/
│   ├── app/                    # Next.js 15 App Router pages
│   ├── components/             # ChatPanel, GraphVisualization, EvalDashboard...
│   ├── lib/api.ts              # Typed API client
│   └── types/index.ts          # Shared TypeScript types
├── scripts/
│   ├── ingest.py               # CLI: ingest a repo
│   ├── ask.py                  # CLI: query with smart routing
│   ├── crag_query.py           # CLI: CRAG pipeline only
│   ├── serve.py                # CLI: start FastAPI server
│   └── eval.py                 # CLI: run evaluation
├── tests/                      # pytest test suite (unit + integration)
├── docker-compose.yml          # Neo4j + APOC
├── pyproject.toml
└── .env.example
```

---

## Setup

### Prerequisites
- Python 3.11+, [`uv`](https://github.com/astral-sh/uv), Docker Desktop, Node.js 18+

### 1. Install Python dependencies
```bash
uv sync
```

### 2. Configure environment
```bash
cp .env.example .env
# Add your GROQ_API_KEY (required)
# LANGFUSE keys are optional (observability)
```

### 3. Start Neo4j
```bash
docker-compose up -d
# Neo4j Browser → http://localhost:7474  (neo4j / password123)
```

### 4. Ingest a codebase
```bash
# From GitHub
uv run python scripts/ingest.py --repo https://github.com/owner/repo

# From local path
uv run python scripts/ingest.py --local /path/to/repo
```

### 5. Start the API + frontend
```bash
# Terminal 1 — Python API
uv run python scripts/serve.py --reload

# Terminal 2 — Next.js UI
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

---

## CLI Usage

```bash
# Smart routing (recommended) — router picks the best pipeline
uv run python scripts/ask.py -q "What are the main sources of technical debt?"

# Interactive REPL
uv run python scripts/ask.py

# CRAG pipeline directly
uv run python scripts/crag_query.py -q "What does AnomalyDetector do?"

# Evaluation
uv run python scripts/eval.py --pipeline crag --limit 5
uv run python scripts/eval.py --compare   # naive vs crag vs routed
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Query with smart routing + CRAG |
| `POST` | `/agent` | Tool-calling agentic pipeline |
| `GET` | `/graph/stats` | Node and relationship counts |
| `GET` | `/graph/neighborhood/{node_id}` | Subgraph for a node |
| `POST` | `/eval/run` | Run evaluation on a pipeline |
| `GET` | `/eval/runs` | List all eval runs |
| `GET` | `/eval/compare` | Pipeline comparison data |

---

## Evaluation Metrics

The system evaluates itself using four RAGAS-inspired metrics:

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) |
| **Answer Relevance** | Does the answer actually address the question asked? |
| **Context Recall** | Did retrieval surface the docs needed to answer? |
| **Context Precision** | Were the retrieved docs relevant (no noise)? |

Run a full pipeline comparison:
```bash
uv run python scripts/eval.py --compare
```

---

## Running Tests

```bash
# Full suite
uv run pytest tests/ -v

# By module
uv run pytest tests/test_crag.py -v
uv run pytest tests/test_retrieval.py -v
uv run pytest tests/test_routing.py -v

# By name
uv run pytest -k "test_hybrid" -v
```

> Neo4j must be running for integration tests. Unit tests mock the DB client.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key for LLaMA-3.3 |
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | No | `password123` | Neo4j password |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBEDDING_DIMENSIONS` | No | `384` | Embedding vector size |
| `LLM_MODEL` | No | `llama-3.3-70b-versatile` | Groq model ID |
| `USE_RERANKER` | No | `true` | Enable cross-encoder reranker |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse observability (optional) |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse observability (optional) |
