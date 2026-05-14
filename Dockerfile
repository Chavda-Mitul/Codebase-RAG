FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by tree-sitter and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer caching)
COPY pyproject.toml .
COPY src/ src/

# Install dependencies (no dev extras)
RUN uv sync --no-dev

# Copy the rest of the project
COPY server/ server/
COPY scripts/ scripts/

# Pre-download the embedding model so first request isn't slow
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
