from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    llm_model: str = "llama-3.3-70b-versatile"

    # Langfuse observability (optional — tracing skipped if keys not set)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Reranker settings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # Ingestion settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    llm_batch_size: int = 5  # files per LLM extraction batch


settings = Settings()
