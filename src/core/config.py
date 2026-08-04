from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:20128/v1"
    openai_model: str = "gemini-2.0-flash"
    chroma_path: str = "data/chroma_db"
    chroma_collection: str = "automotive_manuals"
    port: int = 8001
    log_level: str = "INFO"

    # Chunking Configuration
    chunk_window: int = 512
    chunk_overlap: int = 64

    # Data Paths
    docs_pdf_dir: str = "data/docs_pdf"
    docs_corpus_dir: str = "data/docs_corpus"

    # Embedding & Performance Settings
    use_local_embedding: bool = True
    embedding_model: str = "text-embedding-3-small"

    # AWS & OpenSearch Configuration
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    opensearch_endpoint: str = ""
    opensearch_index: str = "automotive-manuals"
    bedrock_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    vector_db_type: str = "chroma"  # 'chroma' or 'opensearch'

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
