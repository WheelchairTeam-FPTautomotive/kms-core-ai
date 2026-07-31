from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
