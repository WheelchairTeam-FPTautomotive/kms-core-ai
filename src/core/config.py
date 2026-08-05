from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    # MODIFIED: single definition (was duplicated below — broke Ruff PIE794 / AI CI)
    openai_base_url: str = "http://localhost:11434/v1"  # Ollama OpenAI-compatible
    openai_model: str = "llama3.2"
    chroma_path: str = "data/chroma_db"
    chroma_collection: str = "automotive_manuals"
    port: int = 8001
    log_level: str = "INFO"

    chunk_window: int = 512
    chunk_overlap: int = 64

    docs_pdf_dir: str = "data/docs_pdf"
    docs_corpus_dir: str = "data/docs_corpus"

    use_local_embedding: bool = True
    embedding_model: str = "text-embedding-3-small"
    aws_region: str = "ap-southeast-1" # Singapore, ap-southeast-2 = Sydney
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    opensearch_endpoint: str = ""
    opensearch_index: str = "automotive-manuals"
    bedrock_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    vector_db_type: str = "chroma"  # 'chroma' or 'opensearch'

    llm_provider: str = "none"  # none | bedrock | openai_compatible
    system_prompt: str = (
        "Bạn là trợ lý giọng nói trên xe hơi cho tài xế. "
        "Trả lời ngắn gọn, rõ ràng, dễ đọc bằng TTS, DỰA HOÀN TOÀN VÀO ngữ cảnh tài liệu được cung cấp. "
        "Không trích dẫn đường dẫn file, mã spec, hay danh mục thư mục. "
        "Nếu ngữ cảnh không đủ để trả lời, hãy nói lịch sự rằng không tìm thấy thông tin trong tài liệu kỹ thuật. "
        "Ưu tiên tiếng Việt trừ khi người dùng hỏi bằng tiếng Anh."
    )
    llm_temperature: float = 0.0
    llm_max_tokens: int = 400
    llm_top_p: float = 0.9
    rag_top_k: int = 3
    rag_context_chars: int = 2400
    rag_max_distance: float = 1.15
    free_talk_system_prompt: str = (
        "Bạn là trợ lý giọng nói thân thiện trên xe. "
        "Trả lời ngắn gọn, lịch sự, dễ đọc bằng TTS cho chào hỏi và trò chuyện chung. "
        "KHÔNG bịa quy trình vận hành xe, thông số kỹ thuật, hay hướng dẫn từ manual. "
        "Nếu người dùng hỏi về điều khiển xe, bảo dưỡng, an toàn, hoặc thao tác kỹ thuật, "
        "hãy nhắc họ diễn đạt lại như câu hỏi tra cứu tài liệu (ví dụ: 'cách bật HVAC theo manual'), "
        "đừng tự bịa các bước. "
        "Ưu tiên tiếng Việt trừ khi người dùng hỏi bằng tiếng Anh."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
