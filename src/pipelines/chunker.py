from dataclasses import dataclass, field


@dataclass
class ChunkingConfig:
    """Configuration for sliding-window text chunking."""

    window: int = 512   # tokens per chunk
    overlap: int = 64   # overlapping tokens between adjacent chunks


@dataclass
class TextChunk:
    """A single chunk of text extracted from a document page."""

    text: str
    chunk_index: int
    document_id: str
    document_name: str
    page: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    document_id: str,
    document_name: str,
    page: int = 0,
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """
    Split document text into overlapping chunks for vector embedding.

    Stub implementation — full tokenizer-aware logic to be integrated
    in Day 3-4 PDF parser pipeline.

    Args:
        text: Raw text content extracted from a PDF page or section.
        document_id: SHA-256 hash identifier of the source document.
        document_name: Human-readable filename of the source document.
        page: Page number within the source document (1-based).
        config: Chunking parameters. Defaults to ChunkingConfig().

    Returns:
        Ordered list of TextChunk objects ready for embedding and upsert
        into the ChromaDB collection.
    """
    if config is None:
        config = ChunkingConfig()

    # TODO (Day 3-4): Replace with tiktoken-based sliding window using
    # config.window and config.overlap.
    raise NotImplementedError(
        "chunk_text() is a stub — implement tokenizer-aware chunking in Day 3-4."
    )
