import tiktoken
from dataclasses import dataclass, field

_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    if encoding_name not in _ENCODING_CACHE:
        _ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODING_CACHE[encoding_name]


@dataclass
class ChunkingConfig:
    """Configuration for sliding-window text chunking."""

    window: int = 512   # tokens per chunk
    overlap: int = 64   # overlapping tokens between adjacent chunks
    encoding_name: str = "cl100k_base"


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
    Split document text into overlapping chunks using BPE token boundaries.

    Args:
        text: Raw text content extracted from a PDF page or section.
        document_id: Hash identifier of the source document.
        document_name: Human-readable filename or title of the source document.
        page: Page number within the source document (1-based).
        config: Chunking parameters (window size and overlap).

    Returns:
        Ordered list of TextChunk objects ready for embedding and store upsert.
    """
    if config is None:
        config = ChunkingConfig()

    text = text.strip()
    if not text:
        return []

    encoding = _get_encoding(config.encoding_name)
    tokens = encoding.encode(text)

    if not tokens:
        return []

    window = config.window
    overlap = config.overlap
    step = max(1, window - overlap)

    chunks: list[TextChunk] = []
    chunk_index = 0

    for start_idx in range(0, len(tokens), step):
        end_idx = min(start_idx + window, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        chunk_text_str = encoding.decode(chunk_tokens)
        prefix_str = encoding.decode(tokens[:start_idx])
        start_char = len(prefix_str)
        end_char = start_char + len(chunk_text_str)

        chunk_metadata = {
            "document_id": document_id,
            "document_name": document_name,
            "page": page,
            "chunk_index": chunk_index,
            "token_count": len(chunk_tokens),
        }

        chunks.append(
            TextChunk(
                text=chunk_text_str,
                chunk_index=chunk_index,
                document_id=document_id,
                document_name=document_name,
                page=page,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_metadata,
            )
        )
        chunk_index += 1

        # If we reached the end of the tokens, break
        if end_idx >= len(tokens):
            break

    return chunks

