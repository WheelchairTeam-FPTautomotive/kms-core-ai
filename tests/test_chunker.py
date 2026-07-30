import pytest
import tiktoken
from pipelines.chunker import ChunkingConfig, chunk_text


def test_chunk_text_empty():
    chunks = chunk_text("", "doc1", "Doc 1")
    assert chunks == []


def test_chunk_text_small_input():
    text = "Hệ thống điều hòa không khí tự động trên xe hơi."
    config = ChunkingConfig(window=512, overlap=64)
    chunks = chunk_text(text, "doc123", "User Manual", page=1, config=config)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].document_id == "doc123"
    assert chunks[0].document_name == "User Manual"
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0


def test_chunk_text_sliding_window_overlap():
    encoding = tiktoken.get_encoding("cl100k_base")
    # Generate long text of 1000 tokens
    words = ["automotive", "manual", "system", "engine", "brake", "sensor", "battery", "hvac", "adas"]
    long_text = " ".join(words * 200)

    total_tokens = len(encoding.encode(long_text))
    assert total_tokens > 600

    config = ChunkingConfig(window=512, overlap=64)
    chunks = chunk_text(long_text, "doc_long", "Long Manual", page=5, config=config)

    assert len(chunks) >= 2
    # Verify token count of first chunk
    first_chunk_tokens = len(encoding.encode(chunks[0].text))
    assert first_chunk_tokens <= 512

    # Check overlap
    tokens_c1 = encoding.encode(chunks[0].text)
    tokens_c2 = encoding.encode(chunks[1].text)
    # The last 64 tokens of c1 should equal the first 64 tokens of c2
    assert tokens_c1[-64:] == tokens_c2[:64]
