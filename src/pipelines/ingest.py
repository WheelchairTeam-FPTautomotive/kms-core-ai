import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from pipelines.chunker import ChunkingConfig, TextChunk, chunk_text
from utils.logger import setup_logger

logger = setup_logger("kms_pdf_ingest")


def load_document_mapping(corpus_dir: str | Path) -> dict[str, str]:
    """
    Load mapping.json to resolve SHA-256 document hash IDs to human-readable titles.
    """
    mapping_path = Path(corpus_dir) / "mapping.json"
    if not mapping_path.exists():
        logger.warning(f"mapping.json not found at {mapping_path}")
        return {}

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            raw_map: dict[str, list[str]] = json.load(f)

        hash_to_name: dict[str, str] = {}
        for doc_hash, name_list in raw_map.items():
            if name_list:
                hash_to_name[doc_hash] = name_list[0]
            else:
                hash_to_name[doc_hash] = doc_hash

        logger.info(f"Loaded {len(hash_to_name)} document mappings from mapping.json")
        return hash_to_name
    except Exception as e:
        logger.error(f"Failed to load mapping.json: {e}")
        return {}


def calculate_file_hash(filepath: str | Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_section_header(page_text: str, default_page: int) -> str:
    """Extract heuristic section header from page text or fallback to page number."""
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return f"Trang {default_page}"

    # Check top 3 lines for chapter / section indicators
    for line in lines[:3]:
        lower_line = line.lower()
        if any(
            kw in lower_line
            for kw in ["chương", "chapter", "mục", "section", "srs", "part", "bài"]
        ):
            return line[:100]

    # If first line looks like a title (short & capitalized)
    first_line = lines[0]
    if len(first_line) < 80 and (first_line.isupper() or first_line.istitle()):
        return first_line

    return f"Trang {default_page}"


def process_pdf_file(
    filepath: Path,
    doc_id: str,
    doc_name: str,
    config: ChunkingConfig,
) -> list[TextChunk]:
    """Parse a single PDF file page-by-page and chunk text."""
    chunks: list[TextChunk] = []
    try:
        reader = PdfReader(str(filepath))
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            section = extract_section_header(text, page_idx)
            page_chunks = chunk_text(
                text=text,
                document_id=doc_id,
                document_name=doc_name,
                page=page_idx,
                config=config,
            )

            for chunk in page_chunks:
                chunk.metadata["section"] = section
                chunks.append(chunk)

    except Exception as e:
        logger.error(f"Error parsing PDF '{filepath.name}': {e}")

    return chunks


def get_chroma_collection(reset: bool = False) -> chromadb.Collection:
    """Initialize persistent ChromaDB client and collection with fast embedding function."""
    client = chromadb.PersistentClient(path=settings.chroma_path)

    if reset:
        try:
            client.delete_collection(name=settings.chroma_collection)
            logger.info(f"Deleted existing collection '{settings.chroma_collection}'")
        except Exception:
            pass

    # Use default fast local embedding function (ONNXMiniLM_L6_V2) for sub-200ms latency & $0 cost
    if settings.openai_api_key and not settings.use_local_embedding:
        logger.info("Using OpenAI Embedding Function (text-embedding-3-small)...")
        emb_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
        )
    else:
        logger.info(
            "Using Local Default ONNX Embedding Function ($0 Cost, ~10ms Latency)..."
        )
        emb_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=emb_fn,
        metadata={"description": "Automotive manual vector index for KMS Core RAG"},
    )
    return collection


def run_ingestion(
    pdf_dir: str | Path = settings.docs_pdf_dir,
    corpus_dir: str | Path = settings.docs_corpus_dir,
    reset: bool = False,
    batch_size: int = 200,
) -> int:
    """Run full PDF ingestion pipeline across docs_corpus and docs_pdf."""
    start_time = time.time()
    logger.info("Starting Automotive PDF Manual Ingestion Pipeline...")

    hash_to_name = load_document_mapping(corpus_dir)
    chunk_config = ChunkingConfig(
        window=settings.chunk_window,
        overlap=settings.chunk_overlap,
    )

    pdf_files: list[tuple[Path, str, str]] = []

    # 1. Gather files from docs_corpus
    corpus_path = Path(corpus_dir)
    if corpus_path.exists():
        for pdf_file in corpus_path.glob("*.pdf"):
            doc_id = pdf_file.stem
            doc_name = hash_to_name.get(doc_id, doc_id)
            if not doc_name.endswith(".pdf"):
                doc_name = f"{doc_name}.pdf"
            pdf_files.append((pdf_file, doc_id, doc_name))

    # 2. Gather additional files from docs_pdf
    pdf_path = Path(pdf_dir)
    processed_hashes = {doc_id for _, doc_id, _ in pdf_files}
    if pdf_path.exists():
        for pdf_file in pdf_path.glob("*.pdf"):
            doc_name = pdf_file.name
            doc_id = calculate_file_hash(pdf_file)
            if doc_id not in processed_hashes:
                pdf_files.append((pdf_file, doc_id, doc_name))
                processed_hashes.add(doc_id)

    logger.info(f"Found {len(pdf_files)} target PDF documents to process.")

    collection = get_chroma_collection(reset=reset)

    total_chunks = 0
    batch_ids: list[str] = []
    batch_documents: list[str] = []
    batch_metadatas: list[dict[str, Any]] = []

    for file_idx, (pdf_file, doc_id, doc_name) in enumerate(pdf_files, start=1):
        chunks = process_pdf_file(pdf_file, doc_id, doc_name, chunk_config)
        logger.info(
            f"[{file_idx}/{len(pdf_files)}] Processed '{pdf_file.name}' -> {len(chunks)} chunks"
        )

        for chunk in chunks:
            chunk_unique_id = f"{doc_id}_p{chunk.page}_c{chunk.chunk_index}"
            batch_ids.append(chunk_unique_id)
            batch_documents.append(chunk.text)
            batch_metadatas.append(chunk.metadata)
            total_chunks += 1

            if len(batch_ids) >= batch_size:
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )
                batch_ids.clear()
                batch_documents.clear()
                batch_metadatas.clear()

    # Upsert remaining batch
    if batch_ids:
        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )

    elapsed = time.time() - start_time
    logger.info("=== Ingestion Complete ===")
    logger.info(f"Processed Documents: {len(pdf_files)}")
    logger.info(f"Total Chunks Stored: {total_chunks}")
    logger.info(f"Time Taken: {elapsed:.2f} seconds")

    return total_chunks


def run_ingestion_opensearch(
    pdf_dir: str | Path = settings.docs_pdf_dir,
    corpus_dir: str | Path = settings.docs_corpus_dir,
    batch_size: int = 50,
) -> int:
    """Run full PDF ingestion pipeline inserting vectors into Amazon OpenSearch Serverless."""
    from core.aws_client import generate_bedrock_embeddings, get_opensearch_client
    from utils.opensearch_utils import bulk_index_chunks, ensure_opensearch_index

    start_time = time.time()
    logger.info("Starting OpenSearch Serverless PDF Ingestion Pipeline...")

    hash_to_name = load_document_mapping(corpus_dir)
    chunk_config = ChunkingConfig(
        window=settings.chunk_window,
        overlap=settings.chunk_overlap,
    )

    pdf_files: list[tuple[Path, str, str]] = []

    corpus_path = Path(corpus_dir)
    if corpus_path.exists():
        for pdf_file in corpus_path.glob("*.pdf"):
            doc_id = pdf_file.stem
            doc_name = hash_to_name.get(doc_id, doc_id)
            if not doc_name.endswith(".pdf"):
                doc_name = f"{doc_name}.pdf"
            pdf_files.append((pdf_file, doc_id, doc_name))

    pdf_path = Path(pdf_dir)
    processed_hashes = {doc_id for _, doc_id, _ in pdf_files}
    if pdf_path.exists():
        for pdf_file in pdf_path.glob("*.pdf"):
            doc_name = pdf_file.name
            doc_id = calculate_file_hash(pdf_file)
            if doc_id not in processed_hashes:
                pdf_files.append((pdf_file, doc_id, doc_name))
                processed_hashes.add(doc_id)

    logger.info(f"Found {len(pdf_files)} target PDF documents for OpenSearch ingestion.")

    client = get_opensearch_client()
    index_name = settings.opensearch_index
    ensure_opensearch_index(client, index_name, dimension=1024)

    total_chunks = 0
    batch_chunks: list[TextChunk] = []

    for file_idx, (pdf_file, doc_id, doc_name) in enumerate(pdf_files, start=1):
        chunks = process_pdf_file(pdf_file, doc_id, doc_name, chunk_config)
        logger.info(f"[{file_idx}/{len(pdf_files)}] Processed '{pdf_file.name}' -> {len(chunks)} chunks")

        for chunk in chunks:
            batch_chunks.append(chunk)
            total_chunks += 1

            if len(batch_chunks) >= batch_size:
                texts = [c.text for c in batch_chunks]
                embeddings = generate_bedrock_embeddings(texts)
                bulk_index_chunks(client, index_name, batch_chunks, embeddings)
                batch_chunks.clear()

    if batch_chunks:
        texts = [c.text for c in batch_chunks]
        embeddings = generate_bedrock_embeddings(texts)
        bulk_index_chunks(client, index_name, batch_chunks, embeddings)
        batch_chunks.clear()

    elapsed = time.time() - start_time
    logger.info("=== OpenSearch Ingestion Complete ===")
    logger.info(f"Processed Documents: {len(pdf_files)}")
    logger.info(f"Total Chunks Stored: {total_chunks}")
    logger.info(f"Time Taken: {elapsed:.2f} seconds")

    return total_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMS Core AI PDF Ingestion Engine")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collection before ingestion")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for ChromaDB upsert")
    parser.add_argument("--target", type=str, default="chroma", choices=["chroma", "opensearch"], help="Vector database target engine")
    args = parser.parse_args()

    if args.target == "opensearch" or settings.vector_db_type == "opensearch":
        run_ingestion_opensearch(batch_size=args.batch_size)
    else:
        run_ingestion(reset=args.reset, batch_size=args.batch_size)

