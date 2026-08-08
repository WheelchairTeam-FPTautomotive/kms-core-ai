import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from pipelines.chunker import ChunkingConfig, TextChunk, chunk_text
from utils.logger import setup_logger

logger = setup_logger("kms_pdf_ingest")


def _default_ingest_workers() -> int:
    """Bounded CPU process count — OCR is RAM-heavy; avoid oversubscription."""
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu))


def _process_pdf_worker(
    filepath: str,
    doc_id: str,
    doc_name: str,
    window: int,
    overlap: int,
) -> dict[str, Any]:
    """
    Top-level worker for ProcessPoolExecutor (must be picklable on Windows spawn).

    Returns serializable chunk payloads; Chroma upsert stays in the parent process.
    """
    # --- START MODIFICATION ---
    config = ChunkingConfig(window=window, overlap=overlap)
    chunks = process_pdf_file(Path(filepath), doc_id, doc_name, config)
    payload = [
        {
            "text": c.text,
            "chunk_index": c.chunk_index,
            "page": c.page,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    return {
        "filepath": filepath,
        "doc_id": doc_id,
        "doc_name": doc_name,
        "chunks": payload,
        "n_chunks": len(payload),
    }
    # --- END MODIFICATION ---


# Load english words list
VOCAB = None

def get_vocab() -> set[str]:
    global VOCAB
    if VOCAB is not None:
        return VOCAB
        
    vocab_path = Path(__file__).parent / "google_10000_english.txt"
    words = set()
    if vocab_path.exists():
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                words = set(f.read().lower().splitlines())
        except Exception:
            pass
            
    # Add common technical terms/plurals that might not be in the top 10000 general words
    technical_words = {
        "programme", "nicaragua", "evidence", "professional", "criteria", "pending", "approval",
        "requirements", "specification", "specifications", "automotive", "management", "system",
        "validation", "integration", "implementation", "operational", "uninterpretable", "environmental",
        "attainments", "attestation", "attestations", "transferred", "contracting", "authority",
        "directive", "fulfilment", "evidences"
    }
    words.update(technical_words)
    VOCAB = words
    return VOCAB


def join_split_words(text: str) -> str:
    vocab = get_vocab()
    valid_2 = {'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we'}
    valid_3 = {
        'act', 'add', 'age', 'ago', 'aim', 'air', 'all', 'and', 'any', 'art', 'ask', 'bad', 'bag', 'bar', 'bat', 'bed', 'bee', 'bet', 'big', 'bit', 'box', 'boy', 'bus', 'but', 'buy', 'bye', 'can', 'cap', 'car', 'cat', 'cop', 'cry', 'cup', 'cut', 'day', 'did', 'die', 'dig', 'dog', 'dry', 'due', 'ear', 'eat', 'egg', 'end', 'era', 'eye', 'fan', 'far', 'fat', 'fee', 'few', 'fit', 'fix', 'fly', 'for', 'fox', 'fun', 'fur', 'gas', 'gem', 'get', 'god', 'gun', 'guy', 'had', 'has', 'hat', 'her', 'him', 'his', 'hit', 'hop', 'hot', 'how', 'hub', 'hug', 'ice', 'ill', 'ink', 'its', 'jam', 'jar', 'job', 'key', 'kid', 'lab', 'lap', 'law', 'lay', 'led', 'let', 'lie', 'lip', 'log', 'lot', 'low', 'mad', 'man', 'map', 'mat', 'may', 'men', 'met', 'mix', 'mud', 'mug', 'net', 'new', 'nod', 'not', 'now', 'nut', 'odd', 'off', 'oil', 'old', 'one', 'opt', 'our', 'out', 'own', 'pad', 'pan', 'pat', 'pay', 'pen', 'pet', 'pin', 'pit', 'pot', 'pub', 'rag', 'ran', 'raw', 'ray', 'red', 'rib', 'rid', 'rig', 'rim', 'rip', 'rob', 'rod', 'row', 'rub', 'rug', 'run', 'rye', 'sad', 'saw', 'say', 'sea', 'see', 'set', 'sew', 'she', 'sin', 'sip', 'sir', 'sit', 'six', 'ski', 'sky', 'sly', 'sob', 'son', 'soy', 'spa', 'sum', 'sun', 'tag', 'tan', 'tap', 'tax', 'tea', 'ten', 'the', 'tie', 'tin', 'tip', 'toe', 'ton', 'too', 'top', 'toy', 'try', 'tub', 'two', 'urn', 'use', 'van', 'vet', 'via', 'vow', 'wad', 'wag', 'war', 'was', 'way', 'web', 'wed', 'wet', 'who', 'why', 'wig', 'win', 'wit', 'won', 'woo', 'yes', 'yet', 'you', 'zoo'
    }
    
    tokens = re.split(r'([a-zA-Z]+)', text)
    i = 1
    while i < len(tokens) - 2:
        w1 = tokens[i]
        sep = tokens[i+1]
        w2 = tokens[i+2]
        
        if sep == ' ':
            w1_lower = w1.lower()
            w2_lower = w2.lower()
            combined = w1_lower + w2_lower
            
            if combined in vocab:
                w1_is_word = w1_lower in vocab
                w2_is_word = w2_lower in vocab
                
                if len(w1_lower) == 1 and w1_lower not in {'a', 'i'}:
                    w1_is_word = False
                elif len(w1_lower) == 2 and w1_lower not in valid_2:
                    w1_is_word = False
                elif len(w1_lower) == 3 and w1_lower not in valid_3:
                    w1_is_word = False
                    
                if len(w2_lower) == 1 and w2_lower not in {'a', 'i'}:
                    w2_is_word = False
                elif len(w2_lower) == 2 and w2_lower not in valid_2:
                    w2_is_word = False
                elif len(w2_lower) == 3 and w2_lower not in valid_3:
                    w2_is_word = False
                    
                if not w1_is_word or not w2_is_word:
                    if w1[0].isupper():
                        joined_word = w1[0] + combined[1:]
                    else:
                        joined_word = combined
                    tokens[i] = joined_word
                    del tokens[i+1:i+3]
                    continue
        i += 2
        
    return "".join(tokens)


def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. Join words split by spaces
    text = join_split_words(text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Split line into "word groups" by 2 or more spaces
        word_groups = re.split(r' {2,}', line)
        cleaned_groups = []
        for group in word_groups:
            # Within each group, split by single spaces
            tokens = group.split(' ')
            new_tokens = []
            temp_word = []
            for t in tokens:
                # If t is a single alphanumeric char
                if len(t) == 1 and t.isalnum():
                    temp_word.append(t)
                else:
                    if temp_word:
                        new_tokens.append("".join(temp_word))
                        temp_word = []
                    if t:
                        new_tokens.append(t)
            if temp_word:
                new_tokens.append("".join(temp_word))
            cleaned_groups.append(" ".join(new_tokens))
        cleaned_lines.append("  ".join(cleaned_groups))
        
    cleaned_text = "\n".join(cleaned_lines)
    # Normalize multiple spaces
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    # Normalize multiple newlines
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    return cleaned_text.strip()


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
    """Parse a single PDF file page-by-page and chunk text (PyMuPDF + OCR fallback)."""
    # --- START MODIFICATION ---
    from pipelines.pdf_extract import extract_page_text, pdf_page_count

    chunks: list[TextChunk] = []
    ocr_on_thin = bool(getattr(settings, "pdf_ocr_on_thin_page", True))
    ocr_dpi = int(getattr(settings, "pdf_ocr_dpi", 200))
    try:
        n_pages = pdf_page_count(str(filepath))
        ocr_pages = 0
        for page_idx in range(1, n_pages + 1):
            text, method = extract_page_text(
                str(filepath),
                page_idx,
                ocr_on_thin=ocr_on_thin,
                ocr_dpi=ocr_dpi,
            )
            if method == "ocr":
                ocr_pages += 1
            text = clean_pdf_text(text)
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

            # --- START MODIFICATION ---
            # Path-derived vehicle fields — always str (never None) for Chroma
            from utils.vehicle_meta import parse_vehicle_metadata

            vehicle = parse_vehicle_metadata(filepath, doc_name=doc_name)
            for chunk in page_chunks:
                chunk.metadata["section"] = section
                chunk.metadata["extract_method"] = method
                chunk.metadata["make"] = vehicle["make"]
                chunk.metadata["model"] = vehicle["model"]
                chunk.metadata["year"] = vehicle["year"]
                chunks.append(chunk)
            # --- END MODIFICATION ---

        if ocr_pages:
            logger.info(
                f"OCR used on {ocr_pages}/{n_pages} pages for '{filepath.name}'"
            )

    except Exception as e:
        logger.error(f"Error parsing PDF '{filepath.name}': {e}")

    return chunks
    # --- END MODIFICATION ---


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
        metadata={
            "description": "Automotive manual vector index for KMS Core RAG",
            "hnsw:space": "l2",
        },
    )
    return collection


def _path_matches_globs(path: Path, globs: list[str] | None) -> bool:
    if not globs:
        return True
    # Match against both POSIX-style relative-ish and name
    as_posix = path.as_posix()
    name = path.name
    for pattern in globs:
        if path.match(pattern) or Path(as_posix).match(pattern):
            return True
        # fnmatch on full posix path for **/Accent/2020/**
        import fnmatch

        if fnmatch.fnmatch(as_posix, pattern) or fnmatch.fnmatch(name, pattern):
            return True
    return False


def evict_documents_from_chroma(
    collection: chromadb.Collection,
    doc_names: set[str],
    doc_ids: set[str],
) -> int:
    """Delete all Chroma vectors for the given document_name / document_id set."""
    # --- START MODIFICATION ---
    if not doc_names and not doc_ids:
        return 0
    deleted = 0
    # Chroma where $in for document_name
    try:
        if doc_names:
            names = sorted(doc_names)
            # batch $in to avoid oversized filters
            for i in range(0, len(names), 50):
                batch = names[i : i + 50]
                existing = collection.get(
                    where={"document_name": {"$in": batch}},
                    include=[],
                )
                ids = existing.get("ids") or []
                if ids:
                    collection.delete(ids=ids)
                    deleted += len(ids)
    except Exception as exc:
        logger.warning(f"Eviction by document_name failed: {exc}")

    # Also evict by document_id metadata when present
    try:
        if doc_ids:
            ids_list = sorted(doc_ids)
            for i in range(0, len(ids_list), 50):
                batch = ids_list[i : i + 50]
                existing = collection.get(
                    where={"document_id": {"$in": batch}},
                    include=[],
                )
                ids = existing.get("ids") or []
                if ids:
                    collection.delete(ids=ids)
                    deleted += len(ids)
    except Exception as exc:
        logger.warning(f"Eviction by document_id failed: {exc}")

    logger.info(f"Evicted {deleted} existing Chroma ids before re-ingest")
    return deleted
    # --- END MODIFICATION ---


def _flush_upsert_batch(
    collection: chromadb.Collection,
    batch_ids: list[str],
    batch_documents: list[str],
    batch_metadatas: list[dict[str, Any]],
) -> None:
    if not batch_ids:
        return
    collection.upsert(
        ids=batch_ids,
        documents=batch_documents,
        metadatas=batch_metadatas,
    )
    batch_ids.clear()
    batch_documents.clear()
    batch_metadatas.clear()


def _append_chunk_payloads(
    *,
    doc_id: str,
    chunk_payloads: list[dict[str, Any]],
    batch_ids: list[str],
    batch_documents: list[str],
    batch_metadatas: list[dict[str, Any]],
    batch_size: int,
    collection: chromadb.Collection,
) -> int:
    """Queue serializable worker chunks into Chroma upsert batches (parent only)."""
    added = 0
    for chunk in chunk_payloads:
        chunk_unique_id = f"{doc_id}_p{chunk['page']}_c{chunk['chunk_index']}"
        batch_ids.append(chunk_unique_id)
        batch_documents.append(chunk["text"])
        batch_metadatas.append(chunk["metadata"])
        added += 1
        if len(batch_ids) >= batch_size:
            _flush_upsert_batch(
                collection, batch_ids, batch_documents, batch_metadatas
            )
    return added


def run_ingestion(
    pdf_dir: str | Path = settings.docs_pdf_dir,
    corpus_dir: str | Path = settings.docs_corpus_dir,
    reset: bool = False,
    batch_size: int = 200,
    only_globs: list[str] | None = None,
    workers: int | None = None,
) -> int:
    """Run full PDF ingestion pipeline across docs_corpus and docs_pdf."""
    # --- START MODIFICATION ---
    # Parallel PDF extract/OCR via process pool; single-threaded Chroma upsert.
    start_time = time.time()
    logger.info("Starting Automotive PDF Manual Ingestion Pipeline...")

    worker_count = workers if workers is not None else _default_ingest_workers()
    worker_count = max(1, int(worker_count))

    # Pre-warm RapidOCR when thin-page OCR is enabled (avoid mid-run download stall)
    if bool(getattr(settings, "pdf_ocr_on_thin_page", True)):
        from pipelines.pdf_extract import warmup_ocr

        warmup_ocr()

    hash_to_name = load_document_mapping(corpus_dir)
    chunk_config = ChunkingConfig(
        window=settings.chunk_window,
        overlap=settings.chunk_overlap,
    )

    pdf_files: list[tuple[Path, str, str]] = []

    # 1. Gather files from docs_corpus (recursive — HACKATHON uses category subfolders)
    corpus_path = Path(corpus_dir)
    if corpus_path.exists():
        for pdf_file in corpus_path.rglob("*.pdf"):
            if not _path_matches_globs(pdf_file, only_globs):
                continue
            doc_id = pdf_file.stem
            mapped = hash_to_name.get(doc_id)
            # Prefer mapping title; else basename only (never nested path)
            doc_name = mapped if mapped else pdf_file.name
            if not doc_name.endswith(".pdf"):
                doc_name = f"{doc_name}.pdf"
            pdf_files.append((pdf_file, doc_id, doc_name))

    # 2. Gather additional files from docs_pdf (recursive)
    pdf_path = Path(pdf_dir)
    processed_hashes = {doc_id for _, doc_id, _ in pdf_files}
    if pdf_path.exists():
        for pdf_file in pdf_path.rglob("*.pdf"):
            if not _path_matches_globs(pdf_file, only_globs):
                continue
            doc_name = pdf_file.name  # basename only
            doc_id = calculate_file_hash(pdf_file)
            if doc_id not in processed_hashes:
                # Prefer mapping by hash when available
                mapped = hash_to_name.get(doc_id)
                if mapped:
                    doc_name = mapped if mapped.endswith(".pdf") else f"{mapped}.pdf"
                pdf_files.append((pdf_file, doc_id, doc_name))
                processed_hashes.add(doc_id)

    logger.info(f"Found {len(pdf_files)} target PDF documents to process.")
    logger.info(f"Ingest workers: {worker_count}")
    if only_globs:
        logger.info(f"only_globs={only_globs}")

    collection = get_chroma_collection(reset=reset)

    # Targeted re-ingest: purge all prior vectors for these docs (OCR changes chunk counts)
    if only_globs and pdf_files and not reset:
        evict_documents_from_chroma(
            collection,
            doc_names={name for _, _, name in pdf_files},
            doc_ids={doc_id for _, doc_id, _ in pdf_files},
        )

    total_chunks = 0
    batch_ids: list[str] = []
    batch_documents: list[str] = []
    batch_metadatas: list[dict[str, Any]] = []
    n_files = len(pdf_files)

    if worker_count == 1 or n_files <= 1:
        for file_idx, (pdf_file, doc_id, doc_name) in enumerate(pdf_files, start=1):
            chunks = process_pdf_file(pdf_file, doc_id, doc_name, chunk_config)
            logger.info(
                f"[{file_idx}/{n_files}] Processed '{pdf_file.name}' -> {len(chunks)} chunks"
            )
            payloads = [
                {
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "page": c.page,
                    "metadata": c.metadata,
                }
                for c in chunks
            ]
            total_chunks += _append_chunk_payloads(
                doc_id=doc_id,
                chunk_payloads=payloads,
                batch_ids=batch_ids,
                batch_documents=batch_documents,
                batch_metadatas=batch_metadatas,
                batch_size=batch_size,
                collection=collection,
            )
    else:
        # ProcessPool: extract/OCR in workers; upsert serially in parent (Chroma-safe).
        done = 0
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    _process_pdf_worker,
                    str(pdf_file),
                    doc_id,
                    doc_name,
                    chunk_config.window,
                    chunk_config.overlap,
                ): pdf_file.name
                for pdf_file, doc_id, doc_name in pdf_files
            }
            for fut in as_completed(futures):
                name = futures[fut]
                done += 1
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.error(f"[{done}/{n_files}] Worker failed '{name}': {exc}")
                    continue
                logger.info(
                    f"[{done}/{n_files}] Processed '{Path(result['filepath']).name}' "
                    f"-> {result['n_chunks']} chunks"
                )
                total_chunks += _append_chunk_payloads(
                    doc_id=result["doc_id"],
                    chunk_payloads=result["chunks"],
                    batch_ids=batch_ids,
                    batch_documents=batch_documents,
                    batch_metadatas=batch_metadatas,
                    batch_size=batch_size,
                    collection=collection,
                )

    _flush_upsert_batch(collection, batch_ids, batch_documents, batch_metadatas)

    elapsed = time.time() - start_time
    logger.info("=== Ingestion Complete ===")
    logger.info(f"Processed Documents: {len(pdf_files)}")
    logger.info(f"Total Chunks Stored: {total_chunks}")
    logger.info(f"Workers Used: {worker_count}")
    logger.info(f"Time Taken: {elapsed:.2f} seconds")

    # --- START MODIFICATION ---
    try:
        from utils.bm25_index import rebuild_from_chroma

        rebuild_from_chroma(collection)
    except Exception as exc:
        logger.warning("BM25 sidecar rebuild after ingest failed: %s", exc)
    # --- END MODIFICATION ---

    return total_chunks
    # --- END MODIFICATION ---


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
        for pdf_file in corpus_path.rglob("*.pdf"):
            doc_id = pdf_file.stem
            mapped = hash_to_name.get(doc_id)
            doc_name = mapped if mapped else pdf_file.name
            if not doc_name.endswith(".pdf"):
                doc_name = f"{doc_name}.pdf"
            pdf_files.append((pdf_file, doc_id, doc_name))

    pdf_path = Path(pdf_dir)
    processed_hashes = {doc_id for _, doc_id, _ in pdf_files}
    if pdf_path.exists():
        for pdf_file in pdf_path.rglob("*.pdf"):
            doc_name = pdf_file.name
            doc_id = calculate_file_hash(pdf_file)
            if doc_id not in processed_hashes:
                mapped = hash_to_name.get(doc_id)
                if mapped:
                    doc_name = mapped if mapped.endswith(".pdf") else f"{mapped}.pdf"
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
    parser.add_argument(
        "--only-glob",
        action="append",
        default=None,
        help="Limit ingest to paths matching this glob (repeatable). Example: **/Accent/2020/**",
    )
    # --- START MODIFICATION ---
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel PDF extract/OCR processes (Chroma upsert stays single-threaded). "
            f"Default: min(4, CPU count) = {_default_ingest_workers()}. Use 1 for serial."
        ),
    )
    # --- END MODIFICATION ---
    args = parser.parse_args()

    if args.target == "opensearch" or settings.vector_db_type == "opensearch":
        run_ingestion_opensearch(batch_size=args.batch_size)
    else:
        run_ingestion(
            reset=args.reset,
            batch_size=args.batch_size,
            only_globs=args.only_glob,
            workers=args.workers,
        )

