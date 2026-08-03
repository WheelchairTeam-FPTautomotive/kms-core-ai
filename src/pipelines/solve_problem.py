import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from utils.logger import setup_logger
from utils.query_expand import expand_retrieval_queries

logger = setup_logger("core_rag_pipeline")

UNSAFE_TRIGGERS = [
    "hack",
    "bypass brake",
    "bypass brakes",
    "overdrive engine",
    "overdrive engine safety",
    "ignore seatbelt",
    "ignore seatbelt alert",
    "disable aeb",
    "disable abs",
    "jailbreak",
    "ignore previous instruction",
    "ignore previous instructions",
]


def check_safety_and_scope(query: str, language: str | None = "vi") -> tuple[bool, str]:
    # --- START MODIFICATION ---
    # Unsafe blocklist only; automotive allowlist removed (RAG gate owns relevance)
    from core.locale_messages import refused_answer
    from utils.query_expand import fold_vi

    folded = fold_vi(query)
    for trigger in UNSAFE_TRIGGERS:
        if fold_vi(trigger) in folded:
            logger.warning(
                f"Unsafe request detected: '{query}' triggering: '{trigger}'"
            )
            return False, refused_answer(language)
    return True, ""
    # --- END MODIFICATION ---


def solve_automotive_query(query: str) -> Dict[str, Any]:
    logger.info(f"RAG processing query: '{query}'")

    is_valid, refusal_reason = check_safety_and_scope(query)
    if not is_valid:
        return {
            "query": query,
            "answer": refusal_reason,
            "citations": [],
            "status": "refused",
        }

    logger.info("Executing vector database query...")
    citations = [
        {
            "document_id": "949eb66893b5dbf59aa4b4be35ad330c7b8f0c3802f9ccb8d25881128157bf9c",
            "document_name": "2011 - KMS Manual.pdf",
            "section": "Chương 4: Điều hòa & Hệ thống điện",
            "page": 42,
            "matched_text": "Hệ thống điều hòa (HVAC) được điều khiển qua CarPropertyManager với AreaId là 0.",
        },
        {
            "document_id": "1ecc7f4e2b438cb0ac5c336fed7cfffbca78b42f87a31a0c0add50aa38cfc751",
            "document_name": "light-control-system.pdf",
            "section": "Chương 7: ADAS & Phanh khẩn cấp",
            "page": 105,
            "matched_text": "Khi xe chạy quá tốc độ 80km/h, hệ thống ADAS kích hoạt phanh khẩn cấp tự động (AEB) nếu khoảng cách xe trước < 15m.",
        },
    ]

    answer = (
        f"Dựa trên tài liệu hướng dẫn kỹ thuật của xe:\n"
        f"1. Hệ thống điều hòa (HVAC) hoạt động trên VHAL thông qua CarPropertyManager (AreaId: 0).\n"
        f"2. Phanh khẩn cấp tự động (AEB) hoạt động kết hợp với ADAS sẽ kích hoạt để bảo vệ an toàn khi xe chạy > 80km/h và khoảng cách va chạm dưới 15m."
    )

    logger.info("Formulated RAG response with citations.")
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "status": "success",
    }

# Functions for RAG Vector search pipeline
_COLLECTION_CACHE = None

def get_chroma_collection():
    global _COLLECTION_CACHE
    if _COLLECTION_CACHE is None:
        try:
            client = chromadb.PersistentClient(path=settings.chroma_path)
            if settings.openai_api_key and not settings.use_local_embedding:
                emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.openai_api_key,
                    model_name=settings.embedding_model,
                )
            else:
                emb_fn = embedding_functions.DefaultEmbeddingFunction()

            try:
                _COLLECTION_CACHE = client.get_or_create_collection(
                    name=settings.chroma_collection,
                    embedding_function=emb_fn,
                    # MODIFIED: make L2 space explicit for distance-gate calibration
                    metadata={
                        "description": "Automotive manual vector index for KMS Core RAG",
                        "hnsw:space": "l2",
                    },
                )
            except Exception as meta_err:
                logger.warning(
                    f"Chroma create with hnsw:space=l2 failed ({meta_err}); "
                    "opening existing collection as-is"
                )
                _COLLECTION_CACHE = client.get_or_create_collection(
                    name=settings.chroma_collection,
                    embedding_function=emb_fn,
                )
            space = (_COLLECTION_CACHE.metadata or {}).get("hnsw:space", "l2-default")
            logger.info(f"Chroma collection ready (hnsw:space={space})")
            _COLLECTION_CACHE.query(query_texts=["warmup"], n_results=1)
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise
    return _COLLECTION_CACHE


@functools.lru_cache(maxsize=1024)
def _cached_vector_query(query: str, top_k: int = 3) -> tuple:
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted_results = []
    for doc, meta, distance in zip(docs, metas, distances):
        formatted_results.append(
            (
                meta.get("document_id", ""),
                meta.get("document_name", "Automotive Manual"),
                meta.get("section", f"Trang {meta.get('page', 1)}"),
                int(meta.get("page", 1)),
                doc[:300],
                float(distance),
            )
        )
    return tuple(formatted_results)


NOT_FOUND_ANSWER = (
    "Không tìm thấy thông tin phù hợp trong tài liệu kỹ thuật của xe. "
    "Bạn có thể hỏi cách khác hoặc chủ đề có trong manual."
)


def _merge_multi_query_hits(queries: list[str], top_k: int) -> list[tuple]:
    """
    Run cached Chroma lookups for each retrieval variant; keep best distance per chunk.
    --- START MODIFICATION ---
    """
    best_by_key: dict[tuple, tuple] = {}
    for q in queries:
        for hit in _cached_vector_query(q, top_k=top_k):
            # MODIFIED: unused fields prefixed for Ruff RUF059
            doc_id, _doc_name, _section, page, text_snippet, distance = hit
            key = (doc_id, page, text_snippet[:80])
            prev = best_by_key.get(key)
            if prev is None or distance < prev[5]:
                best_by_key[key] = hit
    merged = sorted(best_by_key.values(), key=lambda h: h[5])
    return merged[:top_k]
    # --- END MODIFICATION ---


def solve_automotive_query_live(
    query: str, language: str | None = "vi"
) -> Dict[str, Any]:
    """Live ChromaDB vector retrieval solver with distance relevance gate."""
    from core.locale_messages import not_found_answer

    start_time = time.time()
    logger.info(f"Live RAG processing query: '{query}' language={language}")

    is_valid, refusal_reason = check_safety_and_scope(query, language=language)
    if not is_valid:
        return {
            "query": query,
            "answer": refusal_reason,
            "citations": [],
            "status": "refused",
        }

    try:
        top_k = getattr(settings, "rag_top_k", 3) or 3
        max_distance = float(getattr(settings, "rag_max_distance", 1.15))
        # --- START MODIFICATION ---
        # Tone-free VI / cross-lingual: expand for retrieval only; answer uses original query
        retrieval_queries = expand_retrieval_queries(query)
        logger.info(f"RAG retrieval_queries={retrieval_queries}")
        raw_citations = _merge_multi_query_hits(retrieval_queries, top_k=top_k)
        # --- END MODIFICATION ---
        citations = []
        snippets = []
        best_distance = None

        for idx, (doc_id, doc_name, section, page, text_snippet, distance) in enumerate(
            raw_citations, start=1
        ):
            if best_distance is None or distance < best_distance:
                best_distance = distance
            # --- START MODIFICATION ---
            # Relevance gate: drop weak neighbors above RAG_MAX_DISTANCE
            if distance > max_distance:
                continue
            # --- END MODIFICATION ---
            citations.append(
                {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "section": section,
                    "page": page,
                    "matched_text": text_snippet,
                }
            )
            snippets.append(
                f"{idx}. [{doc_name} - {section} (Trang {page})]: {text_snippet}"
            )

        space = (get_chroma_collection().metadata or {}).get("hnsw:space", "l2")
        logger.info(
            f"RAG gate space={space} best_distance={best_distance} "
            f"max={max_distance} kept={len(snippets)}/{len(raw_citations)}"
        )

        if not snippets:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Live RAG not_found in {elapsed_ms:.2f}ms (weak/no hits)")
            return {
                "query": query,
                "answer": not_found_answer(language),
                "citations": [],
                "status": "not_found",
            }

        from core.answer_generator import generate_driver_answer

        answer = generate_driver_answer(query, snippets, language=language)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Live RAG response formulated in {elapsed_ms:.2f}ms with {len(citations)} citations."
        )

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")
        return {
            "query": query,
            "answer": f"Đã xảy ra lỗi trong quá trình tra cứu dữ liệu: {e!s}",
            "citations": [],
            "status": "error",
        }


def solve_free_talk_query(query: str, language: str | None = "vi") -> Dict[str, Any]:
    """Free-talk path: no RAG retrieval; LLM or polite redirect."""
    from core.answer_generator import generate_free_talk_answer

    answer = generate_free_talk_answer(query, language=language)
    return {
        "query": query,
        "answer": answer,
        "citations": [],
        "status": "success",
    }


def solve_automotive_query_auto(
    query: str, mode: str = "rag", language: str | None = "vi"
) -> Dict[str, Any]:
    """
    Dispatch by mode and vector_db_type.
    Answer language follows UI `language` only.
    """
    resolved = (mode or "rag").strip().lower()
    if resolved == "free_talk":
        return solve_free_talk_query(query, language=language)

    if getattr(settings, "vector_db_type", "chroma") == "opensearch":
        from pipelines.bedrock_rag import solve_automotive_query_bedrock
        return solve_automotive_query_bedrock(query)

    return solve_automotive_query_live(query, language=language)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="KMS RAG Offline Evaluator CLI")
#     parser.add_argument(
#         "--input", required=True, help="Input directory containing queries"
#     )
#     parser.add_argument(
#         "--output", required=True, help="Output file to write responses to"
#     )
#     args = parser.parse_args()

#     logger.info(
#         f"Running offline batch evaluation: input={args.input}, output={args.output}"
#     )

#     # Mock reading inputs
#     queries = ["Làm thế nào kích hoạt phanh khẩn cấp ADAS?"]

#     results = []
#     for q in queries:
#         res = solve_automotive_query(q)
#         results.append(res)

#     os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
#     with open(args.output, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     logger.info(f"Batch evaluation finished. Results written to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMS RAG Offline Evaluator CLI")
    parser.add_argument(
        "--input", required=True, help="Input directory containing JSON query files"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSON file to write responses to"
    )
    args = parser.parse_args()

    logger.info(
        f"Running offline batch evaluation: input={args.input}, output={args.output}"
    )

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {args.input}")
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    total_queries = 0
    skipped_files = 0

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No .json files found in input directory: {args.input}")

    for query_file in json_files:
        try:
            with query_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping malformed JSON file {query_file}: {e}")
            skipped_files += 1
            continue
        except OSError as e:
            logger.warning(f"Skipping unreadable file {query_file}: {e}")
            skipped_files += 1
            continue

        if not isinstance(data, list):
            logger.warning(
                f"Skipping {query_file}: expected JSON list, got {type(data).__name__}"
            )
            skipped_files += 1
            continue

        for item in data:
            if not isinstance(item, str):
                logger.warning(f"Skipping non-string query in {query_file}: {item!r}")
                continue

            total_queries += 1
            result = solve_automotive_query_auto(item)
            results.append(result)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Batch evaluation finished: processed {len(json_files)} file(s), "
        f"ran {total_queries} query(ies), skipped {skipped_files} file(s). "
        f"Results written to: {args.output}"
    )

