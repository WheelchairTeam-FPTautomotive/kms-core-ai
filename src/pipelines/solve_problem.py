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
from openai import OpenAI

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


@functools.lru_cache(maxsize=2048)
def _cached_vector_query(query: str, top_k: int = 3, where_json: str = "") -> tuple:
    collection = get_chroma_collection()
    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_json:
        kwargs["where"] = json.loads(where_json)
    results = collection.query(**kwargs)

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
                int(meta.get("page", 1) or 0),
                doc[:300],
                float(distance),
            )
        )
    return tuple(formatted_results)


NOT_FOUND_ANSWER = (
    "Không tìm thấy thông tin phù hợp trong tài liệu kỹ thuật của xe. "
    "Bạn có thể hỏi cách khác hoặc chủ đề có trong manual."
)


def _merge_multi_query_hits(
    queries: list[str],
    top_k: int,
    where: dict | None = None,
) -> list[tuple]:
    """
    Run cached Chroma lookups for each retrieval variant; keep best distance per chunk.
    --- START MODIFICATION ---
    """
    where_json = json.dumps(where, sort_keys=True) if where else ""
    best_by_key: dict[tuple, tuple] = {}
    for q in queries:
        try:
            hits = _cached_vector_query(q, top_k=top_k, where_json=where_json)
        except Exception as exc:
            logger.warning(f"Chroma query failed where={where}: {exc}")
            continue
        for hit in hits:
            doc_id, _doc_name, _section, page, text_snippet, distance = hit
            key = (doc_id, page, text_snippet[:80])
            prev = best_by_key.get(key)
            if prev is None or distance < prev[5]:
                best_by_key[key] = hit
    merged = sorted(best_by_key.values(), key=lambda h: h[5])
    return merged[:top_k]
    # --- END MODIFICATION ---


def _gate_hits(
    raw_citations: list[tuple],
    max_distance: float,
) -> tuple[list[dict[str, Any]], list[str], float | None]:
    citations: list[dict[str, Any]] = []
    snippets: list[str] = []
    best_distance = None
    for idx, (doc_id, doc_name, section, page, text_snippet, distance) in enumerate(
        raw_citations, start=1
    ):
        if best_distance is None or distance < best_distance:
            best_distance = distance
        if distance > max_distance:
            continue
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
    return citations, snippets, best_distance


def _gate_hybrid_candidates(
    candidates: list[dict[str, Any]],
    max_distance: float,
) -> tuple[list[dict[str, Any]], list[str], float | None]:
    """
    Gate fused candidates using preserved dense_distance and/or CE/BM25 policy.
    Never treat rrf_score as a distance.
    """
    # --- START MODIFICATION ---
    ce_min = float(getattr(settings, "rag_ce_min_score", -2.0))
    bm25_max_rank = int(getattr(settings, "rag_bm25_only_max_rank", 15) or 15)
    citations: list[dict[str, Any]] = []
    snippets: list[str] = []
    best_distance = None

    for cand in candidates:
        dense_d = cand.get("dense_distance")
        ce = cand.get("ce_score")
        bm25_rank = cand.get("bm25_rank")

        if dense_d is not None:
            dense_f = float(dense_d)
            if best_distance is None or dense_f < best_distance:
                best_distance = dense_f
            if dense_f > max_distance:
                # Dense weak: allow if CE strongly prefers it
                if ce is None or float(ce) < ce_min:
                    continue
        else:
            # BM25-only: require lexical rank + CE (when available)
            if bm25_rank is not None and int(bm25_rank) > bm25_max_rank:
                continue
            if ce is not None and float(ce) < ce_min:
                continue

        citations.append(
            {
                "document_id": str(cand.get("document_id") or ""),
                "document_name": str(cand.get("document_name") or "Automotive Manual"),
                "section": str(cand.get("section") or ""),
                "page": int(cand.get("page") or 0),
                "matched_text": str(cand.get("matched_text") or "")[:300],
            }
        )

    for idx, c in enumerate(citations, start=1):
        snippets.append(
            f"{idx}. [{c['document_name']} - {c['section']} (Trang {c['page']})]: "
            f"{c['matched_text']}"
        )
    return citations, snippets, best_distance
    # --- END MODIFICATION ---


def _cascading_retrieve(
    queries: list[str],
    top_k: int,
    max_distance: float,
    make: str,
    model: str,
    year: str,
    *,
    primary_query: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], float | None, dict | None]:
    """
    Procedure retrieval cascade with hybrid BM25+dense RRF + CE rerank:
    make+model+year → make+model → make → unfiltered.
    """
    from utils.hybrid_retrieve import hybrid_retrieve_candidates
    from utils.rerank import rerank_candidates
    from utils.vehicle_meta import build_chroma_where

    attempts: list[tuple[dict | None, str, str, str]] = []
    seen: set[str] = set()

    def _add(where: dict | None, fm: str, fmo: str, fy: str) -> None:
        key = f"{json.dumps(where, sort_keys=True) if where else 'null'}|{fm}|{fmo}|{fy}"
        if key in seen:
            return
        seen.add(key)
        attempts.append((where, fm, fmo, fy))

    if make or model or year:
        _add(build_chroma_where(make, model, year), make, model, year)
    if year and (make or model):
        _add(build_chroma_where(make, model, None), make, model, "")
    if make and model:
        _add(build_chroma_where(make, model, None), make, model, "")
    if model:
        _add(build_chroma_where(None, model, None), "", model, "")
    if make:
        _add(build_chroma_where(make, None, None), make, "", "")
    _add(None, "", "", "")

    last_best = None
    pq = primary_query or (queries[0] if queries else "")
    for where, fm, fmo, fy in attempts:
        raw = _merge_multi_query_hits(queries, top_k=max(top_k, 8), where=where)
        candidates = hybrid_retrieve_candidates(
            queries,
            raw,
            make=fm,
            model=fmo,
            year=fy,
        )
        candidates = rerank_candidates(pq, candidates, top_k=max(top_k, 8))
        citations, snippets, best_distance = _gate_hybrid_candidates(
            candidates, max_distance
        )
        # Trim to top_k after gate
        citations = citations[:top_k]
        snippets = snippets[:top_k]
        last_best = best_distance if best_distance is not None else last_best
        logger.info(
            f"RAG cascade where={where} best_distance={best_distance} "
            f"kept={len(snippets)}/{len(candidates)} hybrid=1"
        )
        if snippets:
            return citations, snippets, best_distance, where
    return [], [], last_best, None


def _citations_match_vehicle(
    citations: list[dict[str, Any]],
    make: str,
    model: str,
) -> bool:
    """
    Citation honesty: when planner pinned a vehicle, at least one cite name
    must mention that model (preferred) or make. Empty vehicle → always ok.
    """
    # --- START MODIFICATION ---
    from utils.query_expand import fold_vi

    make_n = (make or "").strip().lower()
    model_n = (model or "").strip().lower()
    if not make_n and not model_n:
        return True
    if not citations:
        return False

    blob = fold_vi(
        " ".join(str(c.get("document_name") or "") for c in citations)
    )
    compact = "".join(ch for ch in blob if ch.isalnum())

    if model_n:
        # --- START MODIFICATION ---
        # Trim/title pins: bronco raptor / raptor ↔ bronco document names
        from utils.vehicle_meta import model_match_pins

        pins = model_match_pins(model_n)
        if any(p in compact for p in pins if len(p) >= 4):
            return True
        model_compact = "".join(ch for ch in fold_vi(model_n) if ch.isalnum())
        if model_compact and model_compact in compact:
            return True
        tokens = [t for t in fold_vi(model_n).split() if t]
        if tokens and all(t in blob for t in tokens):
            return True
        for c in citations:
            name_c = "".join(
                ch for ch in fold_vi(str(c.get("document_name") or "")) if ch.isalnum()
            )
            hay = name_c
            if any(p in hay or (len(p) >= 4 and hay in p) for p in pins):
                return True
            if model_compact and len(model_compact) >= 4:
                if model_compact in name_c or name_c in model_compact:
                    return True
        return False
        # --- END MODIFICATION ---

    if make_n:
        return fold_vi(make_n) in blob
    return True
    # --- END MODIFICATION ---


def _rag_miss_handoff(
    query: str, language: str | None, reason: str
) -> Dict[str, Any]:
    """Constrained FREE_TALK after RAG miss / ungrounded / cite-vehicle mismatch."""
    # --- START MODIFICATION ---
    from core.answer_generator import generate_rag_miss_handoff

    answer = generate_rag_miss_handoff(query, language=language)
    logger.info(f"RAG soft-handoff reason={reason} query={query!r}")
    return {
        "query": query,
        "answer": answer,
        "citations": [],
        "status": "success",
        "handoff": True,
    }
    # --- END MODIFICATION ---


def solve_automotive_query_live(
    query: str, language: str | None = "vi"
) -> Dict[str, Any]:
    """Live ChromaDB vector retrieval solver with planner routing + distance gate."""
    from utils.corpus_catalog import (
        catalog_citations,
        format_catalog_answer,
        list_documents_for_vehicle,
    )
    from utils.query_planner import plan_query

    start_time = time.time()
    logger.info(f"Live RAG processing query: '{query}' language={language}")

    is_valid, refusal_reason = check_safety_and_scope(query, language=language)
    if not is_valid:
        return {
            "query": query,
            "answer": refusal_reason,
            "citations": [],
            "status": "refused",
            "handoff": False,
        }

    try:
        planned = plan_query(query, language=language)

        # --- START MODIFICATION ---
        # Normalize trim aliases (bronco raptor / raptor → bronco) for Chroma where
        from dataclasses import replace

        from utils.vehicle_meta import normalize_query_vehicle

        _n_make, _n_model = normalize_query_vehicle(planned.make, planned.model)
        if (_n_make and _n_make != (planned.make or "")) or (
            _n_model and _n_model != (planned.model or "")
        ):
            planned = replace(
                planned,
                make=_n_make or planned.make,
                model=_n_model or planned.model,
            )
        # --- END MODIFICATION ---

        # --- START MODIFICATION ---
        # Catalog: metadata inventory (not vector similarity)
        if planned.intent == "catalog":
            names = list_documents_for_vehicle(
                get_chroma_collection,
                make=planned.make,
                model=planned.model,
                year=planned.year,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            if not names:
                logger.info(
                    f"Catalog empty→handoff in {elapsed_ms:.2f}ms "
                    f"source={planned.source}"
                )
                return _rag_miss_handoff(query, language, reason="catalog_empty")
            answer = format_catalog_answer(
                names,
                language=language,
                make=planned.make,
                model=planned.model,
                year=planned.year,
            )
            logger.info(
                f"Catalog answer in {elapsed_ms:.2f}ms docs={len(names)} "
                f"source={planned.source}"
            )
            return {
                "query": query,
                "answer": answer,
                "citations": catalog_citations(names),
                "status": "success",
                "handoff": False,
            }

        if planned.intent == "chitchat":
            from core.answer_generator import generate_free_talk_answer

            return {
                "query": query,
                "answer": generate_free_talk_answer(query, language=language),
                "citations": [],
                "status": "success",
                "handoff": False,
            }

        if planned.intent == "refuse":
            from core.locale_messages import refused_answer

            return {
                "query": query,
                "answer": refused_answer(language),
                "citations": [],
                "status": "refused",
                "handoff": False,
            }

        top_k = getattr(settings, "rag_top_k", 3) or 3
        max_distance = float(getattr(settings, "rag_max_distance", 1.15))
        retrieval_queries = expand_retrieval_queries(query)
        # Prefer planner English search phrase first for embedding
        if planned.search_query and planned.search_query not in retrieval_queries:
            retrieval_queries = [planned.search_query, *retrieval_queries]
        logger.info(f"RAG retrieval_queries={retrieval_queries}")

        citations, snippets, best_distance, used_where = _cascading_retrieve(
            retrieval_queries,
            top_k=top_k,
            max_distance=max_distance,
            make=planned.make,
            model=planned.model,
            year=planned.year,
            primary_query=query,
        )
        space = (get_chroma_collection().metadata or {}).get("hnsw:space", "l2")
        logger.info(
            f"RAG gate space={space} best_distance={best_distance} "
            f"max={max_distance} kept={len(snippets)} where={used_where}"
        )

        # Conditional LLM rewrite when cascade keeps 0
        if not snippets:
            from utils.query_rewrite import rewrite_enabled, rewrite_retrieval_query

            if rewrite_enabled():
                logger.info(
                    "[RAG] Cascade kept=0. Triggering LLM query rewrite..."
                )
                rewritten = rewrite_retrieval_query(query, language=language)
                if rewritten:
                    citations, snippets, best_distance, used_where = _cascading_retrieve(
                        [rewritten],
                        top_k=top_k,
                        max_distance=max_distance,
                        make=planned.make,
                        model=planned.model,
                        year=planned.year,
                        primary_query=query,
                    )
                    if snippets:
                        logger.info(
                            f"[RAG] Rewrite pass best_distance={best_distance} "
                            f"kept={len(snippets)} where={used_where}. SUCCESS."
                        )
                    else:
                        logger.info(
                            f"[RAG] Rewrite pass best_distance={best_distance} "
                            f"kept=0. SECOND_PASS_EMPTY."
                        )
                else:
                    logger.info("[RAG] REWRITE_MISS — falling through to handoff")
            else:
                logger.info(
                    "[RAG] Cascade kept=0; rewrite disabled or LLM_PROVIDER=none"
                )

        if not snippets:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Live RAG miss→handoff in {elapsed_ms:.2f}ms (weak/no hits)")
            return _rag_miss_handoff(query, language, reason="no_hits")

        # Citation honesty: pinned vehicle must appear in cite names
        if not _citations_match_vehicle(citations, planned.make, planned.model):
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Live RAG cite-vehicle mismatch→handoff in {elapsed_ms:.2f}ms "
                f"make={planned.make!r} model={planned.model!r} "
                f"cites={[c.get('document_name') for c in citations]}"
            )
            return _rag_miss_handoff(query, language, reason="cite_vehicle_mismatch")

        from core.answer_generator import extractive_summary, generate_driver_answer
        from core.grounding import is_ungrounded, parse_grounded_answer

        # Citation honesty: LLM soft-deny with retrieved evidence → extractive
        # (keep cites). Only hand off when there is nothing grounded to quote.
        raw_answer = generate_driver_answer(query, snippets, language=language)
        answer, grounded_flag = parse_grounded_answer(raw_answer)
        if is_ungrounded(answer, grounded_flag):
            elapsed_ms = (time.time() - start_time) * 1000
            if snippets:
                logger.info(
                    f"Live RAG ungrounded→extractive in {elapsed_ms:.2f}ms "
                    f"(flag={grounded_flag}, cites={len(citations)})"
                )
                answer = extractive_summary(snippets, language=language)
            else:
                logger.info(
                    f"Live RAG ungrounded→handoff in {elapsed_ms:.2f}ms "
                    f"(flag={grounded_flag}, dropped_cites={len(citations)})"
                )
                return _rag_miss_handoff(query, language, reason="ungrounded")
        # --- END MODIFICATION ---

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Live RAG response formulated in {elapsed_ms:.2f}ms with {len(citations)} citations."
        )

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "status": "success",
            "handoff": False,
        }

    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")
        return {
            "query": query,
            "answer": f"Đã xảy ra lỗi trong quá trình tra cứu dữ liệu: {e!s}",
            "citations": [],
            "status": "error",
            "handoff": False,
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
        "handoff": False,
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
    skipped_items = 0

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No .json files found in input directory: {args.input}")

    # --- START MODIFICATION ---
    # Pre-count for progress logs; accept str or {query, language} objects.
    planned: List[tuple[str, str | None]] = []
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
            if isinstance(item, str):
                planned.append((item, None))
            elif isinstance(item, dict) and isinstance(item.get("query"), str):
                lang = item.get("language")
                if lang is not None and lang not in ("vi", "en"):
                    logger.warning(
                        f"Skipping invalid language in {query_file}: {lang!r}"
                    )
                    skipped_items += 1
                    continue
                planned.append((item["query"], lang if isinstance(lang, str) else None))
            else:
                logger.warning(
                    f"Skipping unsupported query item in {query_file}: {item!r}"
                )
                skipped_items += 1

    total_planned = len(planned)
    for idx, (query, language) in enumerate(planned, start=1):
        preview = query if len(query) <= 72 else query[:69] + "..."
        print(f"[{idx}/{total_planned}] Processing: {preview}", flush=True)
        logger.info(f"[{idx}/{total_planned}] Processing: {preview}")
        total_queries += 1
        if language is None:
            result = solve_automotive_query_auto(query)
        else:
            result = solve_automotive_query_auto(query, language=language)
        results.append(result)
    # --- END MODIFICATION ---

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Batch evaluation finished: processed {len(json_files)} file(s), "
        f"ran {total_queries} query(ies), skipped {skipped_files} file(s), "
        f"skipped {skipped_items} item(s). "
        f"Results written to: {args.output}"
    )

