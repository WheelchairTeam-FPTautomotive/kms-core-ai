"""Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any

from core.config import settings
from utils.bm25_index import get_bm25_sidecar
from utils.logger import setup_logger

logger = setup_logger("hybrid_retrieve")


def rrf_fuse(
    ranked_lists: list[list[tuple[str, Any]]],
    *,
    k: int = 60,
    limit: int = 12,
) -> list[tuple[str, float, Any]]:
    """
    Fuse ranked lists of (key, payload). Returns (key, rrf_score, best_payload).
    Prefer payload that carries dense_distance when merging duplicates.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, Any] = {}
    for ranked in ranked_lists:
        for rank, (key, payload) in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            prev = payloads.get(key)
            if prev is None:
                payloads[key] = payload
                continue
            # Prefer payload with finite dense_distance
            prev_d = prev.get("dense_distance")
            new_d = payload.get("dense_distance")
            if prev_d is None and new_d is not None or (
                prev_d is not None
                and new_d is not None
                and float(new_d) < float(prev_d)
            ):
                payloads[key] = payload
            # merge scores onto kept payload
            kept = payloads[key]
            kept["rrf_score"] = scores[key]
            if new_d is not None and kept.get("dense_distance") is None:
                kept["dense_distance"] = new_d
            if payload.get("bm25_rank") is not None:
                kept["bm25_rank"] = payload.get("bm25_rank")
            if payload.get("source"):
                sources = set(str(kept.get("source") or "").split("+"))
                sources.discard("")
                sources.add(str(payload["source"]))
                kept["source"] = "+".join(sorted(sources))

    fused = [
        (key, score, {**payloads[key], "rrf_score": score})
        for key, score in scores.items()
    ]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:limit]


def dense_hits_to_ranked(
    raw_hits: list[tuple],
) -> list[tuple[str, dict[str, Any]]]:
    """Chroma tuple → ranked list preserving dense_distance."""
    ranked: list[tuple[str, dict[str, Any]]] = []
    for hit in raw_hits:
        doc_id, doc_name, section, page, text_snippet, distance = hit
        key = f"{doc_id}|{page}|{(text_snippet or '')[:80]}"
        ranked.append(
            (
                key,
                {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "section": section,
                    "page": page,
                    "matched_text": text_snippet,
                    "dense_distance": float(distance),
                    "rrf_score": 0.0,
                    "ce_score": None,
                    "bm25_rank": None,
                    "source": "dense",
                },
            )
        )
    return ranked


def bm25_hits_to_ranked(
    query: str,
    *,
    top_k: int,
    make: str,
    model: str,
    year: str,
) -> list[tuple[str, dict[str, Any]]]:
    sidecar = get_bm25_sidecar()
    if sidecar is None:
        return []
    hits = sidecar.search(
        query, top_k=top_k, make=make, model=model, year=year
    )
    ranked: list[tuple[str, dict[str, Any]]] = []
    for h in hits:
        key = f"{h.chunk_id}|{h.page}|{(h.text or '')[:80]}"
        ranked.append(
            (
                key,
                {
                    "document_id": h.chunk_id.split("_p")[0] if "_p" in h.chunk_id else h.chunk_id,
                    "document_name": h.document_name,
                    "section": h.section,
                    "page": h.page,
                    "matched_text": (h.text or "")[:300],
                    "dense_distance": None,
                    "rrf_score": 0.0,
                    "ce_score": None,
                    "bm25_rank": h.rank,
                    "bm25_score": h.score,
                    "source": "bm25",
                    "make": h.make,
                    "model": h.model,
                    "year": h.year,
                },
            )
        )
    return ranked


def hybrid_retrieve_candidates(
    queries: list[str],
    dense_raw: list[tuple],
    *,
    make: str = "",
    model: str = "",
    year: str = "",
    rrf_k: int | None = None,
    pool: int | None = None,
    bm25_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Fuse dense + BM25 via RRF. Each candidate preserves dense_distance when known.
    """
    if not bool(getattr(settings, "rag_hybrid_enabled", True)):
        # dense-only passthrough
        return [p for _, p in dense_hits_to_ranked(dense_raw)]

    k = int(rrf_k if rrf_k is not None else getattr(settings, "rag_rrf_k", 60) or 60)
    limit = int(
        pool if pool is not None else getattr(settings, "rag_candidate_pool", 12) or 12
    )
    btop = int(
        bm25_top_k
        if bm25_top_k is not None
        else getattr(settings, "rag_bm25_top_k", 20) or 20
    )

    dense_ranked = dense_hits_to_ranked(dense_raw)
    bm25_lists: list[list[tuple[str, dict[str, Any]]]] = []
    # --- START MODIFICATION ---
    # Fan-in 4 so prepended OEM EN phrases are not dropped (RC3)
    for q in queries[:4]:
        bm25_lists.append(
            bm25_hits_to_ranked(q, top_k=btop, make=make, model=model, year=year)
        )
    # --- END MODIFICATION ---

    lists = [dense_ranked, *bm25_lists]
    fused = rrf_fuse(lists, k=k, limit=limit)
    logger.info(
        "hybrid RRF dense=%s bm25_lists=%s fused=%s make=%r model=%r",
        len(dense_ranked),
        [len(x) for x in bm25_lists],
        len(fused),
        make,
        model,
    )
    return [payload for _key, _score, payload in fused]
