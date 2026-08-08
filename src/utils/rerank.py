"""Local cross-encoder rerank for hybrid RAG candidate pools."""

from __future__ import annotations

import threading
import time
from typing import Any

from core.config import settings
from utils.logger import setup_logger

logger = setup_logger("rag_rerank")

_LOCK = threading.RLock()
_MODEL = None
_WARM = False


def _model_name() -> str:
    return (
        getattr(settings, "rag_rerank_model", None)
        or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def get_cross_encoder():
    global _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        if not bool(getattr(settings, "rag_rerank_enabled", True)):
            return None
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # noqa: BLE001
            logger.warning("sentence-transformers unavailable (%s); rerank off", exc)
            return None
        name = _model_name()
        logger.info("Loading cross-encoder %s …", name)
        _MODEL = CrossEncoder(name)
        return _MODEL


def warm_cross_encoder() -> None:
    """Startup warm-up: load model + 1-token dummy pair (avoid first-query spike)."""
    global _WARM
    if _WARM:
        return
    if not bool(getattr(settings, "rag_rerank_enabled", True)):
        return
    try:
        model = get_cross_encoder()
        if model is None:
            return
        model.predict([("ping", "pong")])
        _WARM = True
        logger.info("Cross-encoder warm-up completed")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cross-encoder warm-up failed: %s", exc)


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Reorder by ce_score. Preserves dense_distance / rrf_score on each payload.
    """
    if not candidates:
        return []
    if not bool(getattr(settings, "rag_rerank_enabled", True)):
        return candidates[: (top_k or len(candidates))]

    keep = int(top_k if top_k is not None else getattr(settings, "rag_top_k", 3) or 3)
    model = get_cross_encoder()
    if model is None:
        return candidates[:keep]

    pairs = [
        (query, str(c.get("matched_text") or c.get("text") or "")[:1200])
        for c in candidates
    ]
    t0 = time.perf_counter()
    try:
        scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rerank failed (%s); keeping RRF order", exc)
        return candidates[:keep]
    ms = int((time.perf_counter() - t0) * 1000)

    scored: list[dict[str, Any]] = []
    for cand, score in zip(candidates, scores):
        row = dict(cand)
        row["ce_score"] = float(score)
        scored.append(row)
    scored.sort(key=lambda c: float(c.get("ce_score") or -1e9), reverse=True)
    logger.info(
        "rerank ms=%s pool=%s keep=%s top_ce=%.4f",
        ms,
        len(candidates),
        keep,
        float(scored[0]["ce_score"]) if scored else 0.0,
    )
    return scored[:keep]
