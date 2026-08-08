from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import settings
from pipelines.solve_problem import solve_automotive_query_auto
from utils.logger import setup_logger

logger = setup_logger("kms_core_api")

# --- START MODIFICATION ---
# Blocking readiness: health stays not-ready until retrieval warm completes.
_RETRIEVAL_READY = False
_LLM_WARMED = False
# --- END MODIFICATION ---


def _warm_llm() -> bool:
    """One-token Ollama ping so first demo query is not cold."""
    provider = (settings.llm_provider or "none").strip().lower()
    if provider not in {"openai_compatible", "openai", "ollama", "lmstudio"}:
        return True
    if not (settings.openai_base_url or "").strip():
        return True
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key or "ollama",
            base_url=settings.openai_base_url.rstrip("/"),
            timeout=30.0,
        )
        client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        logger.info("LLM warm-up completed for provider=%s", provider)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM warm-up skipped/failed: %s", exc)
        return False


def _warm_retrieval() -> bool:
    """
    Load Chroma + BM25 + cross-encoder synchronously.
    Returns True when retrieval path is usable (BM25 optional soft-degrade).
    """
    # --- START MODIFICATION ---
    ok = True
    try:
        from pipelines.solve_problem import get_chroma_collection

        get_chroma_collection().query(query_texts=["warmup"], n_results=1)
        logger.info("Chroma warm-up completed")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chroma warm-up failed: %s", exc)
        ok = False
    try:
        from utils.bm25_index import get_bm25_sidecar

        sc = get_bm25_sidecar()
        logger.info(
            "BM25 warm-up %s",
            f"n_chunks={len(sc)}" if sc is not None else "missing",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 warm-up skipped/failed: %s", exc)
    try:
        from utils.rerank import warm_cross_encoder

        warm_cross_encoder()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cross-encoder warm-up skipped/failed: %s", exc)
    return ok
    # --- END MODIFICATION ---


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # --- START MODIFICATION ---
    global _RETRIEVAL_READY, _LLM_WARMED
    logger.info("Blocking warm-up starting (RC1 readiness gate)…")
    _RETRIEVAL_READY = _warm_retrieval()
    _LLM_WARMED = _warm_llm()
    logger.info(
        "Warm-up done retrieval_ready=%s llm_warmed=%s",
        _RETRIEVAL_READY,
        _LLM_WARMED,
    )
    yield
    # --- END MODIFICATION ---


app = FastAPI(
    title="KMS Core AI RAG Engine",
    description="Internal search microservice serving grounded manual lookups with citations.",
    version="1.0.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., example="How do I activate the HVAC system?")
    # --- START MODIFICATION ---
    mode: Literal["rag", "free_talk"] = Field(
        default="rag",
        description="rag = retrieve+gate+answer; free_talk = LLM/chitchat without documents",
    )
    language: Literal["vi", "en"] = Field(
        default="vi",
        description="UI locale — answer language follows this, not the query language",
    )
    # --- END MODIFICATION ---


class CitationInfo(BaseModel):
    document_id: str
    document_name: str
    section: str
    page: int
    matched_text: str


class SearchResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationInfo]
    status: str
    # --- START MODIFICATION ---
    # Soft RAG→FT handoff telemetry (constrained free-talk; never invent procedures)
    handoff: bool = False
    # --- END MODIFICATION ---


def _health_payload() -> dict:
    # --- START MODIFICATION ---
    status = "ready" if _RETRIEVAL_READY else "starting"
    return {
        "status": status,
        "service": "kms-core-ai",
        "retrieval_ready": _RETRIEVAL_READY,
        "llm_warmed": _LLM_WARMED,
    }
    # --- END MODIFICATION ---


@app.get("/health")
async def root_health_check():
    body = _health_payload()
    if not _RETRIEVAL_READY:
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/api/v1/health")
async def health_check():
    body = _health_payload()
    if not _RETRIEVAL_READY:
        return JSONResponse(status_code=503, content=body)
    return body


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_knowledge_base(payload: SearchRequest):
    try:
        result = solve_automotive_query_auto(
            payload.query,
            mode=payload.mode,
            language=payload.language,
        )
        return result
    except Exception as e:  # noqa: BLE001
        logger.error(f"RAG execution failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"RAG search execution failed: {e!s}"
        )
