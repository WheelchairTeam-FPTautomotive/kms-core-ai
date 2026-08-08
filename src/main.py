from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.config import settings
from pipelines.solve_problem import solve_automotive_query_auto
from utils.logger import setup_logger

logger = setup_logger("kms_core_api")


def _warm_llm_background() -> None:
    """One-token Ollama ping so first demo query is not cold."""
    provider = (settings.llm_provider or "none").strip().lower()
    if provider not in {"openai_compatible", "openai", "ollama", "lmstudio"}:
        return
    if not (settings.openai_base_url or "").strip():
        return
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM warm-up skipped/failed: %s", exc)


def _warm_retrieval_background() -> None:
    """Load BM25 sidecar + cross-encoder so first RAG query is not cold."""
    # --- START MODIFICATION ---
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
    # --- END MODIFICATION ---


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import threading

    threading.Thread(target=_warm_llm_background, daemon=True).start()
    threading.Thread(target=_warm_retrieval_background, daemon=True).start()
    yield


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


@app.get("/health")
async def root_health_check():
    return {"status": "ready", "service": "kms-core-ai"}


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ready", "service": "kms-core-ai"}


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
