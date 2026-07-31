from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipelines.solve_problem import solve_automotive_query
from utils.logger import setup_logger

logger = setup_logger("kms_core_api")

app = FastAPI(
    title="KMS Core AI RAG Engine",
    description="Internal search microservice serving grounded manual lookups with citations.",
    version="1.0.0",
)


# ==========================================
# Pydantic Schemas
# ==========================================
class SearchRequest(BaseModel):
    query: str = Field(..., example="How do I activate the HVAC system?")


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


# ==========================================
# Endpoints
# ==========================================
@app.get("/health")
async def root_health_check():
    return {"status": "ready", "service": "kms-core-ai"}


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ready", "service": "kms-core-ai"}


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_knowledge_base(payload: SearchRequest):
    try:
        # Call RAG query solver pipeline
        result = solve_automotive_query(payload.query)
        return result
    except Exception as e:  # noqa: BLE001
        logger.error(f"RAG execution failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"RAG search execution failed: {e!s}"
        )
