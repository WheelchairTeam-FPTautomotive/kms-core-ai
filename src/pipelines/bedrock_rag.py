import time
from typing import Any

from core.answer_generator import generate_driver_answer
from core.aws_client import (
    generate_bedrock_embeddings,
    get_opensearch_client,
)
from core.config import settings
from pipelines.solve_problem import check_safety_and_scope
from utils.logger import setup_logger
from utils.opensearch_utils import search_opensearch_knn

logger = setup_logger("bedrock_rag_pipeline")


def solve_automotive_query_bedrock(query: str) -> dict[str, Any]:
    """
    Execute AWS Bedrock RAG pipeline with Amazon OpenSearch Serverless k-NN retrieval.
    Answer synthesis uses the shared AnswerGenerator (LLM_PROVIDER / GenerationConfig).
    """
    start_time = time.time()
    logger.info(f"AWS Bedrock RAG processing query: '{query}'")

    # 1. Safety and Automotive scope verification
    is_valid, refusal_reason = check_safety_and_scope(query)
    if not is_valid:
        return {
            "query": query,
            "answer": refusal_reason,
            "citations": [],
            "status": "refused",
        }

    try:
        # 2. Generate vector embedding for query via Bedrock Titan
        query_vectors = generate_bedrock_embeddings([query])
        if not query_vectors:
            raise ValueError("Failed to generate vector embedding for query.")
        query_vector = query_vectors[0]

        # 3. Retrieve context chunks from OpenSearch Serverless
        opensearch_client = get_opensearch_client()
        top_k = settings.rag_top_k or 3
        hits = search_opensearch_knn(
            client=opensearch_client,
            index_name=settings.opensearch_index,
            query_vector=query_vector,
            top_k=top_k,
        )

        citations = []
        context_snippets = []
        for idx, hit in enumerate(hits, start=1):
            citations.append({
                "document_id": hit["document_id"],
                "document_name": hit["document_name"],
                "section": hit["section"],
                "page": hit["page"],
                "matched_text": hit["matched_text"][:300],
            })
            context_snippets.append(
                f"Snippet [{idx}]:\n"
                f"- Document ID: {hit['document_id']}\n"
                f"- Document Name: {hit['document_name']}\n"
                f"- Section: {hit['section']}\n"
                f"- Page: {hit['page']}\n"
                f"- Content: {hit['matched_text']}"
            )

        if not hits:
            return {
                "query": query,
                "answer": "Không tìm thấy thông tin phù hợp trong tài liệu kỹ thuật của xe.",
                "citations": [],
                "status": "success",
            }

        # --- START MODIFICATION ---
        # Shared generator: OpenSearch path defaults synthesis to Bedrock when provider is none.
        provider = (settings.llm_provider or "none").strip().lower()
        answer = generate_driver_answer(
            query,
            context_snippets,
            provider="bedrock" if provider == "none" else None,
        )
        # --- END MODIFICATION ---

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Bedrock RAG response generated in {elapsed_ms:.2f}ms with {len(citations)} citations.")

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Bedrock RAG execution failed: {e}")
        return {
            "query": query,
            "answer": f"Đã xảy ra lỗi trong quá trình tra cứu dữ liệu AWS Bedrock: {e!s}",
            "citations": [],
            "status": "error",
        }
