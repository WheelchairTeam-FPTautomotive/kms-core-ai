import json
import time
from typing import Any

from core.aws_client import (
    generate_bedrock_embeddings,
    get_bedrock_runtime_client,
    get_opensearch_client,
)
from core.config import settings
from pipelines.solve_problem import check_safety_and_scope
from utils.logger import setup_logger
from utils.opensearch_utils import search_opensearch_knn

logger = setup_logger("bedrock_rag_pipeline")


def solve_automotive_query_bedrock(query: str) -> dict[str, Any]:
    """
    Execute AWS Bedrock Claude 3.5 Sonnet RAG pipeline with Amazon OpenSearch Serverless k-NN retrieval.
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
        hits = search_opensearch_knn(
            client=opensearch_client,
            index_name=settings.opensearch_index,
            query_vector=query_vector,
            top_k=3,
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

        context_str = "\n\n".join(context_snippets)

        # 4. Construct prompt and invoke Bedrock Claude 3.5 Sonnet
        system_prompt = (
            "Bạn là trợ lý AI chuyên gia giải đáp tài liệu kỹ thuật xe hơi. "
            "Hãy trả lời câu hỏi của người dùng DỰA HOÀN TOÀN VÀO thông tin trong ngữ cảnh (Context) được cung cấp dưới đây. "
            "Nếu thông tin trong context không đủ để trả lời, hãy lịch sự thông báo không tìm thấy thông tin trong tài liệu. "
            "Trả lời ngắn gọn, chính xác, khách quan và chuyên nghiệp."
        )

        bedrock_client = get_bedrock_runtime_client()
        model_id = settings.bedrock_model_id or "global.amazon.nova-2-lite-v1:0"

        try:
            response = bedrock_client.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": f"Ngữ cảnh tài liệu (Context):\n{context_str}\n\nCâu hỏi: {query}"}],
                    }
                ],
                system=[{"text": system_prompt}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 1000},
            )
            answer = response["output"]["message"]["content"][0]["text"].strip()
        except Exception as converse_err:
            logger.warning(f"Bedrock Converse API call failed ({converse_err}), attempting invoke_model fallback...")
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.0,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Ngữ cảnh tài liệu (Context):\n{context_str}\n\nCâu hỏi: {query}",
                    }
                ],
            }
            response = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            response_body = json.loads(response["body"].read())
            answer = response_body.get("content", [{}])[0].get("text", "").strip()


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
