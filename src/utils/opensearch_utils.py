from typing import Any

from opensearchpy import OpenSearch, helpers

from utils.logger import setup_logger

logger = setup_logger("opensearch_utils")


def ensure_opensearch_index(client: OpenSearch, index_name: str, dimension: int = 1024) -> None:
    """
    Ensure the OpenSearch Serverless k-NN vector index exists with proper schema mappings.
    """
    try:
        if client.indices.exists(index=index_name):
            logger.info(f"OpenSearch index '{index_name}' already exists.")
            return

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24,
                            },
                        },
                    },
                    "document_id": {"type": "keyword"},
                    "document_name": {"type": "keyword"},
                    "section": {"type": "text"},
                    "page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "text": {"type": "text"},
                }
            },
        }

        client.indices.create(index=index_name, body=index_body)
        logger.info(f"Successfully created OpenSearch k-NN index '{index_name}'.")
    except Exception as e:
        logger.error(f"Failed to create OpenSearch index '{index_name}': {e}")
        raise e


def bulk_index_chunks(
    client: OpenSearch,
    index_name: str,
    chunks: list[Any],
    embeddings: list[list[float]],
) -> int:
    """
    Bulk index document chunks and their corresponding vector embeddings into OpenSearch.
    """
    actions = []
    for chunk, embedding in zip(chunks, embeddings):
        doc_id = chunk.metadata.get("document_id", "doc")
        page = chunk.page
        chunk_idx = chunk.chunk_index
        unique_id = f"{doc_id}_p{page}_c{chunk_idx}"

        action = {
            "_op_type": "index",
            "_index": index_name,
            "_id": unique_id,
            "_source": {
                "vector": embedding,
                "document_id": doc_id,
                "document_name": chunk.metadata.get("document_name", ""),
                "section": chunk.metadata.get("section", f"Trang {page}"),
                "page": page,
                "chunk_index": chunk_idx,
                "text": chunk.text,
            },
        }
        actions.append(action)

    if not actions:
        return 0

    success_count, _ = helpers.bulk(client, actions)
    logger.info(f"Bulk indexed {success_count} documents into OpenSearch index '{index_name}'.")
    return success_count


def search_opensearch_knn(
    client: OpenSearch,
    index_name: str,
    query_vector: list[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Execute k-NN vector similarity search query against OpenSearch Serverless index.
    """
    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        },
    }

    try:
        response = client.search(index=index_name, body=search_query)
        hits = response.get("hits", {}).get("hits", [])
        results: list[dict[str, Any]] = []

        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "document_id": source.get("document_id", ""),
                "document_name": source.get("document_name", "Automotive Manual"),
                "section": source.get("section", f"Trang {source.get('page', 1)}"),
                "page": source.get("page", 1),
                "matched_text": source.get("text", ""),
                "score": hit.get("_score", 0.0),
            })

        return results
    except Exception as e:
        logger.error(f"OpenSearch k-NN query failed: {e}")
        return []
