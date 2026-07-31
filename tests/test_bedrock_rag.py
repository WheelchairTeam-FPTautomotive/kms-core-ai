from unittest.mock import MagicMock, patch

from pipelines.bedrock_rag import solve_automotive_query_bedrock


def test_bedrock_rag_unsafe_refusal():
    result = solve_automotive_query_bedrock("How to hack engine safety")
    assert result["status"] == "refused"
    assert "từ chối" in result["answer"]
    assert result["citations"] == []


@patch("pipelines.bedrock_rag.get_opensearch_client")
@patch("pipelines.bedrock_rag.generate_bedrock_embeddings")
@patch("pipelines.bedrock_rag.get_bedrock_runtime_client")
def test_bedrock_rag_success(mock_get_bedrock_client, mock_generate_embeddings, mock_get_opensearch):
    mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

    mock_opensearch = MagicMock()
    mock_get_opensearch.return_value = mock_opensearch
    mock_opensearch.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 0.98,
                    "_source": {
                        "document_id": "doc_adas_01",
                        "document_name": "ADAS_Manual.pdf",
                        "section": "Chương 5: Phanh khẩn cấp",
                        "page": 45,
                        "text": "Hệ thống phanh AEB sẽ tự động kích hoạt khi phát hiện vật cản gần.",
                    },
                }
            ]
        }
    }

    mock_runtime = MagicMock()
    mock_get_bedrock_client.return_value = mock_runtime
    mock_runtime.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "Phanh AEB tự động kích hoạt khi có vật cản."}]
            }
        }
    }
    mock_response_body = MagicMock()
    mock_response_body.read.return_value = '{"content": [{"text": "Phanh AEB tự động kích hoạt khi có vật cản."}]}'.encode("utf-8")

    mock_runtime.invoke_model.return_value = {"body": mock_response_body}


    result = solve_automotive_query_bedrock("Làm thế nào kích hoạt phanh khẩn cấp ADAS?")

    assert result["status"] == "success"
    assert "Phanh AEB" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["document_id"] == "doc_adas_01"
    assert result["citations"][0]["page"] == 45
