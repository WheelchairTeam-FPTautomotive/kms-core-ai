from unittest.mock import MagicMock, patch

from core.aws_client import generate_bedrock_embeddings
from utils.opensearch_utils import (
    ensure_opensearch_index,
    search_opensearch_knn,
)


@patch("boto3.client")
def test_generate_bedrock_embeddings(mock_boto_client):
    mock_runtime = MagicMock()
    mock_boto_client.return_value = mock_runtime

    mock_response_body = MagicMock()
    mock_response_body.read.return_value = b'{"embedding": [0.1, 0.2, 0.3]}'
    mock_runtime.invoke_model.return_value = {"body": mock_response_body}

    texts = ["Test automotive query"]
    embeddings = generate_bedrock_embeddings(texts)

    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
    mock_runtime.invoke_model.assert_called_once()


def test_ensure_opensearch_index():
    mock_opensearch = MagicMock()
    mock_opensearch.indices.exists.return_value = False

    ensure_opensearch_index(mock_opensearch, "test-index", dimension=1024)

    mock_opensearch.indices.exists.assert_called_once_with(index="test-index")
    mock_opensearch.indices.create.assert_called_once()


def test_search_opensearch_knn():
    mock_opensearch = MagicMock()
    mock_opensearch.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 0.95,
                    "_source": {
                        "document_id": "doc123",
                        "document_name": "Manual.pdf",
                        "section": "Chương 1",
                        "page": 10,
                        "text": "Brake system instructions",
                    },
                }
            ]
        }
    }

    results = search_opensearch_knn(mock_opensearch, "test-index", [0.1, 0.2, 0.3], top_k=1)

    assert len(results) == 1
    assert results[0]["document_id"] == "doc123"
    assert results[0]["page"] == 10
    assert results[0]["matched_text"] == "Brake system instructions"
