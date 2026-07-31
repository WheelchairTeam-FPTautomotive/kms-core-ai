import json

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

from core.config import settings
from utils.logger import setup_logger

logger = setup_logger("aws_client")


def get_bedrock_runtime_client():
    """
    Initialize Boto3 client for AWS Bedrock Runtime.
    Uses explicit credentials from settings if available, otherwise falls back to IAM environment/role.
    """
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        if settings.aws_session_token:
            kwargs["aws_session_token"] = settings.aws_session_token

    return boto3.client("bedrock-runtime", **kwargs)


def generate_bedrock_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate 1024-dimensional vector embeddings using AWS Bedrock Titan Embeddings v2.
    """
    client = get_bedrock_runtime_client()
    model_id = settings.bedrock_embedding_model_id or "amazon.titan-embed-text-v2:0"
    embeddings: list[list[float]] = []

    for text in texts:
        try:
            body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding", [])
            embeddings.append(embedding)
        except Exception as e:
            logger.error(f"Bedrock embedding generation failed: {e}")
            raise e

    return embeddings


def get_opensearch_client() -> OpenSearch:
    """
    Initialize OpenSearch Serverless client signed with AWS Signature Version 4 (SigV4).
    """
    endpoint = settings.opensearch_endpoint
    region = settings.aws_region

    if not endpoint:
        raise ValueError("OPENSEARCH_ENDPOINT is not configured in settings.")

    host = endpoint.replace("https://", "").replace("http://", "").strip("/")

    kwargs = {"region_name": region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        if settings.aws_session_token:
            kwargs["aws_session_token"] = settings.aws_session_token

    session = boto3.Session(**kwargs)
    credentials = session.get_credentials()
    auth = AWSV4SignerAuth(credentials, region, "aoss")

    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=120,
    )
    return client
