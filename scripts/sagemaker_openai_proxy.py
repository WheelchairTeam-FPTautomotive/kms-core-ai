"""Local OpenAI-compatible proxy for a private SageMaker LLM endpoint.

Useful for parity windows: point any OpenAI client (e.g. kms-core-ai's
`LLM_PROVIDER=openai_compatible`) at this proxy, and it forwards chat
completion requests to SageMaker Runtime via boto3.

Required environment variables:
  - SAGEMAKER_LLM_ENDPOINT_NAME

Optional:
  - SAGEMAKER_REGION (default: ap-southeast-2)
  - PROXY_HOST (default: 127.0.0.1)
  - PROXY_PORT (default: 8002)
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.config import Config
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from uvicorn import run

APP = FastAPI(title="SageMaker OpenAI Proxy")

DEFAULT_REGION = "ap-southeast-2"
DEFAULT_MODEL_PATH = "/opt/ml/model"

BOTO_CONFIG = Config(
    connect_timeout=10,
    read_timeout=120,
    retries={"max_attempts": 2},
)


def _get_endpoint_name() -> str:
    name = os.getenv("SAGEMAKER_LLM_ENDPOINT_NAME", "")
    if not name:
        raise RuntimeError("SAGEMAKER_LLM_ENDPOINT_NAME environment variable is required")
    return name


def _get_region() -> str:
    return os.getenv("SAGEMAKER_REGION") or os.getenv("AWS_REGION") or DEFAULT_REGION


def _get_runtime_client() -> boto3.client:
    return boto3.client(
        "sagemaker-runtime",
        region_name=_get_region(),
        config=BOTO_CONFIG,
    )


def _build_sagemaker_body(openai_payload: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI chat-completion request into the vLLM body used by the endpoint."""
    return {
        "model": DEFAULT_MODEL_PATH,
        "messages": openai_payload.get("messages", []),
        "max_tokens": openai_payload.get("max_tokens", 256),
        "temperature": openai_payload.get("temperature", 0.7),
        "top_p": openai_payload.get("top_p", 1.0),
    }


@APP.get("/health")
async def health() -> Response:
    return Response(content='{"status":"ok"}', media_type="application/json")


@APP.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    openai_payload = await request.json()
    body = _build_sagemaker_body(openai_payload)

    try:
        client = _get_runtime_client()
        response = client.invoke_endpoint(
            EndpointName=_get_endpoint_name(),
            Body=json.dumps(body).encode("utf-8"),
            ContentType="application/json",
            Accept="application/json",
        )
        raw = response["Body"].read()
        completion = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"SageMaker invocation failed: {exc}", "type": "sagemaker_error"}},
        )

    return JSONResponse(content=completion)


if __name__ == "__main__":
    host = os.getenv("PROXY_HOST", "127.0.0.1")
    port = int(os.getenv("PROXY_PORT", "8002"))
    run(APP, host=host, port=port)
