"""Conditional LLM rewrite of user queries into English manual search phrases."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from core.config import settings
from utils.logger import setup_logger

logger = setup_logger("query_rewrite")

# --- START MODIFICATION ---
# Zero-chatter rewriter for cross-lingual RAG miss rescue (VI → EN OEM manuals).
REWRITE_SYSTEM_PROMPT = (
    "You are an automotive manual search query rewriter. "
    "Convert the user's vehicle question into ONE concise English search phrase "
    "optimized for an owner's manual index. "
    "Output ONLY the raw search string. Do NOT answer the question. "
    "Do NOT include explanations, quotes, or markdown."
)

_PREFIX_RE = re.compile(
    r"^(?:"
    r"search\s*query|query|rewritten|output|english|result|"
    r"here(?:'s| is)(?:\s+your)?(?:\s+(?:search\s*)?query)?"
    r")\s*[:\-–]\s*",
    re.IGNORECASE,
)
_LLM_REWRITE_PROVIDERS = frozenset(
    {"ollama", "openai_compatible", "openai", "lmstudio", "bedrock"}
)


def sanitize_rewrite_output(raw: str | None) -> str | None:
    """Strip chatty LLM wrappers so Chroma gets a pure search phrase."""
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    # First non-empty line only
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            text = candidate
            break
    else:
        return None

    text = text.strip("`").strip()
    text = text.strip('"').strip("'").strip()
    text = _PREFIX_RE.sub("", text).strip()
    text = text.strip('"').strip("'").strip()
    if not text or len(text) > 240:
        return None
    # Reject obvious answers / multi-sentence chatter
    if text.count(".") >= 2 and len(text) > 80:
        return None
    return text


def rewrite_enabled() -> bool:
    if not bool(getattr(settings, "rag_rewrite_on_miss", True)):
        return False
    provider = (settings.llm_provider or "none").strip().lower()
    return provider in _LLM_REWRITE_PROVIDERS


def _call_openai_compatible(query: str) -> str:
    from openai import OpenAI

    base_url = (settings.openai_base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL required for rewrite")
    client = OpenAI(
        api_key=settings.openai_api_key or "ollama",
        base_url=base_url,
        timeout=float(getattr(settings, "rag_rewrite_timeout_s", 2.0)),
    )
    completion = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.0,
        max_tokens=40,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def _call_bedrock(query: str) -> str:
    from core.aws_client import get_bedrock_runtime_client

    client = get_bedrock_runtime_client()
    model_id = settings.bedrock_model_id or "global.amazon.nova-2-lite-v1:0"
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": query}]}],
        system=[{"text": REWRITE_SYSTEM_PROMPT}],
        inferenceConfig={
            "temperature": 0.0,
            "maxTokens": 40,
            "topP": 0.9,
        },
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def _invoke_rewrite_llm(query: str) -> str:
    provider = (settings.llm_provider or "none").strip().lower()
    if provider == "bedrock":
        return _call_bedrock(query)
    if provider in {"ollama", "openai_compatible", "openai", "lmstudio"}:
        return _call_openai_compatible(query)
    raise ValueError(f"Unsupported rewrite provider: {provider}")


def rewrite_retrieval_query(query: str, language: str | None = "vi") -> str | None:
    """
    Rewrite a driver question into one English manual search phrase.
    Returns None on disable / timeout / empty / chatter. Retrieval-only.
    """
    _ = language
    original = (query or "").strip()
    if not original or not rewrite_enabled():
        return None

    timeout_s = float(getattr(settings, "rag_rewrite_timeout_s", 2.0))
    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_invoke_rewrite_llm, original)
            raw = future.result(timeout=timeout_s)
        cleaned = sanitize_rewrite_output(raw)
        rewrite_ms = int((time.perf_counter() - start) * 1000)
        if not cleaned:
            logger.info(
                f"[RAG] Rewrite produced empty/chatter output in {rewrite_ms}ms "
                f"raw={raw!r}"
            )
            return None
        logger.info(
            f'[RAG] Original: "{original}" -> Rewritten: "{cleaned}" '
            f"(rewrite_ms={rewrite_ms})"
        )
        return cleaned
    except FuturesTimeoutError:
        rewrite_ms = int((time.perf_counter() - start) * 1000)
        logger.warning(f"[RAG] Rewrite timeout after {rewrite_ms}ms (cap={timeout_s}s)")
        return None
    except Exception as exc:
        rewrite_ms = int((time.perf_counter() - start) * 1000)
        logger.warning(f"[RAG] Rewrite failed in {rewrite_ms}ms: {exc}")
        return None


# --- END MODIFICATION ---
