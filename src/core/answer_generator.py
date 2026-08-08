"""Unified answer generation for driver-facing RAG responses."""

from __future__ import annotations

import json
import re
from typing import Protocol

from openai import OpenAI

from core.aws_client import get_bedrock_runtime_client
from core.config import settings
from utils.logger import setup_logger

logger = setup_logger("answer_generator")

# Strip bibliography-style prefixes so TTS never reads file paths/spec codes.
_BRACKET_META_RE = re.compile(r"\[[^\]]{0,200}\]\s*:?\s*")
_LEADING_SNIPPET_IDX_RE = re.compile(r"^\s*\d+\.\s*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


class AnswerGenerator(Protocol):
    def generate(self, query: str, context_snippets: list[str]) -> str: ...


def build_context_block(context_snippets: list[str], max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else settings.rag_context_chars
    parts: list[str] = []
    used = 0
    for idx, snippet in enumerate(context_snippets, start=1):
        chunk = snippet.strip()
        if not chunk:
            continue
        piece = f"[{idx}] {chunk}"
        if used + len(piece) > limit and parts:
            break
        if used + len(piece) > limit:
            piece = piece[: max(0, limit - used)]
        parts.append(piece)
        used += len(piece)
        if used >= limit:
            break
    return "\n\n".join(parts)


def clean_chunk_for_speech(text: str) -> str:
    """Remove citation headers / bracketed metadata before extractive summary."""
    cleaned = _LEADING_SNIPPET_IDX_RE.sub("", text.strip())
    cleaned = _BRACKET_META_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extractive_summary(
    context_snippets: list[str],
    max_sentences: int = 2,
    language: str | None = "vi",
) -> str:
    from core.locale_messages import not_found_answer

    if not context_snippets:
        return not_found_answer(language)

    cleaned = clean_chunk_for_speech(context_snippets[0])
    if not cleaned:
        return not_found_answer(language)

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned[:220]

    summary = " ".join(sentences[:max_sentences])
    if len(summary) > 320:
        summary = summary[:317].rstrip() + "..."
    return summary


class ExtractiveFallbackGenerator:
    def __init__(self, language: str | None = "vi") -> None:
        self.language = language

    def generate(self, query: str, context_snippets: list[str]) -> str:
        _ = query
        return extractive_summary(context_snippets, language=self.language)


class BedrockAnswerGenerator:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or settings.system_prompt

    def generate(self, query: str, context_snippets: list[str]) -> str:
        context_str = build_context_block(context_snippets)
        if context_str:
            user_content = (
                f"Ngữ cảnh tài liệu (Context):\n{context_str}\n\nCâu hỏi: {query}"
            )
        else:
            user_content = query

        bedrock_client = get_bedrock_runtime_client()
        model_id = settings.bedrock_model_id or "global.amazon.nova-2-lite-v1:0"
        system_prompt = self.system_prompt

        try:
            response = bedrock_client.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_content}],
                    }
                ],
                system=[{"text": system_prompt}],
                inferenceConfig={
                    "temperature": settings.llm_temperature,
                    "maxTokens": settings.llm_max_tokens,
                    "topP": settings.llm_top_p,
                },
            )
            return response["output"]["message"]["content"][0]["text"].strip()
        except Exception as converse_err:
            logger.warning(
                "Bedrock Converse failed (%s); trying invoke_model fallback",
                converse_err,
            )
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
            }
            response = bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            response_body = json.loads(response["body"].read())
            return response_body.get("content", [{}])[0].get("text", "").strip()


class OpenAICompatibleAnswerGenerator:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or settings.system_prompt

    def generate(self, query: str, context_snippets: list[str]) -> str:
        context_str = build_context_block(context_snippets)
        if context_str:
            user_content = (
                f"Ngữ cảnh tài liệu (Context):\n{context_str}\n\nCâu hỏi: {query}"
            )
        else:
            user_content = query

        base_url = (settings.openai_base_url or "").rstrip("/")
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is required for openai_compatible provider")

        client = OpenAI(
            api_key=settings.openai_api_key or "ollama",
            base_url=base_url,
        )
        completion = client.chat.completions.create(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return (completion.choices[0].message.content or "").strip()


def get_answer_generator(
    provider: str | None = None,
    system_prompt: str | None = None,
    language: str | None = "vi",
) -> AnswerGenerator:
    from core.locale_messages import rag_system_prompt

    resolved = (provider or settings.llm_provider or "none").strip().lower()
    prompt = system_prompt or rag_system_prompt(language)
    if resolved == "bedrock":
        return BedrockAnswerGenerator(system_prompt=prompt)
    if resolved in {"openai_compatible", "openai", "ollama", "lmstudio"}:
        return OpenAICompatibleAnswerGenerator(system_prompt=prompt)
    return ExtractiveFallbackGenerator(language=language)


def generate_driver_answer(
    query: str,
    context_snippets: list[str],
    provider: str | None = None,
    language: str | None = "vi",
) -> str:
    """
    Produce a short driver-facing answer. Falls back to extractive summary on LLM errors.
    Answer language follows UI `language`, not the query language.
    """
    if not context_snippets:
        return extractive_summary([], language=language)

    generator = get_answer_generator(provider, language=language)
    try:
        answer = generator.generate(query, context_snippets)
        if answer and answer.strip():
            return answer.strip()
    except Exception as exc:
        logger.warning("Answer generation failed (%s); using extractive fallback", exc)

    return extractive_summary(context_snippets, language=language)


def generate_free_talk_answer(query: str, language: str | None = "vi") -> str:
    """Casual reply without RAG context; never invent vehicle procedures."""
    from core.locale_messages import (
        free_talk_llm_down,
        free_talk_no_llm,
        free_talk_system_prompt,
    )

    provider = (settings.llm_provider or "none").strip().lower()
    if provider in {"", "none"}:
        return free_talk_no_llm(language)

    generator = get_answer_generator(
        provider=provider,
        system_prompt=free_talk_system_prompt(language),
        language=language,
    )
    try:
        answer = generator.generate(query, [])
        if answer and answer.strip():
            return answer.strip()
    except Exception as exc:
        # --- START MODIFICATION ---
        # Connection errors mean free_talk mode worked; Ollama is down — do not
        # pretend the product is "manual-only".
        logger.warning("Free-talk generation failed (%s); using LLM-down message", exc)
        return free_talk_llm_down(language)
        # --- END MODIFICATION ---

    return free_talk_no_llm(language)


def generate_rag_miss_handoff(query: str, language: str | None = "vi") -> str:
    """
    Constrained free-talk after RAG miss / ungrounded / cite-vehicle mismatch.
    Must never invent manual procedures.
    """
    # --- START MODIFICATION ---
    from core.locale_messages import (
        free_talk_llm_down,
        rag_miss_handoff_fallback,
        rag_miss_handoff_system_prompt,
    )

    provider = (settings.llm_provider or "none").strip().lower()
    if provider in {"", "none"}:
        return rag_miss_handoff_fallback(language)

    generator = get_answer_generator(
        provider=provider,
        system_prompt=rag_miss_handoff_system_prompt(language),
        language=language,
    )
    try:
        answer = generator.generate(query, [])
        if answer and answer.strip():
            return answer.strip()
    except Exception as exc:
        logger.warning("RAG-miss handoff generation failed (%s)", exc)
        return free_talk_llm_down(language)

    return rag_miss_handoff_fallback(language)
    # --- END MODIFICATION ---
