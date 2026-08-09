"""Dynamic structured query planner for catalog vs procedure RAG routing."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Literal

from core.config import settings
from utils.logger import setup_logger
from utils.vehicle_meta import _norm, canonical_model

logger = setup_logger("query_planner")

# --- START MODIFICATION ---
Intent = Literal["catalog", "procedure", "chitchat", "refuse"]

PLANNER_SYSTEM_PROMPT = (
    "You are an automotive RAG query planner. "
    "Given a driver utterance (and optional prior conversation), output ONLY valid JSON "
    "(no markdown) with keys:\n"
    '  intent: "catalog" | "procedure" | "chitchat" | "refuse"\n'
    "  make: lowercase brand or empty string\n"
    "  model: lowercase model (e.g. \"santa fe\", \"accent\") or empty string\n"
    "  year: four-digit year or empty string\n"
    "  search_query: one concise English owner's-manual search phrase "
    "(for procedure); for catalog use a short vehicle label\n"
    "Rules:\n"
    "- catalog = counting/listing manuals/docs/PDFs about a vehicle or corpus\n"
    "- procedure = how-to / specs / emergency / features from manuals\n"
    "- chitchat = greetings, thanks, unrelated small talk\n"
    "- refuse = unsafe bypass/jailbreak requests\n"
    "- Normalize typos (Santafe→santa fe). Never invent repair steps.\n"
    "- If prior conversation is provided and the current utterance uses anaphora "
    "(it/that/this/cái đó/chức năng đó/nữa/món này), resolve the referent into "
    "search_query as a standalone manual phrase (e.g. heated seat shutoff duration). "
    "Do not leave pronouns in search_query.\n"
    "- For procedure queries with OEM acronyms (EPB, AEB, ISOFIX), put the English "
    "OEM term in search_query (e.g. electronic parking brake / EPB switch).\n"
    "- make is the OEM brand (hyundai/ford/…); model is the product line "
    "(tucson/ioniq 5/bronco). Never put a model name in make.\n"
    "- Empty string for unknown fields. No null."
)

_ANAPHORA_RE = re.compile(
    r"\b("
    r"it|that|this|those|these|"
    r"cai\s*do|chuc\s*nang\s*(do|day|nay)|mon\s*(nay|do)|"
    r"nua|same|again|the\s+number|bao\s*lau"
    r")\b",
    re.IGNORECASE,
)

_LLM_PROVIDERS = frozenset(
    {"ollama", "openai_compatible", "openai", "lmstudio", "bedrock"}
)
_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_CATALOG_FALLBACK = re.compile(
    r"(bao\s*nhieu\s*tai\s*lieu|liet\s*ke.*(manual|tai\s*lieu|pdf)|"
    r"tai\s*lieu\s*lien\s*quan|how\s*many\s*(documents?|manuals?|pdfs?)|"
    r"list\s*(all\s*)?(docs?|documents?|manuals?|pdfs?))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlannedQuery:
    intent: Intent
    make: str
    model: str
    year: str
    search_query: str
    source: str  # "llm" | "fallback"


def planner_enabled() -> bool:
    if not bool(getattr(settings, "rag_planner_enabled", True)):
        return False
    provider = (settings.llm_provider or "none").strip().lower()
    return provider in _LLM_PROVIDERS


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    # First {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _sanitize_plan(data: dict[str, Any], original: str) -> PlannedQuery | None:
    intent_raw = str(data.get("intent") or "").strip().lower()
    if intent_raw not in {"catalog", "procedure", "chitchat", "refuse"}:
        return None
    make = _norm(str(data.get("make") or ""))
    model = _norm(
        canonical_model(str(data.get("model") or "")) if data.get("model") else ""
    )
    year = str(data.get("year") or "").strip()
    if year and not re.fullmatch(r"(19|20)\d{2}", year):
        year = ""
    search_query = str(data.get("search_query") or "").strip()
    if not search_query:
        search_query = original.strip()
    if len(search_query) > 240:
        search_query = search_query[:240]
    return PlannedQuery(
        intent=intent_raw,  # type: ignore[arg-type]
        make=make,
        model=model,
        year=year,
        search_query=search_query,
        source="llm",
    )


def _call_openai_compatible(
    query: str, timeout_s: float, conversation_context: str = ""
) -> str:
    from openai import OpenAI

    base_url = (settings.openai_base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL required for planner")
    client = OpenAI(
        api_key=settings.openai_api_key or "ollama",
        base_url=base_url,
        timeout=timeout_s,
    )
    user_content = query
    if (conversation_context or "").strip():
        user_content = (
            f"Prior conversation:\n{conversation_context.strip()}\n\n"
            f"Current utterance:\n{query}"
        )
    completion = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.0,
        max_tokens=120,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def _call_bedrock(query: str, conversation_context: str = "") -> str:
    from core.aws_client import get_bedrock_runtime_client

    client = get_bedrock_runtime_client()
    model_id = settings.bedrock_model_id or "global.amazon.nova-2-lite-v1:0"
    user_text = query
    if (conversation_context or "").strip():
        user_text = (
            f"Prior conversation:\n{conversation_context.strip()}\n\n"
            f"Current utterance:\n{query}"
        )
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        system=[{"text": PLANNER_SYSTEM_PROMPT}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 120, "topP": 0.9},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def _invoke_planner_llm(
    query: str, timeout_s: float, conversation_context: str = ""
) -> str:
    provider = (settings.llm_provider or "none").strip().lower()
    if provider == "bedrock":
        return _call_bedrock(query, conversation_context=conversation_context)
    if provider in {"ollama", "openai_compatible", "openai", "lmstudio"}:
        return _call_openai_compatible(
            query, timeout_s, conversation_context=conversation_context
        )
    raise ValueError(f"Unsupported planner provider: {provider}")


def _fallback_plan(
    query: str, conversation_context: str = ""
) -> PlannedQuery:
    """Regex/heuristic planner when LLM is down — not the happy path."""
    from utils.query_expand import fold_vi
    from utils.vehicle_meta import KNOWN_MAKES, MODEL_ALIASES, parse_vehicle_metadata

    folded = fold_vi(query)
    # --- START MODIFICATION ---
    # Do not use a fake "unknown.pdf" path — it pollutes model with "unknown …"
    meta = parse_vehicle_metadata("query.txt", doc_name=query)
    # --- END MODIFICATION ---
    # Also scan query text via filename-style parser
    make, model, year = meta["make"], meta["model"], meta["year"]
    # Lightweight vehicle sniff from known aliases in query
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    if not model:
        for key, canonical in sorted(MODEL_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if key.replace(" ", "") in compact or key in folded:
                model = canonical
                break
    if not make:
        for known in KNOWN_MAKES:
            if known in compact:
                make = known
                break
    y = re.search(r"\b((?:19|20)\d{2})\b", folded)
    if y:
        year = y.group(1)

    # Drop polluted heuristic models (e.g. "unknown thong" from VI filler tokens)
    model_n = _norm(model)
    if model_n.startswith("unknown") or model_n in {"thong", "he", "lam", "huong", "quy"}:
        model = ""
    if make == "unknown":
        make = ""

    if _CATALOG_FALLBACK.search(folded):
        intent: Intent = "catalog"
        search_query = f"{make} {model} {year}".strip() or query
    else:
        intent = "procedure"
        search_query = query
        # Anaphora heuristic: stitch last Driver topic when pronouns present
        ctx = (conversation_context or "").strip()
        if ctx and _ANAPHORA_RE.search(folded):
            last_driver = ""
            for line in ctx.splitlines():
                if line.lower().startswith("driver:"):
                    last_driver = line.split(":", 1)[-1].strip()
            if last_driver:
                search_query = f"{last_driver} {query}".strip()

    return PlannedQuery(
        intent=intent,
        make=_norm(make),
        model=_norm(model),
        year=str(year or "").strip(),
        search_query=search_query.strip() or query,
        source="fallback",
    )


def plan_query(
    query: str,
    language: str | None = "vi",
    conversation_context: str = "",
) -> PlannedQuery:
    """
    Produce structured routing plan. Always returns a plan (LLM or fallback).
    When conversation_context is set, resolve anaphora into search_query.
    """
    _ = language
    original = (query or "").strip()
    ctx = (conversation_context or "").strip()
    if not original:
        return PlannedQuery("procedure", "", "", "", "", "fallback")

    if not planner_enabled():
        return _fallback_plan(original, conversation_context=ctx)

    timeout_s = float(getattr(settings, "rag_planner_timeout_s", 4.0))
    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _invoke_planner_llm, original, timeout_s, ctx
            )
            raw = future.result(timeout=timeout_s)
        data = _extract_json_object(raw)
        planned = _sanitize_plan(data, original) if data else None
        ms = int((time.perf_counter() - start) * 1000)
        if planned:
            logger.info(
                f"[PLANNER] intent={planned.intent} make={planned.make!r} "
                f"model={planned.model!r} year={planned.year!r} "
                f"search={planned.search_query!r} ctx={bool(ctx)} ms={ms}"
            )
            return planned
        logger.warning(f"[PLANNER] Invalid JSON after {ms}ms raw={raw[:200]!r}")
    except FuturesTimeoutError:
        ms = int((time.perf_counter() - start) * 1000)
        logger.warning(f"[PLANNER] Timeout after {ms}ms (cap={timeout_s}s)")
    except Exception as exc:  # noqa: BLE001
        ms = int((time.perf_counter() - start) * 1000)
        logger.warning(f"[PLANNER] Failed in {ms}ms: {exc}")

    fb = _fallback_plan(original, conversation_context=ctx)
    logger.info(
        f"[PLANNER] fallback intent={fb.intent} make={fb.make!r} "
        f"model={fb.model!r} year={fb.year!r} search={fb.search_query!r}"
    )
    return fb


# --- END MODIFICATION ---
