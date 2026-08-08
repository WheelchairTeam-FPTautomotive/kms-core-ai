"""Corpus inventory answers for catalog intent (metadata aggregation)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from utils.logger import setup_logger
from utils.vehicle_meta import build_chroma_where

logger = setup_logger("corpus_catalog")

# --- START MODIFICATION ---
_CACHE: dict[str, Any] = {"ts": 0.0, "names": None, "ttl": 60.0}


def _distinct_document_names(
    collection: Any,
    where: dict | None = None,
) -> list[str]:
    """Scan Chroma metadatas for unique document_name values."""
    names: set[str] = set()
    kwargs: dict[str, Any] = {"include": ["metadatas"]}
    if where:
        kwargs["where"] = where
    try:
        batch = collection.get(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"catalog get failed where={where}: {exc}")
        return []
    for meta in batch.get("metadatas") or []:
        if not meta:
            continue
        name = meta.get("document_name") or ""
        if name:
            names.add(str(name))
    return sorted(names)


def list_documents_for_vehicle(
    get_collection: Callable[[], Any],
    *,
    make: str = "",
    model: str = "",
    year: str = "",
) -> list[str]:
    """
    Distinct manuals matching cascading filters:
    make+model+year → make+model → make → all.
    """
    collection = get_collection()
    attempts = [
        build_chroma_where(make, model, year),
        build_chroma_where(make, model, None) if year else None,
        build_chroma_where(make, None, None) if make else None,
        None,
    ]
    seen_where: set[str] = set()
    for where in attempts:
        key = repr(where)
        if key in seen_where:
            continue
        seen_where.add(key)
        names = _distinct_document_names(collection, where)
        if names:
            return names
        if where is None:
            return names
    return []


def format_catalog_answer(
    names: list[str],
    *,
    language: str | None,
    make: str,
    model: str,
    year: str,
) -> str:
    from core.locale_messages import normalize_language

    lang = normalize_language(language)
    vehicle = " ".join(x for x in (make.title() if make else "", model.title() if model else "", year) if x).strip()
    if lang == "en":
        if not names:
            if vehicle:
                return (
                    f"No manuals were found in the index for {vehicle}. "
                    "Try another model year or ask a how-to question from the owner's manual."
                )
            return (
                "The knowledge index has no documents yet. "
                "Ask about a specific vehicle model after ingestion."
            )
        if vehicle:
            head = f"Found {len(names)} related document(s) for {vehicle}:"
        else:
            head = f"Found {len(names)} document(s) in the knowledge index:"
        body = "\n".join(f"- {n}" for n in names[:40])
        more = "" if len(names) <= 40 else f"\n… and {len(names) - 40} more."
        return f"{head}\n{body}{more}"

    # Vietnamese default
    if not names:
        if vehicle:
            return (
                f"Không tìm thấy tài liệu nào trong chỉ mục cho {vehicle}. "
                "Thử model/năm khác hoặc hỏi thao tác cụ thể trong manual."
            )
        return (
            "Chỉ mục hiện chưa có tài liệu. "
            "Hãy hỏi theo model xe cụ thể sau khi ingest."
        )
    if vehicle:
        head = f"Có {len(names)} tài liệu liên quan đến {vehicle}:"
    else:
        head = f"Có {len(names)} tài liệu trong chỉ mục kiến thức:"
    body = "\n".join(f"- {n}" for n in names[:40])
    more = "" if len(names) <= 40 else f"\n… và {len(names) - 40} tài liệu nữa."
    return f"{head}\n{body}{more}"


def catalog_citations(names: list[str]) -> list[dict[str, Any]]:
    cites: list[dict[str, Any]] = []
    for name in names[:20]:
        cites.append(
            {
                "document_id": "",
                "document_name": name,
                "section": "catalog",
                "page": 0,
                "matched_text": name,
            }
        )
    return cites


# --- END MODIFICATION ---
