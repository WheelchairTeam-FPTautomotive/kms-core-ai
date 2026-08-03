"""
Isolated mock solver for unit tests and offline development.

This module mirrors the original mock behaviour of `solve_automotive_query`
without depending on the live evaluation pipeline or external vector stores.
It is safe to import from test suites and local debugging scripts.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

UNSAFE_TRIGGERS = [
    "hack",
    "bypass brake",
    "bypass brakes",
    "overdrive engine",
    "overdrive engine safety",
    "ignore seatbelt",
    "ignore seatbelt alert",
    "disable aeb",
    "disable abs",
    "jailbreak",
    "ignore previous instruction",
    "ignore previous instructions",
]


def _fold_vi_local(text: str) -> str:
    import unicodedata

    lowered = text.lower().strip().replace("đ", "d").replace("Đ", "d")
    nfd = unicodedata.normalize("NFD", lowered)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def check_safety_and_scope(query: str) -> tuple[bool, str]:
    """
    Lightweight safety guard duplicated here so the mock solver
    remains self-contained and does not import the live pipeline.
    """
    # --- START MODIFICATION ---
    # Unsafe blocklist only; automotive allowlist removed
    folded = _fold_vi_local(query)
    for trigger in UNSAFE_TRIGGERS:
        if _fold_vi_local(trigger) in folded:
            logger.warning(
                "Unsafe request detected: '%s' triggering: '%s'", query, trigger
            )
            return False, "Yêu cầu bị từ chối vì lý do an toàn vận hành xe."
    return True, ""
    # --- END MODIFICATION ---


def solve_automotive_query(query: str) -> dict[str, Any]:
    """
    Mock RAG solver that returns deterministic automotive citations.

    Use this for:
      - Unit tests that must not depend on ChromaDB/OpenSearch.
      - Offline development when the vector store is unavailable.
      - CI pipelines that need a stable, predictable response shape.
    """
    logger.info("RAG processing query: '%s'", query)

    is_valid, refusal_reason = check_safety_and_scope(query)
    if not is_valid:
        return {
            "query": query,
            "answer": refusal_reason,
            "citations": [],
            "status": "refused",
        }

    logger.info("Executing vector database query...")

    citations = [
        {
            "document_id": "949eb66893b5dbf59aa4b4be35ad330c7b8f0c3802f9ccb8d25881128157bf9c",
            "document_name": "2011 - KMS Manual.pdf",
            "section": "Chương 4: Điều hòa & Hệ thống điện",
            "page": 42,
            "matched_text": (
                "Hệ thống điều hòa (HVAC) được điều khiển qua CarPropertyManager "
                "với AreaId là 0."
            ),
        },
        {
            "document_id": "1ecc7f4e2b438cb0ac5c336fed7cfffbca78b42f87a31a0c0add50aa38cfc751",
            "document_name": "light-control-system.pdf",
            "section": "Chương 7: ADAS & Phanh khẩn cấp",
            "page": 105,
            "matched_text": (
                "Khi xe chạy quá tốc độ 80km/h, hệ thống ADAS kích hoạt phanh "
                "khẩn cấp tự động (AEB) nếu khoảng cách xe trước < 15m."
            ),
        },
    ]

    answer = (
        "Dựa trên tài liệu hướng dẫn kỹ thuật của xe:\n"
        "1. Hệ thống điều hòa (HVAC) hoạt động trên VHAL thông qua "
        "CarPropertyManager (AreaId: 0).\n"
        "2. Phanh khẩn cấp tự động (AEB) hoạt động kết hợp với ADAS sẽ kích "
        "hoạt để bảo vệ an toàn khi xe chạy > 80km/h và khoảng cách va chạm "
        "dưới 15m."
    )

    logger.info("Formulated RAG response with citations.")
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "status": "success",
    }


if __name__ == "__main__":
    # Quick sanity check when running the file directly.
    sample_query = "Làm thế nào kích hoạt phanh khẩn cấp ADAS?"
    print(solve_automotive_query(sample_query))
