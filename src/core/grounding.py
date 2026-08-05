"""Detect ungrounded RAG answers so citations are not returned with soft denies."""

from __future__ import annotations

import re

# --- START MODIFICATION ---
# Citation honesty: GROUNDED marker + narrow insufficient-context phrases only.
_GROUNDED_HEADER_RE = re.compile(
    r"^GROUNDED:\s*(yes|no)\s*[\r\n]+",
    re.IGNORECASE,
)
_GROUNDED_ONLY_RE = re.compile(
    r"^GROUNDED:\s*(yes|no)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_UNGROUNDED_EN = re.compile(
    r"("
    r"not\s+mentioned\s+in|"
    r"does\s+not\s+contain\s+information|"
    r"no\s+mention\s+of|"
    r"not\s+found\s+in\s+the\s+(technical\s+)?(documents?|manual)|"
    r"no\s+matching\s+information\s+was\s+found|"
    r"was\s+not\s+found\s+in\s+the\s+(technical\s+)?(documents?|manual)|"
    r"insufficient\s+(document\s+)?context|"
    r"context\s+(is\s+)?insufficient|"
    r"not\s+covered\s+in\s+the\s+(manual|documents?)"
    r")",
    re.IGNORECASE,
)

_UNGROUNDED_VI = re.compile(
    r"("
    r"không\s+có\s+thông\s+tin|"
    r"không\s+được\s+đề\s+cập|"
    r"không\s+đề\s+cập\s+đến|"
    r"không\s+tìm\s+thấy\s+trong\s+tài\s+liệu|"
    r"không\s+tìm\s+thấy\s+thông\s+tin\s+phù\s+hợp|"
    r"ngữ\s+cảnh\s+không\s+đủ|"
    r"không\s+có\s+trong\s+(tài\s+liệu|manual)"
    r")",
    re.IGNORECASE,
)
# --- END MODIFICATION ---


def parse_grounded_answer(raw: str) -> tuple[str, bool | None]:
    """
    Strip optional GROUNDED: yes|no header for TTS safety.

    Returns (cleaned_body, flag) where flag is True/False if marker present, else None.
    """
    text = (raw or "").strip()
    if not text:
        return "", None

    match = _GROUNDED_HEADER_RE.match(text)
    if match:
        flag = match.group(1).lower() == "yes"
        body = text[match.end() :].strip()
        return body, flag

    only = _GROUNDED_ONLY_RE.match(text)
    if only:
        flag = only.group(1).lower() == "yes"
        return "", flag

    return text, None


def looks_ungrounded(answer: str) -> bool:
    """Phrase detector: insufficient-context wording only (not bare not/không)."""
    text = (answer or "").strip()
    if not text:
        return False
    return bool(_UNGROUNDED_EN.search(text) or _UNGROUNDED_VI.search(text))


def is_ungrounded(answer: str, grounded_flag: bool | None) -> bool:
    """
    Ungrounded when the spoken body uses insufficient-context phrasing.

    Small local models over-emit GROUNDED: no on valid retrievals; never flip
    on the marker alone. Empty body + GROUNDED: no still counts as ungrounded.
    Explicit GROUNDED: yes does not override a soft-deny body (honesty > marker).
    """
    text = (answer or "").strip()
    if looks_ungrounded(text):
        return True
    if grounded_flag is False and not text:
        return True
    return False
