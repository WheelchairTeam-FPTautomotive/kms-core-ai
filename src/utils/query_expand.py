"""Retrieval query expansion for tone-free Vietnamese / cross-lingual drift."""

from __future__ import annotations

import re
import unicodedata

# Demo-only optional enrichment. Pattern expansions still work without a hit.
_DOMAIN_ALIASES: dict[str, list[str]] = {
    "hvac": [
        "heating ventilation air conditioning",
        "điều hòa",
    ],
    "aeb": [
        "automatic emergency braking",
        "phanh khẩn cấp tự động",
    ],
    "adas": [
        "advanced driver assistance",
        "hỗ trợ lái xe",
    ],
}

_MAX_VARIANTS = 5

# Matched against fold_vi(query)
_DEFINITIONAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<topic>.+?)\s+(?:la gi|nghia la gi)\s*\??$"),
    re.compile(r"^(?:what is|what's|whats)\s+(?P<topic>.+?)\s*\??$"),
    re.compile(r"^(?:meaning of)\s+(?P<topic>.+?)\s*\??$"),
    re.compile(r"^(?P<topic>.+?)\s+meaning\s*\??$"),
    re.compile(
        r"^(?P<topic>.+?)\s+(?:hoat dong the nao|hoat dong nhu the nao)\s*\??$"
    ),
    re.compile(r"^(?:how does)\s+(?P<topic>.+?)\s+work\s*\??$"),
    re.compile(r"^(?P<topic>.+?)\s+(?:duoc dieu khien|controlled by)\b.*$"),
)


def fold_vi(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics. Explicitly map đ/Đ → d."""
    # --- START MODIFICATION ---
    lowered = text.lower().strip()
    lowered = lowered.replace("đ", "d").replace("Đ", "d")
    nfd = unicodedata.normalize("NFD", lowered)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # --- END MODIFICATION ---


def _extract_topic(folded: str) -> str | None:
    for pattern in _DEFINITIONAL_PATTERNS:
        match = pattern.match(folded)
        if match:
            topic = (match.group("topic") or "").strip(" ?.,!;:")
            # Strip leading articles / fillers
            topic = re.sub(
                r"^(he thong|he|the|a|an|he thong kiem soat)\s+",
                "",
                topic,
            ).strip()
            if topic:
                return topic
    return None


def expand_retrieval_queries(query: str) -> list[str]:
    """
    Build up to 5 retrieval variants. Original query is always first.
    Expansions are for vector lookup only — answer generation keeps the original.
    """
    # --- START MODIFICATION ---
    original = (query or "").strip()
    if not original:
        return []

    variants: list[str] = [original]
    seen = {original.casefold()}

    def _add(candidate: str) -> None:
        text = candidate.strip()
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        if len(variants) >= _MAX_VARIANTS:
            return
        seen.add(key)
        variants.append(text)

    folded = fold_vi(original)
    topic = _extract_topic(folded)

    if topic:
        _add(f"What is {topic}")
        _add(f"{topic} meaning")
        _add(f"how does {topic} work")
        _add(f"định nghĩa {topic}")
        for alias in _DOMAIN_ALIASES.get(topic, []):
            _add(alias)
            _add(f"What is {alias}")

    return variants
    # --- END MODIFICATION ---
