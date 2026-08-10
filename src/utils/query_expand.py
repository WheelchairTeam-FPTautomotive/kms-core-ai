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

# --- START MODIFICATION ---
# OEM / procedure synonyms for hybrid BM25 (ISOFIX, MIST, VI defrost, seat memory, EPB, TPMS)
_OEM_SYNONYMS: tuple[tuple[str, list[str]], ...] = (
    ("isofix", ["latch", "child restraint anchor", "lower anchors"]),
    ("latch", ["isofix", "child restraint"]),
    ("mist", ["single wipe", "wiper mist", "single wiping cycle"]),
    ("say kinh sau", ["rear window defroster", "rear defroster", "rear defrost"]),
    ("bat say kinh", ["rear window defroster", "defrost", "rear defrost"]),
    ("say kinh", ["rear window defroster", "rear defroster", "defrost"]),
    ("defroster", ["rear window defroster", "defrost"]),
    ("nho ghe", ["driver position memory", "power seat memory", "seat memory"]),
    ("washer fluid", ["washer reservoir", "windshield washer"]),
    ("nuoc rua kinh", ["washer fluid", "washer reservoir"]),
    ("wheel nut", ["lug nut torque", "wheel lug nut", "nut torque"]),
    ("moc siet", ["wheel nut torque", "lug nut torque", "wheel lug nut"]),
    ("torque", ["wheel nut torque", "lug nut"]),
    ("phanh do dien tu", ["electronic parking brake", "EPB switch", "parking brake switch"]),
    ("phanh dien tu", ["electronic parking brake", "EPB switch", "parking brake"]),
    ("phanh tay dien tu", ["electronic parking brake", "EPB switch", "parking brake switch"]),
    ("phanh tay", ["electronic parking brake", "EPB switch", "parking brake"]),
    ("epb", ["electronic parking brake", "EPB switch", "parking brake switch"]),
    ("ap suat lop", ["tire pressure", "TPMS", "tire pressure monitoring"]),
    ("ap suat", ["tire pressure", "TPMS"]),
    ("tpms", ["tire pressure monitoring system", "tire pressure"]),
)
# --- END MODIFICATION ---

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


def matched_oem_token_set(query: str) -> set[str]:
    """
    Folded OEM operational tokens for the longest matching synonym family.
    Empty when the query does not hit any _OEM_SYNONYMS needle.
    """
    # --- START MODIFICATION ---
    folded = fold_vi(query or "")
    if not folded:
        return set()
    for needle, aliases in sorted(_OEM_SYNONYMS, key=lambda kv: -len(kv[0])):
        if needle not in folded:
            continue
        tokens: set[str] = {fold_vi(needle)}
        for alias in aliases:
            fa = fold_vi(alias)
            if fa:
                tokens.add(fa)
        return {t for t in tokens if len(t) >= 3}
    return set()
    # --- END MODIFICATION ---


def cite_text_hits_oem(text: str, tokens: set[str] | None = None, *, query: str = "") -> bool:
    """True when matched_text contains any token from the OEM family."""
    # --- START MODIFICATION ---
    toks = tokens if tokens is not None else matched_oem_token_set(query)
    if not toks:
        return False
    folded = fold_vi(text or "")
    return any(tok in folded for tok in toks)
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


def oem_english_search_phrase(query: str) -> str | None:
    """Primary English OEM manual phrase when query hits a synonym family."""
    # --- START MODIFICATION ---
    folded = fold_vi(query or "")
    if not folded:
        return None
    for needle, aliases in sorted(_OEM_SYNONYMS, key=lambda kv: -len(kv[0])):
        if needle in folded and aliases:
            return aliases[0].strip()
    return None
    # --- END MODIFICATION ---


def expand_retrieval_queries(query: str) -> list[str]:
    """
    Build up to 5 retrieval variants.
    Matched OEM EN phrases are prepended so BM25 fan-in sees them first (RC3).
    Answer generation keeps the original utterance.
    """
    # --- START MODIFICATION ---
    original = (query or "").strip()
    if not original:
        return []

    folded = fold_vi(original)
    oem_first: list[str] = []
    for needle, aliases in sorted(_OEM_SYNONYMS, key=lambda kv: -len(kv[0])):
        if needle in folded:
            for alias in aliases:
                if alias.strip() and alias.casefold() not in {
                    x.casefold() for x in oem_first
                }:
                    oem_first.append(alias.strip())
            break  # one OEM family per query

    variants: list[str] = []
    seen: set[str] = set()

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

    for phrase in oem_first:
        _add(phrase)
    _add(original)

    topic = _extract_topic(folded)

    if topic:
        _add(f"What is {topic}")
        _add(f"{topic} meaning")
        _add(f"how does {topic} work")
        _add(f"định nghĩa {topic}")
        for alias in _DOMAIN_ALIASES.get(topic, []):
            _add(alias)
            _add(f"What is {alias}")

    for needle, aliases in _OEM_SYNONYMS:
        if needle in folded:
            for alias in aliases:
                _add(alias)

    return variants
    # --- END MODIFICATION ---