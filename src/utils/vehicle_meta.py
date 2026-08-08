"""Parse make / model / year from PDF paths and filenames for Chroma metadata."""

from __future__ import annotations

import re
from pathlib import Path

from utils.query_expand import fold_vi

# --- START MODIFICATION ---
_KNOWN_MAKES = (
    "hyundai",
    "toyota",
    "ford",
    "kia",
    "honda",
    "mazda",
    "nissan",
    "chevrolet",
    "vinfast",
    "bmw",
    "mercedes",
    "audi",
    "volkswagen",
    "subaru",
    "lexus",
)

# Multi-word models checked before single tokens (folded, no spaces in key variants)
_MODEL_ALIASES: dict[str, str] = {
    "santafe": "santa fe",
    "santa fe": "santa fe",
    "ioniq5": "ioniq 5",
    "ioniq 5": "ioniq 5",
    "ranger raptor": "ranger raptor",
    "rangerraptor": "ranger raptor",
}

_MODEL_DEFAULT_MAKE: dict[str, str] = {
    "santa fe": "hyundai",
    "accent": "hyundai",
    "tucson": "hyundai",
    "sonata": "hyundai",
    "ioniq 5": "hyundai",
    "camry": "toyota",
    "ranger raptor": "ford",
}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_HEX_ID_RE = re.compile(r"^[a-f0-9]{32,}$")

_NOISE = {
    "owners",
    "owner",
    "manual",
    "quick",
    "reference",
    "guide",
    "qrg",
    "gsg",
    "getting",
    "started",
    "display",
    "audio",
    "user",
    "users",
    "pdf",
    "hyundai",
    "toyota",
    "ford",
    "kia",
    "final",
    "web",
    "en",
    "us",
    "my",
    "om",
    "system",
    "trading",
    "central",
}


def _norm(value: str | None) -> str:
    """Chroma-safe primitive: never None."""
    return (value or "").strip().lower()


def _fold_compact(text: str) -> str:
    return _NON_ALNUM.sub("", fold_vi(text))


def _is_hash_token(token: str) -> bool:
    t = (token or "").strip().lower()
    return bool(_HEX_ID_RE.match(t))


def canonical_model(raw: str) -> str:
    folded = fold_vi(raw).strip()
    if _is_hash_token(_fold_compact(folded)):
        return ""
    compact = _fold_compact(raw)
    if folded in _MODEL_ALIASES:
        return _MODEL_ALIASES[folded]
    if compact in _MODEL_ALIASES:
        return _MODEL_ALIASES[compact]
    # "santa_fe" / "Santa-Fe"
    spaced = fold_vi(raw).replace("_", " ").replace("-", " ")
    spaced = re.sub(r"\s+", " ", spaced).strip()
    if spaced in _MODEL_ALIASES:
        return _MODEL_ALIASES[spaced]
    if _is_hash_token(_fold_compact(spaced)):
        return ""
    return spaced


KNOWN_MAKES = _KNOWN_MAKES
MODEL_ALIASES = _MODEL_ALIASES


def _display_stem(path: Path, doc_name: str) -> str:
    """Prefer human title (mapping.json) over corpus SHA-256 stems."""
    stem = path.stem
    if _is_hash_token(stem):
        title = Path(doc_name).stem if doc_name else ""
        return title or ""
    if doc_name:
        return f"{stem} {Path(doc_name).stem}"
    return stem


def parse_vehicle_metadata(filepath: Path | str, doc_name: str = "") -> dict[str, str]:
    """
    Derive make/model/year from path segments and filename.

    Prefer directory layout: .../<Make>/<Model>/<Year>/<file>.pdf
    Fallback: tokens inside filename / doc_name (never SHA hash stems).
    Missing fields → "" (Chroma rejects None).
    """
    path = Path(filepath)
    parts = [p for p in path.parts if p not in (path.anchor, "/", "\\")]
    dir_parts = [fold_vi(p) for p in parts[:-1]]
    display = _display_stem(path, doc_name)
    name_blob = fold_vi(display)

    make = ""
    model = ""
    year = ""

    # Year from any path segment or display name
    for part in dir_parts + [name_blob]:
        if not part:
            continue
        m = _YEAR_RE.search(part)
        if m:
            year = m.group(0)
            break

    # Make from path segment
    for part in dir_parts:
        compact = _fold_compact(part)
        if _is_hash_token(compact):
            continue
        for known in _KNOWN_MAKES:
            if compact == known or compact.startswith(known):
                make = known
                break
        if make:
            break
    if not make and name_blob:
        for known in _KNOWN_MAKES:
            if known in _fold_compact(name_blob):
                make = known
                break

    # Model: segment after make in path
    if make:
        try:
            idx = next(
                i
                for i, p in enumerate(dir_parts)
                if _fold_compact(p) == make or _fold_compact(p).startswith(make)
            )
            for part in dir_parts[idx + 1 :]:
                if _YEAR_RE.fullmatch(part.strip()):
                    continue
                if part in {"docs_pdf", "docs_corpus", "data", "pdf"}:
                    continue
                if _is_hash_token(_fold_compact(part)):
                    continue
                cand = canonical_model(part)
                if cand:
                    model = cand
                    break
        except StopIteration:
            pass

    if not model and name_blob:
        for alias_key, canonical in sorted(
            _MODEL_ALIASES.items(), key=lambda kv: -len(kv[0])
        ):
            if alias_key.replace(" ", "") in _fold_compact(name_blob) or alias_key in name_blob:
                model = canonical
                break

    if not model and name_blob:
        tokens = [
            t
            for t in re.split(r"[^a-z0-9]+", name_blob)
            if t
            and t not in _NOISE
            and not _YEAR_RE.fullmatch(t)
            and not _is_hash_token(t)
            and len(t) >= 3
        ]
        if make:
            tokens = [t for t in tokens if t != make]
        if tokens:
            model = canonical_model(" ".join(tokens[:2]))

    if model and not make:
        make = _MODEL_DEFAULT_MAKE.get(model, "")

    # Final hash guard
    if _is_hash_token(_fold_compact(model)):
        model = ""

    return {
        "make": _norm(make),
        "model": _norm(model),
        "year": str(year or "").strip(),
    }


def build_chroma_where(
    make: str | None = None,
    model: str | None = None,
    year: str | None = None,
) -> dict | None:
    """Build Chroma where clause from non-empty vehicle fields."""
    clauses: list[dict] = []
    m = _norm(make)
    mo = _norm(model)
    y = str(year or "").strip()
    if m:
        clauses.append({"make": m})
    if mo:
        clauses.append({"model": mo})
    if y:
        clauses.append({"year": y})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# --- END MODIFICATION ---
