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
    "tucson": "tucson",
    "ranger raptor": "ranger raptor",
    "rangerraptor": "ranger raptor",
    # Bronco Raptor trim → parent Bronco (keep ranger raptor distinct)
    "bronco raptor": "bronco",
    "broncoraptor": "bronco",
    "raptor": "bronco",
    "bronco": "bronco",
}

_MODEL_DEFAULT_MAKE: dict[str, str] = {
    "santa fe": "hyundai",
    "accent": "hyundai",
    "tucson": "hyundai",
    "sonata": "hyundai",
    "ioniq 5": "hyundai",
    "camry": "toyota",
    "ranger raptor": "ford",
    "bronco": "ford",
}

# Extra match tokens for trim/title matching (beyond canonical model)
_MODEL_MATCH_EXTRAS: dict[str, tuple[str, ...]] = {
    "bronco": ("raptor",),
    "ranger raptor": ("raptor", "ranger"),
}

# Warm-time corpus model→make (from BM25 sidecar metas). Static table wins on conflict.
_CORPUS_MODEL_MAKE: dict[str, str] = {}

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
    "unknown",
    "query",
    "he",
    "thong",
    "lam",
    "sao",
    "huong",
    "dan",
    "quy",
    "trinh",
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
    # Prefer longer alias keys first (bronco raptor before raptor)
    spaced = fold_vi(raw).replace("_", " ").replace("-", " ")
    spaced = re.sub(r"\s+", " ", spaced).strip()
    for key in sorted(_MODEL_ALIASES.keys(), key=len, reverse=True):
        key_c = _fold_compact(key)
        if folded == key or compact == key_c or spaced == key:
            return _MODEL_ALIASES[key]
        if key_c and key_c in compact and len(key_c) >= 6:
            return _MODEL_ALIASES[key]
    if folded in _MODEL_ALIASES:
        return _MODEL_ALIASES[folded]
    if compact in _MODEL_ALIASES:
        return _MODEL_ALIASES[compact]
    if spaced in _MODEL_ALIASES:
        return _MODEL_ALIASES[spaced]
    if _is_hash_token(_fold_compact(spaced)):
        return ""
    return spaced


def dedupe_meta_value(value: str | None) -> str:
    """Collapse duplicated tokens: 'bronco bronco' → 'bronco'."""
    # --- START MODIFICATION ---
    parts = [p for p in _norm(value).split() if p]
    if not parts:
        return ""
    out: list[str] = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return " ".join(out)
    # --- END MODIFICATION ---


def model_match_pins(model: str | None) -> set[str]:
    """
    Compact strings that count as a vehicle match for BM25/cite filters.
    Includes canonical model + trim extras (e.g. bronco + raptor).
    """
    # --- START MODIFICATION ---
    raw = _norm(model)
    if not raw:
        return set()
    canon = canonical_model(raw) or raw
    pins = {_fold_compact(raw), _fold_compact(canon)}
    for extra in _MODEL_MATCH_EXTRAS.get(canon, ()):
        pins.add(_fold_compact(extra))
    # Bare "raptor" (non-ranger) also pins bronco
    if "raptor" in pins and "ranger" not in _fold_compact(raw):
        pins.add("bronco")
    return {p for p in pins if p and len(p) >= 3}
    # --- END MODIFICATION ---


def set_corpus_model_make_map(mapping: dict[str, str] | None) -> None:
    """Replace warm-time corpus model→make map (empty clears)."""
    # --- START MODIFICATION ---
    global _CORPUS_MODEL_MAKE
    _CORPUS_MODEL_MAKE = {
        dedupe_meta_value(canonical_model(k) or k): _norm(v)
        for k, v in (mapping or {}).items()
        if (k or "").strip() and (v or "").strip()
    }
    # --- END MODIFICATION ---


def get_corpus_model_make_map() -> dict[str, str]:
    return dict(_CORPUS_MODEL_MAKE)


def build_model_make_from_bm25_metas(metas: list[dict] | None) -> dict[str, str]:
    """
    Majority make per canonical model from BM25 sidecar metas (in-memory, no Chroma walk).
    """
    # --- START MODIFICATION ---
    from collections import Counter

    votes: dict[str, Counter[str]] = {}
    for meta in metas or []:
        raw_model = str((meta or {}).get("model") or "").strip()
        raw_make = _norm((meta or {}).get("make"))
        if not raw_model or not raw_make:
            continue
        if raw_make not in _KNOWN_MAKES:
            continue
        model = dedupe_meta_value(canonical_model(raw_model) or raw_model)
        if not model or len(_fold_compact(model)) < 3:
            continue
        votes.setdefault(model, Counter())[raw_make] += 1
    return {m: c.most_common(1)[0][0] for m, c in votes.items() if c}
    # --- END MODIFICATION ---


def normalize_query_vehicle(
    make: str | None, model: str | None
) -> tuple[str, str]:
    """Normalize planner make/model for Chroma where (canonical model)."""
    # --- START MODIFICATION ---
    # Remap known model mistaken as make (e.g. make=tucson → hyundai/tucson)
    mo = canonical_model(model or "") if (model or "").strip() else ""
    mo = dedupe_meta_value(mo)
    mk = _norm(make)
    if not mo and mk:
        as_model = dedupe_meta_value(canonical_model(mk) or "")
        if as_model and (
            as_model in _MODEL_DEFAULT_MAKE or as_model in _CORPUS_MODEL_MAKE
        ):
            if mk not in _KNOWN_MAKES or _fold_compact(mk) == _fold_compact(as_model):
                mo = as_model
                mk = _MODEL_DEFAULT_MAKE.get(mo) or _CORPUS_MODEL_MAKE.get(mo, "")
    if mo and not mk:
        # Static overrides win over corpus for known models
        mk = _MODEL_DEFAULT_MAKE.get(mo) or _CORPUS_MODEL_MAKE.get(mo, "")
    return mk, mo
    # --- END MODIFICATION ---


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

    # --- START MODIFICATION ---
    make = dedupe_meta_value(make)
    model = dedupe_meta_value(canonical_model(model) if model else "")
    if model and not make:
        make = _MODEL_DEFAULT_MAKE.get(model, "")
    # --- END MODIFICATION ---

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
