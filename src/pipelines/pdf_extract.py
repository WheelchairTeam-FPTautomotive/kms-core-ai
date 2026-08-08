"""Hybrid PDF page extract: PyMuPDF first, RapidOCR on thin/junk pages."""

from __future__ import annotations

from functools import lru_cache

from utils.logger import setup_logger

logger = setup_logger("pdf_extract")

# --- START MODIFICATION ---
# Ingest-time only: keep query RAG path free of OCR latency.
_JUNK_PREFIXES: tuple[str, ...] = (
    "tier 1 – vds icons",
    "tier 1 - vds icons",
    "tier 1 – vds",
    "tier 1 - vds",
)

_MIN_ALPHA_CHARS = 80


def is_thin_or_junk_text(text: str) -> bool:
    """True when page text is too weak for RAG and should escalate to OCR."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    alpha = sum(1 for c in cleaned if c.isalpha())
    if alpha < _MIN_ALPHA_CHARS:
        return True
    head = cleaned[:120].lower()
    return any(head.startswith(p) or p in head[:80] for p in _JUNK_PREFIXES)


@lru_cache(maxsize=1)
def _get_rapidocr():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def warmup_ocr() -> None:
    """Download/cache RapidOCR ONNX weights before batch ingest."""
    try:
        _get_rapidocr()
        logger.info("RapidOCR weights ready")
    except Exception as exc:
        logger.warning(f"RapidOCR warm-up skipped/failed: {exc}")


def extract_text_pymupdf(filepath: str, page_index0: int) -> str:
    import pymupdf

    with pymupdf.open(filepath) as doc:
        if page_index0 < 0 or page_index0 >= doc.page_count:
            return ""
        page = doc.load_page(page_index0)
        return (page.get_text("text") or "").strip()


def ocr_page_pymupdf(filepath: str, page_index0: int, dpi: int = 200) -> str:
    import pymupdf

    ocr = _get_rapidocr()
    with pymupdf.open(filepath) as doc:
        if page_index0 < 0 or page_index0 >= doc.page_count:
            return ""
        page = doc.load_page(page_index0)
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
    result, _ = ocr(png_bytes)
    if not result:
        return ""
    # RapidOCR returns list of [box, text, score]
    lines: list[str] = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lines.append(str(item[1]))
    return "\n".join(lines).strip()


def extract_page_text(
    filepath: str,
    page_number: int,
    *,
    ocr_on_thin: bool = True,
    ocr_dpi: int = 200,
) -> tuple[str, str]:
    """
    Extract one 1-based page.

    Returns (text, method) where method is 'pymupdf' | 'ocr' | 'empty'.
    """
    page_index0 = page_number - 1
    text = extract_text_pymupdf(filepath, page_index0)
    if text and not is_thin_or_junk_text(text):
        return text, "pymupdf"

    if not ocr_on_thin:
        return text, "pymupdf" if text else "empty"

    try:
        ocr_text = ocr_page_pymupdf(filepath, page_index0, dpi=ocr_dpi)
    except Exception as exc:
        logger.warning(
            f"OCR failed for {filepath} page={page_number}: {exc}; "
            f"keeping pymupdf text"
        )
        return text, "pymupdf" if text else "empty"

    if ocr_text and not is_thin_or_junk_text(ocr_text):
        return ocr_text, "ocr"
    if ocr_text and len(ocr_text) > len(text or ""):
        return ocr_text, "ocr"
    return text, "pymupdf" if text else "empty"


def pdf_page_count(filepath: str) -> int:
    import pymupdf

    with pymupdf.open(filepath) as doc:
        return int(doc.page_count)


# --- END MODIFICATION ---
