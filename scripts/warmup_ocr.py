"""Pre-download RapidOCR ONNX weights for offline / CI ingest."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipelines.pdf_extract import warmup_ocr

if __name__ == "__main__":
    warmup_ocr()
    print("OK: RapidOCR warm-up complete")
