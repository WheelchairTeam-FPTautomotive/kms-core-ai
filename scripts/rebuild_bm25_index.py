#!/usr/bin/env python3
"""Rebuild BM25 sidecar from the live Chroma collection (no PDF re-ingest)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pipelines.solve_problem import get_chroma_collection
from utils.bm25_index import rebuild_from_chroma


def main() -> int:
    collection = get_chroma_collection()
    path = rebuild_from_chroma(collection)
    print(f"BM25 sidecar rebuilt: {path} (chroma_count={collection.count()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
