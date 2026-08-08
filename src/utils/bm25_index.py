"""BM25 sidecar index for hybrid RAG (lexical exactness for OEM tokens)."""

from __future__ import annotations

import json
import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import settings
from utils.logger import setup_logger
from utils.query_expand import fold_vi

logger = setup_logger("bm25_index")

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_LOCK = threading.RLock()
_CACHE: "Bm25Sidecar | None" = None


def bm25_index_dir() -> Path:
    root = Path(getattr(settings, "bm25_index_path", "data/bm25_index") or "data/bm25_index")
    return root


def tokenize(text: str) -> list[str]:
    folded = fold_vi(text or "")
    return _TOKEN_RE.findall(folded)


@dataclass
class Bm25Hit:
    chunk_id: str
    text: str
    document_name: str
    section: str
    page: int
    make: str
    model: str
    year: str
    rank: int
    score: float


class Bm25Sidecar:
    """In-memory BM25 over Chroma chunk texts + vehicle metadata."""

    def __init__(
        self,
        *,
        chunk_ids: list[str],
        texts: list[str],
        metas: list[dict[str, Any]],
        tokenized: list[list[str]],
        bm25: Any,
    ) -> None:
        self.chunk_ids = chunk_ids
        self.texts = texts
        self.metas = metas
        self.tokenized = tokenized
        self.bm25 = bm25

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        make: str = "",
        model: str = "",
        year: str = "",
    ) -> list[Bm25Hit]:
        if not query or not self.chunk_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        make_n = (make or "").strip().lower()
        model_n = (model or "").strip().lower()
        year_n = (year or "").strip().lower()
        model_compact = "".join(ch for ch in model_n if ch.isalnum())

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        hits: list[Bm25Hit] = []
        filtered_rank = 0
        for _global_rank, (idx, score) in enumerate(indexed, start=1):
            if score <= 0:
                break
            meta = self.metas[idx] or {}
            if make_n and str(meta.get("make") or "").strip().lower() != make_n:
                # allow make miss when model substring already pins the vehicle
                if not model_compact:
                    continue
            if model_compact:
                from utils.vehicle_meta import model_match_pins

                pins = model_match_pins(model_n) or {model_compact}
                meta_model = str(meta.get("model") or "").strip().lower()
                meta_compact = "".join(ch for ch in meta_model if ch.isalnum())
                name_compact = "".join(
                    ch
                    for ch in str(meta.get("document_name") or "").lower()
                    if ch.isalnum()
                )
                trim_compact = "".join(
                    ch
                    for ch in str(meta.get("trim") or "").lower()
                    if ch.isalnum()
                )
                hay = meta_compact + name_compact + trim_compact
                if not any(
                    p in hay or p in meta_compact or meta_compact in p
                    for p in pins
                    if len(p) >= 3
                ):
                    continue
            if year_n and str(meta.get("year") or "").strip().lower() != year_n:
                continue
            filtered_rank += 1
            hits.append(
                Bm25Hit(
                    chunk_id=self.chunk_ids[idx],
                    text=self.texts[idx],
                    document_name=str(meta.get("document_name") or "Automotive Manual"),
                    section=str(meta.get("section") or f"Trang {meta.get('page', 1)}"),
                    page=int(meta.get("page") or 0),
                    make=str(meta.get("make") or ""),
                    model=str(meta.get("model") or ""),
                    year=str(meta.get("year") or ""),
                    rank=filtered_rank,
                    score=float(score),
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def build_sidecar_from_rows(
    rows: list[dict[str, Any]],
) -> Bm25Sidecar:
    from rank_bm25 import BM25Okapi

    chunk_ids: list[str] = []
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    tokenized: list[list[str]] = []
    for row in rows:
        cid = str(row.get("id") or "")
        text = str(row.get("text") or "")
        if not cid or not text.strip():
            continue
        meta = {
            "document_name": row.get("document_name") or "",
            "section": row.get("section") or "",
            "page": int(row.get("page") or 0),
            "make": row.get("make") or "",
            "model": row.get("model") or "",
            "year": row.get("year") or "",
            "document_id": row.get("document_id") or "",
            "trim": row.get("trim") or "",
        }
        # --- START MODIFICATION ---
        try:
            from utils.vehicle_meta import dedupe_meta_value

            meta["make"] = dedupe_meta_value(str(meta["make"]))
            meta["model"] = dedupe_meta_value(str(meta["model"]))
        except Exception:  # noqa: BLE001
            pass
        # --- END MODIFICATION ---
        toks = tokenize(text)
        if not toks:
            continue
        chunk_ids.append(cid)
        texts.append(text)
        metas.append(meta)
        tokenized.append(toks)
    bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([["empty"]])
    return Bm25Sidecar(
        chunk_ids=chunk_ids,
        texts=texts,
        metas=metas,
        tokenized=tokenized,
        bm25=bm25,
    )


def save_sidecar(sidecar: Bm25Sidecar, directory: Path | None = None) -> Path:
    out = directory or bm25_index_dir()
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_ids": sidecar.chunk_ids,
        "texts": sidecar.texts,
        "metas": sidecar.metas,
        "tokenized": sidecar.tokenized,
    }
    bin_path = out / "bm25.pkl"
    meta_path = out / "manifest.json"
    with bin_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(
        json.dumps({"n_chunks": len(sidecar), "path": str(bin_path)}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote BM25 sidecar n_chunks=%s path=%s", len(sidecar), bin_path)
    return bin_path


def load_sidecar(directory: Path | None = None) -> Bm25Sidecar | None:
    from rank_bm25 import BM25Okapi

    out = directory or bm25_index_dir()
    bin_path = out / "bm25.pkl"
    if not bin_path.exists():
        return None
    with bin_path.open("rb") as f:
        payload = pickle.load(f)
    tokenized = payload.get("tokenized") or []
    bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([["empty"]])
    return Bm25Sidecar(
        chunk_ids=list(payload.get("chunk_ids") or []),
        texts=list(payload.get("texts") or []),
        metas=list(payload.get("metas") or []),
        tokenized=tokenized,
        bm25=bm25,
    )


def get_bm25_sidecar(*, reload: bool = False) -> Bm25Sidecar | None:
    global _CACHE
    with _LOCK:
        if _CACHE is not None and not reload:
            return _CACHE
        _CACHE = load_sidecar()
        if _CACHE is not None:
            logger.info("BM25 sidecar loaded n_chunks=%s", len(_CACHE))
        else:
            logger.warning("BM25 sidecar missing at %s", bm25_index_dir())
        return _CACHE


def append_chunks_to_sidecar(chunk_rows: list[dict[str, Any]]) -> None:
    """
    Best-effort incremental update: reload full index rebuild recommended.
    For ingest, callers should prefer rebuild_from_chroma after large batches.
    """
    if not chunk_rows:
        return
    existing = load_sidecar()
    rows: list[dict[str, Any]] = []
    if existing is not None:
        for i, cid in enumerate(existing.chunk_ids):
            meta = existing.metas[i]
            rows.append(
                {
                    "id": cid,
                    "text": existing.texts[i],
                    **meta,
                }
            )
    by_id = {r["id"]: r for r in rows}
    for row in chunk_rows:
        by_id[str(row["id"])] = row
    sidecar = build_sidecar_from_rows(list(by_id.values()))
    save_sidecar(sidecar)
    get_bm25_sidecar(reload=True)


def export_rows_from_chroma(collection: Any, *, batch_size: int = 500) -> list[dict[str, Any]]:
    """Pull all Chroma documents into BM25 row dicts."""
    rows: list[dict[str, Any]] = []
    offset = 0
    total = collection.count()
    while offset < total:
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        if not ids:
            break
        for cid, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            rows.append(
                {
                    "id": cid,
                    "text": doc or "",
                    "document_name": meta.get("document_name") or "",
                    "section": meta.get("section") or "",
                    "page": int(meta.get("page") or 0),
                    "make": meta.get("make") or "",
                    "model": meta.get("model") or "",
                    "year": meta.get("year") or "",
                    "document_id": meta.get("document_id") or "",
                }
            )
        offset += len(ids)
        logger.info("BM25 export progress %s/%s", min(offset, total), total)
    return rows


def rebuild_from_chroma(collection: Any) -> Path:
    rows = export_rows_from_chroma(collection)
    sidecar = build_sidecar_from_rows(rows)
    path = save_sidecar(sidecar)
    get_bm25_sidecar(reload=True)
    return path
