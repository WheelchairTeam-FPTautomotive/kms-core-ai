# Golden S2 Score Report

- Generated (UTC): `2026-08-05T16:52:01Z`
- Git SHA: `bc582c0`
- OPENAI_MODEL: `llama3.2:3b`
- LLM_PROVIDER: `ollama`
- OPENAI_BASE_URL: `http://localhost:11434/v1`
- VECTOR_DB_TYPE: `chroma`
- RAG_MAX_DISTANCE: `1.15`
- Chroma rebuild: run `uv run python src/pipelines/ingest.py --target chroma --reset` before scoring
- Note: OEM manuals live under repo `data/docs_pdf`; set `DOCS_PDF_DIR=data/docs_pdf` if `.env` points at HACKATHON-only paths

## Result: **42/43** (97.7%) — threshold 90%

Overall: `PASS`

## Failures

- `car-rag-01`: status_mismatch expected=success actual=not_found
