# Cold vs warm Core AI latency (#14)

**Date:** 2026-08-07  
**Host:** Windows local (Docker Desktop + Ollama)  
**Endpoint:** `POST http://127.0.0.1:8001/api/v1/search` (Core AI alone; gateway E2E out of scope)  
**Fixed query:** `What is the HVAC system?` (corpus-grounded `status=success`; same input for stacks)  
**Warm protocol:** 1 discard + N=10 consecutive samples (no sleep — Ollama default ~5m `keep_alive` unload cannot fire mid-loop)

## Sandbox / non-root

Image `kms-core-ai:local` (Dockerfile HEALTHCHECK → `/api/v1/health`).

```text
docker run --rm --entrypoint /bin/bash kms-core-ai:local \
  -c 'gosu appuser whoami; gosu appuser id'
# appuser
# uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)
```

Entrypoint [`scripts/entrypoint.sh`](../scripts/entrypoint.sh) starts as root briefly for bind-mount `chown`, then `exec gosu appuser` before Uvicorn.

## Scorecard alignment (Sprint 2 freeze)

| Stack | Config | Target warm p95 | Measured warm p95 | Result |
|-------|--------|-----------------|-------------------|--------|
| Extractive | `LLM_PROVIDER=none` | < 2.0 s | **2.7 ms** | PASS |
| Freeze LLM | `LLM_PROVIDER=ollama` `OPENAI_MODEL=qwen2.5:7b-instruct` | < 5.0 s | **946.1 ms** | PASS |
| Prior local | `llama3.2:3b` (pre-pull) | < 5.0 s | **1894.5 ms** | PASS (reference) |

Primary freeze scorecard stack: **Qwen2.5-7B-Instruct** (pulled via `ollama pull qwen2.5:7b-instruct`).

### Extractive (`LLM_PROVIDER=none`)

| Phase | n | p50_ms | p95_ms | max_ms | avg_ms | status |
|-------|---|--------|--------|--------|--------|--------|
| cold (1st) | 1 | 1591.9 | 1591.9 | 1591.9 | 1591.9 | success |
| warm | 10 | 2.3 | 2.7 | 2.8 | 2.3 | success |

### Ollama freeze model (`qwen2.5:7b-instruct`, `language=en`)

| Phase | n | p50_ms | p95_ms | max_ms | avg_ms | status |
|-------|---|--------|--------|--------|--------|--------|
| cold (1st) | 1 | 1159.5 | 1159.5 | 1159.5 | 1159.5 | success |
| warm | 10 | 921.7 | 946.1 | 959.8 | 922.7 | success |

### Reference: `llama3.2:3b` (before Qwen pull)

| Phase | n | p50_ms | p95_ms | max_ms | avg_ms | status |
|-------|---|--------|--------|--------|--------|--------|
| cold (1st) | 1 | 1735.4 | 1735.4 | 1735.4 | 1735.4 | success |
| warm | 10 | 1796.1 | 1894.5 | 1924.8 | 1798.3 | success |

## User-facing E2E (gateway typed ask → answer)

Measured 2026-08-07 on `POST /api/v1/copilot/query` (cockpit text path).  
Metric: **client wall-clock** from request start until full JSON body received (intent + Core AI/Qwen + TTS).

Harness: [`backend-orchestrator/scripts/measure_user_e2e.py`](../../backend-orchestrator/scripts/measure_user_e2e.py)

| Phase | User wait (wall_ms) | Core AI | TTS | Notes |
|-------|---------------------|---------|-----|-------|
| **First user query** | **~9.4 s** | ~4.4 s | ~4.9 s | Cold-ish; `status=success`, audio present |
| **Warm (bypass, n=5) p95** | **~2.0 s** | ~0.8 s | ~1.2 s | Typical repeat ask without cache |
| **Cache HIT** | **~2–3 ms** | 0 | cached | Identical query within TTL; text answer only (no audio replay) |

**Interpretation:** User-perceived speed on a fresh ask is dominated by **Core AI generation + TTS**, not intent routing. Warm typed path lands near **~2s**. Cache HIT is instant for text. Do not claim &lt;300ms for generation+TTS E2E.

**Do not claim <300ms end-to-end with 7B generation without extractive mode.**

- [`tests/test_latency.py`](../tests/test_latency.py) asserts a **retrieval-only** path (expand mocked; Chroma/ONNX).
- Full RAG + LLM generation is governed by the warm scorecard above (extractive <2s **or** Qwen warm <5s p95).
- Measured Qwen warm ~0.95s is **honest generation latency**, not a 300ms E2E marketing claim.

## Reproduce

```bash
ollama pull qwen2.5:7b-instruct

# Extractive
LLM_PROVIDER=none uv run uvicorn main:app --app-dir src --host 127.0.0.1 --port 8001
uv run python scripts/measure_cold_warm.py --fail-on-p95-ms 2000

# Freeze Qwen (keep samples consecutive)
LLM_PROVIDER=ollama OPENAI_MODEL=qwen2.5:7b-instruct \
  uv run uvicorn main:app --app-dir src --host 127.0.0.1 --port 8001
uv run python scripts/measure_cold_warm.py --language en --fail-on-p95-ms 5000
```
