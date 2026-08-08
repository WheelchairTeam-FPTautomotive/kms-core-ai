# Gateway diverse report (customer path)

**Suite:** `massive_situation_smoke.py --diverse --gateway`  
**Path:** Gateway `:8000` → Core AI `:8001`  
**Artifact:** [`../output/diverse_manual_smoke_gateway.json`](../output/diverse_manual_smoke_gateway.json)  
**Date:** 2026-08-08 (post RC1–RC3 overwhelm)

---

## Executive verdict

| KPI | Before (baseline) | After RC1–RC3 | Target |
|---|---|---|---|
| **Diverse grounded SLI** | 92.1% (35/38) | **100% (38/38)** | ≥95% stretch |
| **SLO floor (≥90%)** | PASS | **PASS** | ≥90% |
| **Stretch (≥95%)** | FAIL | **PASS** | ≥95% |
| **Handoff rate** | 5.3% (2/38) | **0% (0/38)** | ≤10% |
| **Latency p50** | 11.2s (TTS) | **7.6s** (TTS) | voice still TTS-bound |

Stretch is **overwhelmed**. RC1–RC3 cleared without raising `RAG_MAX_DISTANCE`. Remaining bottleneck is **voice latency (TTS + LLM)**, not grounded accuracy.

---

## Core vs gateway (same 38-case suite)

| Metric | Core `:8001` | Gateway `:8000` | SLO floor | Stretch |
|---|---|---|---|---|
| Grounded pass rate | **100% (38/38)** | **100% (38/38)** | ≥90% | ≥95% |
| Handoff rate | **0% (0/38)** | **0% (0/38)** | ≤25% | ≤10% |
| Fail count | 0 | 0 | — | — |
| Latency p50 | ~6–7s | **7.6s** | unset | ≪6s voice |
| Latency p95 / max | — | **19.7s / 42.1s** | unset | — |

**Delta Core → Gateway:** 0pp grounded (parity). First scored case after discard warmup: **9.3s** (no cold handoff).

---

## RC1–RC3 fixes shipped

| RC | Case | Fix | Result |
|---|---|---|---|
| RC1 cold-start | `div-seatbelt-pretensioner` | Blocking lifespan warm; health `ready` only after retrieval warm; smoke discard warmup | PASS, cites, ~9s first scored |
| RC2 trim alias | `div-wheel-nut-torque` | `bronco raptor`/`raptor` → `bronco`; title/trim pins; metadata dedupe; BM25 rebuild | PASS cite `2024-ford-bronco.pdf` (WHEEL NUTS - RAPTOR) |
| RC3 VI→EN | `div-vi-defroster` | Prepend OEM EN (`rear window defroster`); BM25 fan-in 2→4 | PASS cite Tucson + defrost tokens |

---

## Failures

**None** on post-fix gateway diverse.

---

## Historical baseline RCA (pre-fix)

Prior run was **35/38**. Root causes (resolved above):

1. **RC1** — first query paid CE/planner/Chroma warm → soft handoff canary.
2. **RC2** — planner pin `raptor`/`bronco raptor` vs metadata `bronco` filtered torque chunk.
3. **RC3** — BM25 `queries[:2]` dropped OEM expands after VI original.

Do **not** reopen by loosening `RAG_MAX_DISTANCE`.

---

## Production judgment

| Question | Answer |
|---|---|
| Accuracy floor ready? | **Yes** (100% diverse grounded) |
| Stretch overwhelmed? | **Yes** (≥95% cleared with margin) |
| Full prod voice latency? | **Not yet** — TTS still dominates p50/p95 |
| Safe next step | Commit/push Core + gateway; optional `--no-tts` smoke for retrieval-only latency |

---

## Reproduce

```bash
# Core
cd kms-core-ai
uv run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8001

# Gateway
cd backend-orchestrator
uv run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000

# Prove
cd kms-core-ai
uv run python scripts/massive_situation_smoke.py --diverse
uv run python scripts/massive_situation_smoke.py --diverse --gateway
```
