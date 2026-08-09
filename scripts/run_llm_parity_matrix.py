"""Multi-provider Bedrock + local Ollama Acc parity for judge evidence.

Runs the same --diverse case set in-process (one retrieval stack, swap answer LLM).

Usage:
  AWS creds in env; then:
  uv run python scripts/run_llm_parity_matrix.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Creds / region before importing settings-heavy modules
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
os.environ.setdefault("AWS_REGION", "ap-southeast-2")

from core.config import settings  # noqa: E402
from pipelines.solve_problem import solve_automotive_query_live  # noqa: E402

# Reuse case builder + scorer from smoke harness
import massive_situation_smoke as smoke  # noqa: E402


# Judge matrix: local baseline + one strong pick per Bedrock provider family
CANDIDATES: list[tuple[str, str, str]] = [
    ("ollama", "qwen2.5:7b-instruct", "Local Ollama freeze"),
    ("bedrock", "amazon.nova-lite-v1:0", "Amazon Nova Lite"),
    ("bedrock", "amazon.nova-pro-v1:0", "Amazon Nova Pro"),
    ("bedrock", "global.amazon.nova-2-lite-v1:0", "Amazon Nova 2 Lite"),
    ("bedrock", "openai.gpt-oss-120b-1:0", "OpenAI GPT-OSS 120B"),
    ("bedrock", "openai.gpt-oss-20b-1:0", "OpenAI GPT-OSS 20B"),
    ("bedrock", "qwen.qwen3-32b-v1:0", "Bedrock Qwen3 32B"),
    ("bedrock", "qwen.qwen3-next-80b-a3b", "Bedrock Qwen3 Next 80B"),
    ("bedrock", "deepseek.v3.2", "DeepSeek V3.2"),
    ("bedrock", "mistral.mistral-large-2402-v1:0", "Mistral Large"),
    ("bedrock", "nvidia.nemotron-super-3-120b", "NVIDIA Nemotron Super 120B"),
    ("bedrock", "moonshotai.kimi-k2.5", "Moonshot Kimi K2.5"),
]

# Resume-only subset when --resume-rest is passed
RESUME_REST: list[tuple[str, str, str]] = [
    ("bedrock", "qwen.qwen3-next-80b-a3b", "Bedrock Qwen3 Next 80B"),
    ("bedrock", "deepseek.v3.2", "DeepSeek V3.2"),
    ("bedrock", "mistral.mistral-large-2402-v1:0", "Mistral Large"),
    ("bedrock", "nvidia.nemotron-super-3-120b", "NVIDIA Nemotron Super 120B"),
    ("bedrock", "moonshotai.kimi-k2.5", "Moonshot Kimi K2.5"),
]

# Strongest callable Bedrock picks for Acc challenge (--best)
BEST_ONLY: list[tuple[str, str, str]] = [
    ("bedrock", "qwen.qwen3-235b-a22b-2507-v1:0", "Bedrock Qwen3 235B"),
    ("bedrock", "qwen.qwen3-next-80b-a3b", "Bedrock Qwen3 Next 80B"),
    ("bedrock", "deepseek.v3.2", "DeepSeek V3.2"),
    ("bedrock", "nvidia.nemotron-super-3-120b", "NVIDIA Nemotron Super 120B"),
    ("bedrock", "mistral.mistral-large-2402-v1:0", "Mistral Large"),
    ("bedrock", "moonshotai.kimi-k2.5", "Moonshot Kimi K2.5"),
]


def _configure(provider: str, model: str) -> None:
    settings.llm_provider = provider
    if provider == "ollama":
        settings.openai_base_url = os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:11434/v1"
        )
        settings.openai_model = model
        settings.openai_api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    else:
        settings.bedrock_model_id = model
        settings.aws_region = os.environ.get("AWS_REGION", "ap-southeast-2")


def _score_rag_case(case: smoke.Case, data: dict, ms: int) -> dict:
    # Mirror smoke.run_case scoring for rag bucket only via temporary Case path
    # by constructing a fake HTTP-like payload shape used in run_case.
    # Easiest: reuse run_case logic by posting locally — instead inline score.

    class _Fake:
        pass

    # Call the same scoring by fabricating what run_case expects after response
    # Copy minimal fields into smoke.run_case by monkeypatching _post — too heavy.
    # Use duplicated scoring entry: invoke smoke.run_case with patched CORE.

    status = data.get("status", "")
    answer = data.get("answer") or ""
    cites = data.get("citations") or []
    handoff = bool(data.get("handoff"))
    cmd = data.get("command_id")
    blob = (
        answer
        + " "
        + " ".join(
            f"{c.get('document_name', '')} {c.get('matched_text', '')}" for c in cites
        )
    ).lower()

    ok = True
    reasons: list[str] = []
    if status == "success":
        if handoff:
            if case.expect_any or case.doc_hint:
                ok = False
                reasons.append("handoff_instead_of_grounded")
        else:
            if case.expect_any and not any(k.lower() in blob for k in case.expect_any):
                ok = False
                reasons.append("missing_expect_any_in_answer_or_cites")
            if case.doc_hint:
                names = " ".join(str(c.get("document_name") or "") for c in cites)
                if case.doc_hint.lower() not in names.lower():
                    reasons.append(f"doc_hint_miss:{case.doc_hint}")
                    ok = False
    elif status == "not_found":
        ok = False
        reasons.append("not_found")
    else:
        ok = False
        reasons.append(f"status={status}")

    return {
        "tag": case.tag,
        "bucket": case.bucket,
        "ok": ok,
        "ms": ms,
        "status": status,
        "handoff": handoff,
        "n_cites": len(cites),
        "cite0": (cites[0].get("document_name") if cites else None),
        "answer_preview": answer.replace("\n", " ")[:140],
        "reason": "; ".join(reasons) if reasons else "",
        "answer_path": data.get("answer_path"),
    }


def _ollama_reachable() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def run_model(provider: str, model: str, label: str, cases: list) -> dict:
    print(f"\n=== {label} | {provider} | {model} ===", flush=True)
    if provider == "ollama" and not _ollama_reachable():
        print("  SKIP ollama unreachable — use prior parity_ollama_diverse.json", flush=True)
        prior = ROOT / "output" / "parity_ollama_diverse.json"
        if prior.exists():
            rows = json.loads(prior.read_text(encoding="utf-8"))
            rag = [r for r in rows if r.get("bucket") == "rag"]
            grounded = sum(1 for r in rag if r.get("ok") and not r.get("handoff"))
            ms_vals = sorted(int(r.get("ms") or 0) for r in rag) or [0]
            n = len(ms_vals)
            return {
                "label": label,
                "provider": provider,
                "model": model,
                "total": len(rag),
                "passed": sum(1 for r in rag if r.get("ok")),
                "grounded": grounded,
                "pass_rate": round(100.0 * grounded / max(1, len(rag)), 1),
                "handoff_rate": sum(1 for r in rag if r.get("handoff")),
                "latency_p50_ms": ms_vals[n // 2],
                "latency_p95_ms": ms_vals[min(n - 1, int(0.95 * (n - 1)))],
                "latency_max_ms": ms_vals[-1],
                "fails": [
                    {"tag": r["tag"], "reason": r.get("reason"), "cite0": r.get("cite0")}
                    for r in rag
                    if not r.get("ok")
                ],
                "source": "prior_artifact",
            }
        return {
            "label": label,
            "provider": provider,
            "model": model,
            "error": "ollama_unreachable",
            "grounded": 0,
            "total": 0,
            "pass_rate": 0.0,
        }

    _configure(provider, model)
    # Warm one RAG
    t0 = time.perf_counter()
    try:
        solve_automotive_query_live(
            "seatbelt pretensioner how does it work", language="en"
        )
        print(f"  warmup_ms={int((time.perf_counter() - t0) * 1000)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARMUP_FAIL {exc}", flush=True)
        return {
            "label": label,
            "provider": provider,
            "model": model,
            "error": str(exc),
            "grounded": 0,
            "total": 0,
            "pass_rate": 0.0,
        }

    results = []
    for i, case in enumerate(cases, 1):
        if case.bucket != "rag":
            continue
        print(f"  [{i}] {case.tag} …", flush=True)
        t1 = time.perf_counter()
        try:
            data = solve_automotive_query_live(
                case.query, language=case.language or "en"
            )
            ms = int((time.perf_counter() - t1) * 1000)
            row = _score_rag_case(case, data, ms)
        except Exception as exc:  # noqa: BLE001
            ms = int((time.perf_counter() - t1) * 1000)
            row = {
                "tag": case.tag,
                "bucket": "rag",
                "ok": False,
                "ms": ms,
                "status": "error",
                "handoff": False,
                "n_cites": 0,
                "cite0": None,
                "answer_preview": "",
                "reason": f"exception:{type(exc).__name__}:{exc}",
            }
        results.append(row)
        mark = "PASS" if row["ok"] else "FAIL"
        print(
            f"    {mark} {row['status']} {row['ms']}ms :: {row.get('cite0')} {row.get('reason','')}",
            flush=True,
        )

    rag = results
    grounded = sum(1 for r in rag if r["ok"] and not r.get("handoff"))
    ms_vals = sorted(int(r.get("ms") or 0) for r in rag) or [0]
    n = len(ms_vals)
    summary = {
        "label": label,
        "provider": provider,
        "model": model,
        "total": len(rag),
        "passed": sum(1 for r in rag if r["ok"]),
        "grounded": grounded,
        "pass_rate": round(100.0 * grounded / max(1, len(rag)), 1),
        "handoff_rate": sum(1 for r in rag if r.get("handoff")),
        "latency_p50_ms": ms_vals[n // 2],
        "latency_p95_ms": ms_vals[min(n - 1, int(0.95 * (n - 1)))],
        "latency_max_ms": ms_vals[-1],
        "fails": [
            {"tag": r["tag"], "reason": r.get("reason"), "cite0": r.get("cite0")}
            for r in rag
            if not r["ok"]
        ],
        "results": results,
    }
    print(
        f"  SUMMARY grounded={grounded}/{len(rag)} ({summary['pass_rate']}%) "
        f"p50={summary['latency_p50_ms']} p95={summary['latency_p95_ms']}",
        flush=True,
    )
    return summary


def main() -> int:
    cases = smoke.build_cases(diverse_only=True)
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional filter: --only substring
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = a.split("=", 1)[1].lower()

    candidates = CANDIDATES
    if "--resume-rest" in sys.argv:
        candidates = RESUME_REST
    if "--best" in sys.argv:
        candidates = BEST_ONLY
    if only:
        candidates = [c for c in candidates if only in c[1].lower() or only in c[2].lower()]

    summaries = []
    for provider, model, label in candidates:
        s = run_model(provider, model, label, cases)
        summaries.append({k: v for k, v in s.items() if k != "results"})
        # Persist full per-model artifact
        slug = model.replace("/", "_").replace(":", "_").replace(".", "_")
        (out_dir / f"parity_matrix_{slug}.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    leaderboard = sorted(
        [s for s in summaries if s.get("total")],
        key=lambda x: (-x.get("grounded", 0), x.get("latency_p95_ms", 10**9)),
    )
    report = {
        "suite": "diverse_manual",
        "n_cases": 38,
        "kpi_order": ["accuracy", "latency", "budget"],
        "leaderboard": leaderboard,
        "summaries": summaries,
    }
    path = out_dir / "LLM_PARITY_MATRIX.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {path}", flush=True)
    print("\n=== LEADERBOARD (Acc desc, then Lat p95 asc) ===", flush=True)
    for i, s in enumerate(leaderboard, 1):
        print(
            f"{i}. {s['pass_rate']}% ({s['grounded']}/{s['total']}) "
            f"p95={s['latency_p95_ms']}ms | {s['label']} | {s['model']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
