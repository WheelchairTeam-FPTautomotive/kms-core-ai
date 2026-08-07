"""Cold vs warm Core AI latency harness for issue #14.

Measures POST /api/v1/search end-to-end (retrieval + answer generation).

Ollama note: default keep_alive ~5 minutes unloads VRAM. Warm samples run
back-to-back with no sleep so mid-loop unload cannot inflate p95. If you pause
>5m between runs, re-warm or set Ollama keep_alive=-1 for the session.

This is NOT the retrieval-only SLA in tests/test_latency.py.
Do not claim <300ms E2E with 7B generation.

Example:
  uv run python scripts/measure_cold_warm.py --base-url http://localhost:8001
  uv run python scripts/measure_cold_warm.py --fail-on-p95-ms 2000   # extractive
  uv run python scripts/measure_cold_warm.py --fail-on-p95-ms 5000   # ollama/Qwen
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# --- START MODIFICATION ---
# Fixed query across extractive + LLM stacks for fair p50/p95 comparison (#14).
# Prefer a corpus-grounded golden-style question so status=success exercises generation
# (not_found short-circuits and under-reports LLM latency).
DEFAULT_QUERY = "What is the HVAC system?"
# --- END MODIFICATION ---


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _post_search(base_url: str, query: str, mode: str, language: str, timeout_s: float) -> tuple[float, dict]:
    url = base_url.rstrip("/") + "/api/v1/search"
    body = json.dumps({"query": query, "mode": mode, "language": language}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, payload


def _health(base_url: str, timeout_s: float = 5.0) -> None:
    url = base_url.rstrip("/") + "/api/v1/health"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        if resp.status != 200:
            raise RuntimeError(f"health status {resp.status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Core AI cold/warm latency → Markdown table")
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--n", type=int, default=10, help="Warm sample count (consecutive)")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--mode", default="rag", choices=["rag", "free_talk"])
    parser.add_argument("--language", default="vi", choices=["vi", "en"])
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--fail-on-p95-ms",
        type=float,
        default=None,
        help="Exit 1 if warm p95 exceeds this (e.g. 2000 extractive, 5000 ollama)",
    )
    parser.add_argument(
        "--skip-cold",
        action="store_true",
        help="Skip cold sample (process already warm)",
    )
    args = parser.parse_args()

    if args.n < 1:
        print("error: --n must be >= 1", file=sys.stderr)
        return 2

    try:
        _health(args.base_url)
    except Exception as exc:  # noqa: BLE001
        print(f"error: Core AI not healthy at {args.base_url}: {exc}", file=sys.stderr)
        return 2

    cold_ms: float | None = None
    cold_status = ""
    if not args.skip_cold:
        try:
            cold_ms, cold_payload = _post_search(
                args.base_url, args.query, args.mode, args.language, args.timeout_s
            )
            cold_status = str(cold_payload.get("status", "?"))
        except urllib.error.URLError as exc:
            print(f"error: cold request failed: {exc}", file=sys.stderr)
            return 2

    # Discard one warm ping so ONNX/LLM path is hot, then N consecutive samples.
    try:
        _post_search(args.base_url, args.query, args.mode, args.language, args.timeout_s)
    except urllib.error.URLError as exc:
        print(f"error: warm discard failed: {exc}", file=sys.stderr)
        return 2

    warm_ms: list[float] = []
    last_status = ""
    for _ in range(args.n):
        ms, payload = _post_search(
            args.base_url, args.query, args.mode, args.language, args.timeout_s
        )
        warm_ms.append(ms)
        last_status = str(payload.get("status", "?"))

    warm_sorted = sorted(warm_ms)
    p50 = _percentile(warm_sorted, 50)
    p95 = _percentile(warm_sorted, 95)
    warm_max = max(warm_ms)
    warm_avg = statistics.fmean(warm_ms)

    print("## Core AI cold / warm latency")
    print()
    print(f"- base_url: `{args.base_url}`")
    print(f"- query: `{args.query}`")
    print(f"- mode: `{args.mode}` language: `{args.language}`")
    print(f"- warm_n: {args.n} (consecutive; no sleep — Ollama keep_alive safe)")
    print(f"- last_status: `{last_status}`")
    print()
    print("| Phase | n | p50_ms | p95_ms | max_ms | avg_ms | status |")
    print("|-------|---|--------|--------|--------|--------|--------|")
    if cold_ms is not None:
        print(
            f"| cold (1st) | 1 | {cold_ms:.1f} | {cold_ms:.1f} | {cold_ms:.1f} | "
            f"{cold_ms:.1f} | {cold_status} |"
        )
    else:
        print("| cold (1st) | — | — | — | — | — | skipped |")
    print(
        f"| warm | {args.n} | {p50:.1f} | {p95:.1f} | {warm_max:.1f} | "
        f"{warm_avg:.1f} | {last_status} |"
    )
    print()
    print(
        "> Anti-claim: do **not** report <300ms E2E with 7B generation; "
        "`tests/test_latency.py` is retrieval-only."
    )

    if args.fail_on_p95_ms is not None and p95 > args.fail_on_p95_ms:
        print(
            f"\nFAIL: warm p95 {p95:.1f}ms > --fail-on-p95-ms {args.fail_on_p95_ms}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
