#!/usr/bin/env python3
"""Run golden_set_s2 against live Core HTTP (:8001) and score ≥90%."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "test_queries" / "golden_set_s2.json"
OUT_RESULTS = ROOT / "output" / "golden_results_aws.json"
OUT_SCORE = ROOT / "output" / "golden_score_aws.md"
PASS_THRESHOLD = 0.90


def _norm(q: str) -> str:
    return " ".join((q or "").split()).strip().lower()


def _citation_blob(citations: list) -> str:
    parts = []
    for c in citations or []:
        parts.extend(
            [
                str(c.get("document_name") or ""),
                str(c.get("document_id") or ""),
                str(c.get("section") or ""),
                str(c.get("matched_text") or ""),
            ]
        )
    return " ".join(parts).lower()


def _keywords_hit(citations: list, keywords: list[str]) -> bool:
    if not keywords:
        return True
    blob = _citation_blob(citations)
    return any(kw.lower() in blob for kw in keywords if kw)


def post_core(base: str, query: str, language: str) -> dict:
    data = json.dumps(
        {"query": query, "mode": "rag", "language": language or "en"},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/api/v1/search",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--out-results", type=Path, default=OUT_RESULTS)
    ap.add_argument("--out-score", type=Path, default=OUT_SCORE)
    args = ap.parse_args()

    cases = json.loads(args.golden.read_text(encoding="utf-8"))
    results: list[dict] = []
    by_q: dict[str, dict] = {}

    for i, case in enumerate(cases, 1):
        q = case["query"]
        lang = case.get("language") or "en"
        t0 = time.perf_counter()
        try:
            body = post_core(args.base_url, q, lang)
            ms = int((time.perf_counter() - t0) * 1000)
            row = {
                "id": case.get("id"),
                "query": q,
                "status": body.get("status"),
                "citations": body.get("citations") or [],
                "answer": body.get("answer"),
                "handoff": body.get("handoff"),
                "ms": ms,
            }
        except Exception as exc:  # noqa: BLE001
            ms = int((time.perf_counter() - t0) * 1000)
            row = {
                "id": case.get("id"),
                "query": q,
                "status": "error",
                "citations": [],
                "answer": str(exc),
                "ms": ms,
            }
        results.append(row)
        by_q[_norm(q)] = row
        print(
            f"[{i}/{len(cases)}] {case.get('id')} status={row['status']} "
            f"cites={len(row['citations'])} {ms}ms"
        )

    args.out_results.parent.mkdir(parents=True, exist_ok=True)
    args.out_results.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    passed = 0
    failures = []
    for case in cases:
        cid = case.get("id", "?")
        res = by_q.get(_norm(case["query"]))
        expected = case.get("expected_status")
        if res is None:
            failures.append(f"{cid}: missing_result")
            continue
        actual = res.get("status")
        if actual != expected:
            failures.append(f"{cid}: status_mismatch expected={expected} actual={actual}")
            continue
        if expected == "success":
            kws = case.get("expected_doc_keywords") or []
            if not _keywords_hit(res.get("citations") or [], kws):
                failures.append(f"{cid}: citation_keywords_miss {kws}")
                continue
        passed += 1

    total = len(cases)
    rate = passed / max(1, total)
    ok = rate >= PASS_THRESHOLD
    lines = [
        "# Golden S2 Score Report (AWS live Core)",
        "",
        f"- Generated (UTC): `{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- Core base: `{args.base_url}`",
        f"- Threshold: {PASS_THRESHOLD:.0%}",
        "",
        f"## Result: **{passed}/{total}** ({100 * rate:.1f}%) — threshold {PASS_THRESHOLD:.0%}",
        "",
        f"Overall: `{'PASS' if ok else 'FAIL'}`",
        "",
    ]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for f in failures:
            lines.append(f"- `{f}`")
        lines.append("")
    args.out_score.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
