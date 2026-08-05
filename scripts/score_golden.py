"""Score golden_set_s2 results: status match + citation keyword grounding (≥90%)."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "data" / "test_queries" / "golden_set_s2.json"
DEFAULT_RESULTS = ROOT / "output" / "golden_results.json"
DEFAULT_OUT = ROOT / "output" / "golden_score.md"
PASS_THRESHOLD = 0.90


def _norm(q: str) -> str:
    return " ".join((q or "").split()).strip().lower()


def _citation_blob(citations: list[dict[str, Any]]) -> str:
    parts: list[str] = []
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


def _keywords_hit(citations: list[dict[str, Any]], keywords: list[str]) -> bool:
    if not keywords:
        return True
    blob = _citation_blob(citations)
    return any(kw.lower() in blob for kw in keywords if kw)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _settings_snapshot() -> dict[str, str]:
    try:
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from core.config import settings

        return {
            "openai_model": str(settings.openai_model),
            "llm_provider": str(settings.llm_provider),
            "openai_base_url": str(settings.openai_base_url),
            "vector_db_type": str(settings.vector_db_type),
            "rag_max_distance": str(settings.rag_max_distance),
        }
    except (ImportError, OSError) as e:
        return {"error": str(e)}


def score_case(case: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any]:
    cid = case.get("id", "?")
    expected = case.get("expected_status")
    if result is None:
        return {
            "id": cid,
            "pass": False,
            "reason": "missing_result",
            "expected_status": expected,
            "actual_status": None,
        }

    actual = result.get("status")
    if actual != expected:
        return {
            "id": cid,
            "pass": False,
            "reason": f"status_mismatch expected={expected} actual={actual}",
            "expected_status": expected,
            "actual_status": actual,
        }

    if expected == "success":
        citations = result.get("citations") or []
        keywords = case.get("expected_doc_keywords") or []
        if not citations:
            return {
                "id": cid,
                "pass": False,
                "reason": "success_empty_citations",
                "expected_status": expected,
                "actual_status": actual,
            }
        if keywords and not _keywords_hit(citations, keywords):
            names = [c.get("document_name") for c in citations]
            return {
                "id": cid,
                "pass": False,
                "reason": f"citation_keyword_miss keywords={keywords} docs={names}",
                "expected_status": expected,
                "actual_status": actual,
            }

    return {
        "id": cid,
        "pass": True,
        "reason": "ok",
        "expected_status": expected,
        "actual_status": actual,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score S2 golden RAG results")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=PASS_THRESHOLD,
        help="Minimum pass rate (default 0.90)",
    )
    args = parser.parse_args()

    cases = json.loads(args.golden.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not isinstance(results, list):
        raise SystemExit("golden and results must be JSON arrays")

    by_query: dict[str, dict[str, Any]] = {}
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("query"), str):
            by_query[_norm(r["query"])] = r

    rows = []
    for case in cases:
        q = case.get("query") if isinstance(case, dict) else None
        result = by_query.get(_norm(q)) if isinstance(q, str) else None
        rows.append(score_case(case, result))

    passed = sum(1 for r in rows if r["pass"])
    total = len(rows)
    rate = (passed / total) if total else 0.0
    snap = _settings_snapshot()
    sha = _git_sha()
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "generated_at_utc": now,
        "git_sha": sha,
        "settings": snap,
        "threshold": args.threshold,
        "passed": passed,
        "total": total,
        "pass_rate": round(rate, 4),
        "ok": rate >= args.threshold,
        "cases": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_out = args.out.with_suffix(".json")
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fail_lines = [
        f"- `{r['id']}`: {r['reason']}" for r in rows if not r["pass"]
    ]
    md = [
        "# Golden S2 Score Report",
        "",
        f"- Generated (UTC): `{now}`",
        f"- Git SHA: `{sha}`",
        f"- OPENAI_MODEL: `{snap.get('openai_model', '?')}`",
        f"- LLM_PROVIDER: `{snap.get('llm_provider', '?')}`",
        f"- OPENAI_BASE_URL: `{snap.get('openai_base_url', '?')}`",
        f"- VECTOR_DB_TYPE: `{snap.get('vector_db_type', '?')}`",
        f"- RAG_MAX_DISTANCE: `{snap.get('rag_max_distance', '?')}`",
        "- Chroma rebuild: run `uv run python src/pipelines/ingest.py --target chroma --reset` before scoring",
        "- Note: OEM manuals live under repo `data/docs_pdf`; set `DOCS_PDF_DIR=data/docs_pdf` if `.env` points at HACKATHON-only paths",
        "",
        f"## Result: **{passed}/{total}** ({rate:.1%}) — threshold {args.threshold:.0%}",
        "",
        f"Overall: `{'PASS' if payload['ok'] else 'FAIL'}`",
        "",
        "## Failures",
        "",
    ]
    if fail_lines:
        md.extend(fail_lines)
    else:
        md.append("_None_")
    md.append("")
    args.out.write_text("\n".join(md), encoding="utf-8")

    print(f"Score {passed}/{total} ({rate:.1%}) -> {args.out}")
    print(f"JSON detail -> {json_out}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
