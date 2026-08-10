#!/usr/bin/env python3
"""Run demo_pack_slo against local or AWS gateway (full UC1–UC3 pack).

Usage:
  python scripts/run_demo_pack_slo.py --base-url http://127.0.0.1:8000
  python scripts/run_demo_pack_slo.py --base-url http://52.64.18.95:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "data" / "test_queries" / "demo_pack_slo.json"


def _status_ok(got: str, expect) -> bool:
    if isinstance(expect, list):
        return got in expect
    return got == expect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--pack", type=Path, default=PACK_PATH)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--session-ttl-min",
        type=int,
        default=5,
        help="STM idle TTL minutes for session_group chains (0=off)",
    )
    args = parser.parse_args()

    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    cases = pack["cases"]
    url = args.base_url.rstrip("/") + "/api/v1/copilot/query"

    # Persist session_id per STM group so follow-ups share conversation_context
    session_ids: dict[str, str] = {}

    passed = 0
    rows = []
    by_uc: dict[str, list[bool]] = {}

    with httpx.Client(timeout=args.timeout) as client:
        for case in cases:
            uc = case.get("use_case") or "OTHER"
            group = case.get("session_group")
            payload: dict = {
                "query": case["query"],
                "language": case.get("language", "vi"),
            }
            if group:
                if group not in session_ids:
                    session_ids[group] = str(uuid.uuid4())
                payload["session_id"] = session_ids[group]
                payload["session_ttl_min"] = args.session_ttl_min

            t0 = time.perf_counter()
            r = client.post(
                url,
                json=payload,
                headers={"X-Cache-Bypass": "1", "X-Skip-TTS": "1"},
            )
            ms = int((time.perf_counter() - t0) * 1000)
            body = (
                r.json()
                if r.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            # Echo session for next turn in group
            if group and body.get("session_id"):
                session_ids[group] = body["session_id"]

            status = body.get("status")
            cmd = body.get("command_id")
            ok_status = _status_ok(str(status), case["expect_status"])
            expect_cmd = case.get("expect_command_id")
            ok_cmd = (cmd == expect_cmd) if "expect_command_id" in case else True
            min_cites = int(case.get("expect_min_citations") or 0)
            cite_n = len(body.get("citations") or [])
            ok_cites = cite_n >= min_cites

            if case.get("expect_intent") == "REFUSED":
                ok_intent = status == "refused"
            elif case.get("expect_intent") == "CAR_CONTROL":
                ok_intent = status in {"success", "refused"} and (
                    cmd == expect_cmd or (expect_cmd is None and status == "refused")
                )
            else:
                ok_intent = True

            ok = (
                r.status_code == 200
                and ok_status
                and ok_cmd
                and ok_intent
                and ok_cites
            )
            if ok:
                passed += 1
            by_uc.setdefault(uc, []).append(ok)
            rows.append(
                {
                    "id": case["id"],
                    "use_case": uc,
                    "ok": ok,
                    "http": r.status_code,
                    "status": status,
                    "command_id": cmd,
                    "citations": cite_n,
                    "ms": ms,
                    "session_group": group,
                    "answer": (body.get("answer") or "")[:120],
                }
            )
            mark = "PASS" if ok else "FAIL"
            print(
                f"[{mark}] {case['id']} status={status} cmd={cmd} "
                f"cites={cite_n} {ms}ms"
            )

    total = len(cases)
    print(f"\nSUMMARY {passed}/{total} ({100.0 * passed / max(1, total):.1f}%)")
    for uc, flags in by_uc.items():
        n = sum(1 for x in flags if x)
        print(f"  {uc}: {n}/{len(flags)}")

    out = ROOT / "output" / "demo_pack_slo_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "passed": passed,
                "total": total,
                "by_use_case": {
                    uc: {"passed": sum(1 for x in flags if x), "total": len(flags)}
                    for uc, flags in by_uc.items()
                },
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
