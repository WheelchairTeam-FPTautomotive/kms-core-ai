#!/usr/bin/env python3
"""Export golden_set_s2.json to run.sh-compatible input (query+language objects)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "data" / "test_queries" / "golden_set_s2.json"
DEFAULT_OUT_DIR = ROOT / "data" / "test_queries" / "golden_s2_input"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export S2 golden queries for run.sh")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    cases = json.loads(args.golden.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise SystemExit(f"Expected list in {args.golden}")

    exported = []
    for case in cases:
        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            raise SystemExit(f"Invalid query in case: {case!r}")
        language = case.get("language") or "vi"
        if language not in ("vi", "en"):
            raise SystemExit(f"Invalid language in case {case.get('id')}: {language}")
        exported.append({"query": query, "language": language})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "queries.json"
    out_file.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(exported)} queries -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
