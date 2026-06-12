#!/usr/bin/env python3
"""
Deduplicate raw ClinAuthBench model-output JSONL files.

Keeps one row per case_id:
1. Prefer the latest row with status == "ok"
2. If no ok row exists, keep the latest row

This script does not parse, validate, or score model outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")

            row["_source_line_number"] = line_number
            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean_row = dict(row)
            clean_row.pop("_source_line_number", None)
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")


def deduplicate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Missing case_id on source line {row.get('_source_line_number')}")

        by_case.setdefault(case_id, []).append(row)

    selected: List[Dict[str, Any]] = []

    for case_id in sorted(by_case):
        candidates = by_case[case_id]

        ok_rows = [r for r in candidates if r.get("status") == "ok"]

        if ok_rows:
            selected.append(ok_rows[-1])
        else:
            selected.append(candidates[-1])

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate raw ClinAuthBench model-output JSONL."
    )

    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=180)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rows = load_jsonl(args.input_path)
    deduped = deduplicate(rows)

    if args.expected_count and len(deduped) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} deduped rows, found {len(deduped)}"
        )

    write_jsonl(args.output_path, deduped)

    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(deduped)}")
    print(f"Wrote: {args.output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())