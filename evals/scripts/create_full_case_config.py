#!/usr/bin/env python3
"""
Create the ClinAuthBench v1 full 180-case evaluation config.

This script:
- reads the released dataset JSON
- extracts all case IDs in dataset order
- writes evals/config/v1_full_180_cases.json

It does NOT:
- call any model
- parse model outputs
- score results
- read metadata.gold or hidden labels
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def get_records(dataset: Any) -> List[Dict[str, Any]]:
    if isinstance(dataset, list):
        records = dataset
    elif isinstance(dataset, dict):
        if isinstance(dataset.get("cases"), list):
            records = dataset["cases"]
        elif isinstance(dataset.get("data"), list):
            records = dataset["data"]
        elif isinstance(dataset.get("test"), list):
            records = dataset["test"]
        else:
            raise ValueError(
                "Dataset JSON must be a top-level list or contain a list under cases/data/test."
            )
    else:
        raise ValueError(f"Expected dataset JSON list or object, got {type(dataset).__name__}")

    clean_records = [record for record in records if isinstance(record, dict)]

    if len(clean_records) != len(records):
        raise ValueError("Dataset contains non-object records.")

    return clean_records


def extract_case_ids(records: List[Dict[str, Any]]) -> List[str]:
    case_ids: List[str] = []

    for idx, record in enumerate(records):
        case_id = record.get("id") or record.get("case_id")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Record at index {idx} is missing a valid case id.")

        case_ids.append(case_id.strip())

    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})

    if duplicates:
        raise ValueError(f"Duplicate case IDs found: {duplicates}")

    return case_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ClinAuthBench v1 full 180-case config."
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/release/synthetic_bh_cases_v1_mdp_180.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("evals/config/v1_full_180_cases.json"),
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=180,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset = read_json(args.dataset_path)
    records = get_records(dataset)
    case_ids = extract_case_ids(records)

    if args.expected_count and len(case_ids) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} cases, found {len(case_ids)}."
        )

    payload = {
        "case_set": "v1_full_180_cases",
        "created_utc": utc_now_iso(),
        "dataset_path": str(args.dataset_path),
        "n_cases": len(case_ids),
        "case_ids": case_ids,
    }

    write_json(args.output_path, payload)

    print(f"Wrote: {args.output_path}")
    print(f"Cases: {len(case_ids)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())