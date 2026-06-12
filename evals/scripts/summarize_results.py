#!/usr/bin/env python3
"""
Summarize ClinAuthBench v1 full-180 evaluation results.

This script:
- reads scored metrics JSON files
- reads parse/schema summary JSON files
- writes one consolidated JSON summary
- writes one Markdown summary table

It does NOT:
- call any model
- parse raw model outputs
- validate schemas
- rescore cases
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_METRICS_PATHS = [
    Path("evals/model_outputs/scored/hf/openai__gpt-oss-120b/v1_full_180_metrics.json"),
    Path("evals/model_outputs/scored/openai/gpt-5.2/v1_full_180_metrics.json"),
    Path("evals/model_outputs/scored/hf/meta-llama__Meta-Llama-3-8B-Instruct/v1_full_180_metrics.json"),
]


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.1f}%"


def count_fraction(numerator: Optional[int], denominator: Optional[int]) -> str:
    if numerator is None or denominator is None:
        return "NA"
    return f"{numerator}/{denominator}"


def parse_summary_path_for(metrics_path: Path) -> Path:
    text = str(metrics_path)

    if text.endswith("_metrics.json"):
        return Path(text.replace("_metrics.json", "_parse_schema_summary.json"))

    raise ValueError(f"Cannot infer parse/schema summary path from: {metrics_path}")


def model_label(model_id: str, source: str) -> str:
    if model_id == "openai/gpt-oss-120b":
        return "GPT-OSS 120B"
    if model_id == "gpt-5.2":
        return "GPT-5.2"
    if model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
        return "Llama 3 8B Instruct"
    return model_id


def summarize_one(metrics_path: Path) -> Dict[str, Any]:
    parse_path = parse_summary_path_for(metrics_path)

    metrics = read_json(metrics_path)
    parse_summary = read_json(parse_path)

    overall = metrics.get("overall", {})
    model_id = metrics.get("model_id", "unknown")
    source = metrics.get("source", "unknown")

    entry = {
        "model_label": model_label(model_id, source),
        "model_id": model_id,
        "source": source,
        "case_set": metrics.get("case_set"),
        "metrics_path": str(metrics_path),
        "parse_schema_summary_path": str(parse_path),
        "n_cases": overall.get("n_cases"),
        "n_raw_jsonl_rows": parse_summary.get("n_raw_jsonl_rows"),
        "n_retrieval_ok": parse_summary.get("n_retrieval_ok"),
        "n_valid_json": parse_summary.get("n_valid_json"),
        "n_schema_valid": parse_summary.get("n_schema_valid"),
        "retrieval_ok_rate": parse_summary.get("retrieval_ok_rate"),
        "valid_json_rate": parse_summary.get("valid_json_rate"),
        "schema_valid_rate": parse_summary.get("schema_valid_rate"),
        "safe_for_lloc_accuracy": overall.get("safe_for_lloc_accuracy"),
        "expected_los_exact_match": overall.get("expected_los_exact_match"),
        "safe_for_lloc_accuracy_schema_valid_only": overall.get("safe_for_lloc_accuracy_schema_valid_only"),
        "expected_los_exact_match_schema_valid_only": overall.get("expected_los_exact_match_schema_valid_only"),
        "failure_counts_by_type": parse_summary.get("failure_counts_by_type", {}),
        "n_bad_jsonl_lines": parse_summary.get("n_bad_jsonl_lines"),
        "n_duplicate_case_ids_in_raw_jsonl": parse_summary.get("n_duplicate_case_ids_in_raw_jsonl"),
        "missing_case_ids": parse_summary.get("missing_case_ids", []),
        "extra_case_ids_in_raw_jsonl": parse_summary.get("extra_case_ids_in_raw_jsonl", []),
        "by_documentation_challenge": metrics.get("by_documentation_challenge", {}),
    }

    return entry


def markdown_table(rows: List[List[str]], headers: List[str]) -> str:
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def make_overall_markdown(results: List[Dict[str, Any]]) -> str:
    rows: List[List[str]] = []

    for r in results:
        n_cases = r.get("n_cases")

        rows.append(
            [
                r["model_label"],
                r["source"],
                str(n_cases),
                count_fraction(r.get("n_valid_json"), n_cases),
                count_fraction(r.get("n_schema_valid"), n_cases),
                pct(r.get("safe_for_lloc_accuracy")),
                pct(r.get("expected_los_exact_match")),
                pct(r.get("safe_for_lloc_accuracy_schema_valid_only")),
                pct(r.get("expected_los_exact_match_schema_valid_only")),
            ]
        )

    return markdown_table(
        rows=rows,
        headers=[
            "Model",
            "Source",
            "Cases",
            "Valid JSON",
            "Schema valid",
            "Safe-for-LLOC accuracy",
            "LOS exact match",
            "Safe valid-only",
            "LOS valid-only",
        ],
    )


def make_challenge_markdown(results: List[Dict[str, Any]]) -> str:
    sections: List[str] = []

    for r in results:
        challenge_rows: List[List[str]] = []

        by_challenge = r.get("by_documentation_challenge", {})

        for challenge, metrics in sorted(by_challenge.items()):
            n_cases = metrics.get("n_cases")
            n_schema_valid = metrics.get("n_schema_valid")

            challenge_rows.append(
                [
                    challenge,
                    str(n_cases),
                    count_fraction(n_schema_valid, n_cases),
                    pct(metrics.get("safe_for_lloc_accuracy")),
                    pct(metrics.get("expected_los_exact_match")),
                    pct(metrics.get("safe_for_lloc_accuracy_schema_valid_only")),
                    pct(metrics.get("expected_los_exact_match_schema_valid_only")),
                ]
            )

        sections.append(f"## {r['model_label']} challenge breakdown\n")
        sections.append(
            markdown_table(
                rows=challenge_rows,
                headers=[
                    "Documentation challenge",
                    "Cases",
                    "Schema valid",
                    "Safe-for-LLOC accuracy",
                    "LOS exact match",
                    "Safe valid-only",
                    "LOS valid-only",
                ],
            )
        )
        sections.append("")

    return "\n".join(sections).strip()


def make_markdown(payload: Dict[str, Any]) -> str:
    results = payload["results"]

    lines = [
        "# ClinAuthBench v1 full-180 evaluation summary",
        "",
        "ClinAuthBench v1 is fully synthetic and is not real patient data.",
        "",
        "Primary metrics use all expected cases as the denominator. Retrieval, JSON parsing, and schema-validation failures count as incorrect. Valid-only metrics are included only as diagnostics.",
        "",
        "## Overall results",
        "",
        make_overall_markdown(results),
        "",
        "## Notes",
        "",
        "- GPT-OSS 120B and GPT-5.2 produced schema-valid JSON for all 180 cases.",
        "- Llama 3 8B Instruct is reported as a smaller open-weight diagnostic model because many outputs failed the strict JSON/schema contract.",
        "- These results should not be interpreted as clinical performance, payer-policy performance, or real-world authorization safety.",
        "",
        make_challenge_markdown(results),
        "",
    ]

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ClinAuthBench v1 full-180 evaluation results."
    )

    parser.add_argument(
        "--metrics-path",
        type=Path,
        action="append",
        default=None,
        help="Path to a v1_full_180_metrics.json file. Can be repeated.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evals/results/v1_full_180_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("evals/results/v1_full_180_summary.md"),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    metrics_paths = args.metrics_path or DEFAULT_METRICS_PATHS

    results = [summarize_one(path) for path in metrics_paths]

    payload = {
        "record_type": "clinauthbench_v1_full_180_results_summary",
        "created_utc": utc_now_iso(),
        "n_models": len(results),
        "metric_definition": {
            "primary_denominator": "all expected cases",
            "failure_policy": "retrieval, parse, and schema failures count as incorrect",
            "valid_only_metrics": "diagnostic only",
        },
        "results": results,
    }

    write_json(args.output_json, payload)
    write_text(args.output_md, make_markdown(payload))

    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_md}")

    for r in results:
        print(
            f"{r['model_label']}: "
            f"safe={pct(r.get('safe_for_lloc_accuracy'))}, "
            f"los={pct(r.get('expected_los_exact_match'))}, "
            f"schema={count_fraction(r.get('n_schema_valid'), r.get('n_cases'))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())