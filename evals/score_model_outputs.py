#!/usr/bin/env python3
"""
Score raw ClinAuthBench model outputs against local hidden gold labels.

This script intentionally does not call any model API and does not retrieve outputs.
It reads raw JSONL outputs produced earlier, parses/validates the model JSON,
loads gold labels locally, computes metrics, and writes aggregate result files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import jsonschema
    from jsonschema import validators
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: jsonschema\n"
        "Install it with: pip install jsonschema"
    ) from exc


MODEL_OUTPUT_TEXT_FIELDS = (
    "raw_output",
    "model_output",
    "output_text",
    "response_text",
    "content",
    "text",
)

SAFE_GOLD_KEYS = (
    "safe_for_lloc",
    "safe_for_lower_level_of_care",
)

LOS_GOLD_KEYS = (
    "expected_los_recommendation",
    "expected_los",
    "los_recommendation",
)

LOS_VALUES = {0, 1, 2, 3, 4}

CHALLENGE_KEYS = (
    "documentation_challenge",
    "documentation_challenge_category",
    "challenge",
)

OK_STATUSES = {None, "ok", "success", "completed"}


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


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", value.strip())
    return slug.strip("_") or "unknown"


def rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def get_first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False

    if isinstance(value, int) and value in {0, 1}:
        return bool(value)

    return None


def normalize_los(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None

    if isinstance(value, int) and value in LOS_VALUES:
        return value

    if isinstance(value, str):
        match = re.search(r"\b([0-4])\b", value.strip().lower())
        if match:
            return int(match.group(1))

    return None


def normalize_challenge(value: Any) -> str:
    if value is None:
        return "unknown"

    if isinstance(value, str):
        return value.strip() or "unknown"

    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return ";".join(parts) if parts else "unknown"

    if isinstance(value, dict):
        for key in ("name", "category", "label", "type"):
            if key in value and value[key] is not None:
                return str(value[key]).strip() or "unknown"

    return str(value).strip() or "unknown"


def normalize_case_id_entries(entries: Sequence[Any]) -> List[str]:
    case_ids: List[str] = []

    for entry in entries:
        if isinstance(entry, str):
            case_ids.append(entry)
        elif isinstance(entry, dict):
            case_id = entry.get("case_id") or entry.get("id")
            if not isinstance(case_id, str):
                raise ValueError(f"Could not extract case id from config entry: {entry}")
            case_ids.append(case_id)
        else:
            raise ValueError(f"Unexpected case-id config entry type: {type(entry).__name__}")

    seen = set()
    deduped: List[str] = []

    for case_id in case_ids:
        if case_id not in seen:
            deduped.append(case_id)
            seen.add(case_id)

    return deduped


def load_case_ids_from_config(path: Path) -> List[str]:
    config = read_json(path)

    if isinstance(config, list):
        return normalize_case_id_entries(config)

    if not isinstance(config, dict):
        raise ValueError(f"Expected case config to be a list or object, got {type(config).__name__}")

    list_keys = (
        "case_ids",
        "smoke_case_ids",
        "selected_case_ids",
        "ids",
        "cases",
        "smoke_cases",
        "selected_cases",
        "samples",
    )

    for key in list_keys:
        if key in config:
            value = config[key]
            if not isinstance(value, list):
                raise ValueError(f"Config field {key!r} must be a list.")
            return normalize_case_id_entries(value)

    raise ValueError(
        "Could not find case IDs in config. Expected one of: "
        "case_ids, smoke_case_ids, selected_case_ids, ids, cases, smoke_cases, "
        "selected_cases, samples."
    )


def iter_dataset_records(dataset_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(dataset_obj, list):
        records = dataset_obj
    elif isinstance(dataset_obj, dict):
        if isinstance(dataset_obj.get("cases"), list):
            records = dataset_obj["cases"]
        elif isinstance(dataset_obj.get("test"), list):
            records = dataset_obj["test"]
        elif isinstance(dataset_obj.get("data"), list):
            records = dataset_obj["data"]
        else:
            raise ValueError(
                "Dataset JSON object must contain a list under 'cases', 'test', or 'data', "
                "or be a top-level list."
            )
    else:
        raise ValueError(f"Expected dataset JSON to be list or object, got {type(dataset_obj).__name__}")

    return [record for record in records if isinstance(record, dict)]


def load_gold_and_challenges(dataset_path: Path, case_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    dataset_obj = read_json(dataset_path)
    records = iter_dataset_records(dataset_obj)
    wanted = set(case_ids)

    by_case: Dict[str, Dict[str, Any]] = {}

    for record in records:
        case_id = record.get("id") or record.get("case_id")
        if case_id not in wanted:
            continue

        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        gold = metadata.get("gold")
        if not isinstance(gold, dict):
            gold = record.get("gold")

        if not isinstance(gold, dict):
            raise ValueError(f"Case {case_id!r} is missing metadata.gold.")

        gold_safe = normalize_bool(get_first_present(gold, SAFE_GOLD_KEYS))
        gold_los = normalize_los(get_first_present(gold, LOS_GOLD_KEYS))

        if gold_safe is None:
            raise ValueError(f"Case {case_id!r} is missing valid gold safe_for_lloc.")

        if gold_los is None:
            raise ValueError(f"Case {case_id!r} is missing valid gold expected_los_recommendation.")

        challenge_value = get_first_present(metadata, CHALLENGE_KEYS)
        if challenge_value is None:
            challenge_value = get_first_present(record, CHALLENGE_KEYS)

        by_case[case_id] = {
            "safe_for_lloc": gold_safe,
            "expected_los_recommendation": gold_los,
            "documentation_challenge": normalize_challenge(challenge_value),
        }

    missing = [case_id for case_id in case_ids if case_id not in by_case]
    if missing:
        raise ValueError(f"Selected case IDs missing from dataset/gold: {missing}")

    return by_case


def load_jsonl_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Raw output JSONL not found: {path}")

    rows: List[Dict[str, Any]] = []
    bad_lines: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                bad_lines.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                    }
                )
                continue

            if isinstance(row, dict):
                row["_jsonl_line_number"] = line_number
                rows.append(row)
            else:
                bad_lines.append(
                    {
                        "line_number": line_number,
                        "error": f"JSONL row is {type(row).__name__}, expected object.",
                    }
                )

    return rows, bad_lines


def index_raw_rows_by_case(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    by_case: Dict[str, Dict[str, Any]] = {}
    counts: Counter[str] = Counter()

    for row in rows:
        case_id = row.get("case_id") or row.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            continue

        case_id = case_id.strip()
        counts[case_id] += 1
        by_case[case_id] = row

    duplicate_counts = {case_id: n for case_id, n in counts.items() if n > 1}
    return by_case, duplicate_counts


def get_raw_output_text(row: Dict[str, Any]) -> Optional[str]:
    for field in MODEL_OUTPUT_TEXT_FIELDS:
        if field not in row:
            continue

        value = row[field]

        if isinstance(value, str):
            return value

        if value is not None:
            return json.dumps(value, ensure_ascii=False)

    return None


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)

    if fence_match:
        return fence_match.group(1).strip()

    return stripped


def first_balanced_json_substring(text: str) -> Optional[str]:
    start_positions = [idx for idx, ch in enumerate(text) if ch in "{["]

    matching = {
        "{": "}",
        "[": "]",
    }

    for start in start_positions:
        opening = text[start]
        closing = matching[opening]
        stack = [closing]
        in_string = False
        escape = False

        for idx in range(start + 1, len(text)):
            ch = text[idx]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in matching:
                stack.append(matching[ch])
            elif stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start : idx + 1]
            elif ch in "}]" and (not stack or ch != stack[-1]):
                break

    return None


def parse_model_json(raw_text: Optional[str]) -> Tuple[bool, Optional[Any], Optional[str]]:
    if raw_text is None:
        return False, None, "No raw model output text field found."

    text = raw_text.strip()
    if not text:
        return False, None, "Raw model output is empty."

    candidates: List[str] = [text]

    unfenced = strip_markdown_code_fence(text)
    if unfenced != text:
        candidates.append(unfenced)

    balanced = first_balanced_json_substring(unfenced)
    if balanced is not None and balanced not in candidates:
        candidates.append(balanced)

    last_error: Optional[str] = None

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return True, parsed, None
        except json.JSONDecodeError as exc:
            last_error = str(exc)

    return False, None, last_error or "Could not parse model output as JSON."


def make_validator(schema: Dict[str, Any]) -> Any:
    validator_class = validators.validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema)


def format_schema_error(error: jsonschema.ValidationError) -> Dict[str, Any]:
    path = "$"

    if error.absolute_path:
        path += "".join(
            f"[{part!r}]" if isinstance(part, int) else f".{part}"
            for part in error.absolute_path
        )

    return {
        "path": path,
        "message": error.message,
        "validator": error.validator,
    }


def validate_against_schema(parsed: Any, validator: Any, max_errors: int = 10) -> Tuple[bool, List[Dict[str, Any]]]:
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.absolute_path))

    if not errors:
        return True, []

    return False, [format_schema_error(error) for error in errors[:max_errors]]


def extract_model_labels(parsed: Any) -> Tuple[Optional[bool], Optional[int]]:
    if not isinstance(parsed, dict):
        return None, None

    model_safe = normalize_bool(parsed.get("safe_for_lloc"))
    model_los = normalize_los(parsed.get("expected_los_recommendation"))

    return model_safe, model_los


def infer_single_value(rows: Sequence[Dict[str, Any]], key: str, fallback: Optional[str] = None) -> str:
    values = sorted(
        {
            row.get(key)
            for row in rows
            if isinstance(row.get(key), str) and row.get(key).strip()
        }
    )

    if len(values) == 1:
        return values[0]

    if len(values) > 1:
        raise ValueError(f"Raw output file contains multiple {key} values: {values}")

    return fallback or "unknown"


def source_folder_name(source: str) -> str:
    lowered = source.strip().lower()

    if lowered in {"huggingface", "hf"}:
        return "hf"

    return safe_slug(source)


def build_case_score_record(
    case_id: str,
    raw_row: Optional[Dict[str, Any]],
    gold: Dict[str, Any],
    validator: Any,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "case_id": case_id,
        "retrieval_status": None,
        "valid_json": False,
        "schema_valid": False,
        "parse_error": None,
        "schema_errors": [],
        "parsed_output": None,
        "model_safe_for_lloc": None,
        "model_expected_los_recommendation": None,
        "safe_for_lloc_correct": False,
        "expected_los_correct": False,
        "documentation_challenge": gold["documentation_challenge"],
        "failure_type": None,
    }

    if raw_row is None:
        base.update(
            {
                "retrieval_status": "missing_raw_output",
                "failure_type": "missing_raw_output",
            }
        )
        return base

    retrieval_status = raw_row.get("status")
    base["retrieval_status"] = retrieval_status or "ok"

    if retrieval_status not in OK_STATUSES:
        base["failure_type"] = "retrieval_error"

        if raw_row.get("error") is not None:
            base["parse_error"] = json.dumps(raw_row.get("error"), ensure_ascii=False)

        return base

    raw_text = get_raw_output_text(raw_row)
    valid_json, parsed, parse_error = parse_model_json(raw_text)

    base["valid_json"] = valid_json
    base["parse_error"] = parse_error

    if not valid_json:
        base["failure_type"] = "json_parse_error"
        return base

    base["parsed_output"] = parsed

    schema_valid, schema_errors = validate_against_schema(parsed, validator)

    base["schema_valid"] = schema_valid
    base["schema_errors"] = schema_errors

    if not schema_valid:
        base["failure_type"] = "schema_validation_error"
        return base

    model_safe, model_los = extract_model_labels(parsed)

    base["model_safe_for_lloc"] = model_safe
    base["model_expected_los_recommendation"] = model_los
    base["safe_for_lloc_correct"] = model_safe == gold["safe_for_lloc"]
    base["expected_los_correct"] = model_los == gold["expected_los_recommendation"]
    base["failure_type"] = None

    return base


def summarize_parse_schema(
    records: Sequence[Dict[str, Any]],
    raw_rows: Sequence[Dict[str, Any]],
    bad_jsonl_lines: Sequence[Dict[str, Any]],
    duplicate_counts: Dict[str, int],
) -> Dict[str, Any]:
    n_cases = len(records)
    n_retrieval_ok = sum(1 for r in records if r["retrieval_status"] in {"ok", "success", "completed"})
    n_valid_json = sum(1 for r in records if r["valid_json"])
    n_schema_valid = sum(1 for r in records if r["schema_valid"])

    failure_counts = Counter(r["failure_type"] or "none" for r in records)

    return {
        "n_cases": n_cases,
        "n_raw_jsonl_rows": len(raw_rows),
        "n_bad_jsonl_lines": len(bad_jsonl_lines),
        "n_duplicate_case_ids_in_raw_jsonl": len(duplicate_counts),
        "duplicate_case_id_counts": duplicate_counts,
        "n_retrieval_ok": n_retrieval_ok,
        "n_valid_json": n_valid_json,
        "n_schema_valid": n_schema_valid,
        "retrieval_ok_rate": rate(n_retrieval_ok, n_cases),
        "valid_json_rate": rate(n_valid_json, n_cases),
        "schema_valid_rate": rate(n_schema_valid, n_cases),
        "failure_counts_by_type": dict(sorted(failure_counts.items())),
        "bad_jsonl_lines": list(bad_jsonl_lines),
    }


def summarize_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n_cases = len(records)
    n_schema_valid = sum(1 for r in records if r["schema_valid"])

    safe_correct = sum(1 for r in records if r["safe_for_lloc_correct"])
    los_correct = sum(1 for r in records if r["expected_los_correct"])

    schema_valid_records = [r for r in records if r["schema_valid"]]
    safe_correct_valid_only = sum(1 for r in schema_valid_records if r["safe_for_lloc_correct"])
    los_correct_valid_only = sum(1 for r in schema_valid_records if r["expected_los_correct"])

    return {
        "n_cases": n_cases,
        "n_schema_valid": n_schema_valid,
        "denominator_note": (
            "Primary metrics use all expected cases; retrieval, JSON, and schema failures count as incorrect."
        ),
        "safe_for_lloc_accuracy": rate(safe_correct, n_cases),
        "expected_los_exact_match": rate(los_correct, n_cases),
        "safe_for_lloc_accuracy_schema_valid_only": rate(safe_correct_valid_only, n_schema_valid),
        "expected_los_exact_match_schema_valid_only": rate(los_correct_valid_only, n_schema_valid),
    }


def group_records_by_challenge(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in records:
        grouped[record["documentation_challenge"]].append(record)

    return dict(grouped)


def strip_private_fields_for_parsed_output(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": record["case_id"],
        "retrieval_status": record["retrieval_status"],
        "valid_json": record["valid_json"],
        "schema_valid": record["schema_valid"],
        "parse_error": record["parse_error"],
        "schema_errors": record["schema_errors"],
        "parsed_output": record["parsed_output"],
        "model_safe_for_lloc": record["model_safe_for_lloc"],
        "model_expected_los_recommendation": record["model_expected_los_recommendation"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score raw ClinAuthBench model outputs against local hidden gold labels."
    )

    parser.add_argument(
        "--raw-output-path",
        type=Path,
        required=True,
        help="Raw model-output JSONL produced by retrieval/import.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/release/synthetic_bh_cases_v1_mdp_180.json"),
        help="ClinAuthBench v1 release JSON containing local gold labels.",
    )
    parser.add_argument(
        "--case-config",
        type=Path,
        default=Path("evals/config/v1_smoke_cases.json"),
        help="JSON config containing selected case IDs to score.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("evals/schema/baseline_output_schema.json"),
        help="Output JSON schema used for validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for scored outputs. Default derives from source/model ID.",
    )
    parser.add_argument(
        "--case-set-name",
        default="v1_smoke_cases",
        help="Case-set name recorded in metrics files.",
    )
    parser.add_argument(
        "--run-name",
        default="v1_smoke",
        help="Filename prefix for written result files.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional model ID override if the raw JSONL does not contain model_id.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source override if the raw JSONL does not contain source.",
    )
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=12,
        help="Expected number of selected cases. Use 0 to disable this check.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    case_ids = load_case_ids_from_config(args.case_config)

    if args.expected_case_count and len(case_ids) != args.expected_case_count:
        raise SystemExit(
            f"Expected {args.expected_case_count} case IDs, found {len(case_ids)} in {args.case_config}"
        )

    schema = read_json(args.schema_path)

    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be a JSON object: {args.schema_path}")

    validator = make_validator(schema)

    gold_by_case = load_gold_and_challenges(args.dataset_path, case_ids)

    raw_rows, bad_jsonl_lines = load_jsonl_rows(args.raw_output_path)
    raw_by_case, duplicate_counts = index_raw_rows_by_case(raw_rows)

    extra_case_ids = sorted(set(raw_by_case) - set(case_ids))
    missing_case_ids = sorted(set(case_ids) - set(raw_by_case))

    model_id = args.model_id or infer_single_value(raw_rows, "model_id", fallback="unknown_model")
    source = args.source or infer_single_value(raw_rows, "source", fallback="unknown")

    if args.output_dir is None:
        args.output_dir = (
            Path("evals/model_outputs/scored")
            / source_folder_name(source)
            / safe_slug(model_id)
        )

    scored_records: List[Dict[str, Any]] = []

    for case_id in case_ids:
        scored_records.append(
            build_case_score_record(
                case_id=case_id,
                raw_row=raw_by_case.get(case_id),
                gold=gold_by_case[case_id],
                validator=validator,
            )
        )

    parse_schema_summary = summarize_parse_schema(
        records=scored_records,
        raw_rows=raw_rows,
        bad_jsonl_lines=bad_jsonl_lines,
        duplicate_counts=duplicate_counts,
    )

    parse_schema_summary["missing_case_ids"] = missing_case_ids
    parse_schema_summary["extra_case_ids_in_raw_jsonl"] = extra_case_ids

    overall_metrics = summarize_metrics(scored_records)

    by_challenge = {
        challenge: summarize_metrics(group_records)
        for challenge, group_records in sorted(group_records_by_challenge(scored_records).items())
    }

    metrics_payload = {
        "record_type": "clinauthbench_model_metrics_v1",
        "created_utc": utc_now_iso(),
        "model_id": model_id,
        "source": source,
        "case_set": args.case_set_name,
        "raw_output_path": str(args.raw_output_path),
        "dataset_path": str(args.dataset_path),
        "case_config": str(args.case_config),
        "schema_path": str(args.schema_path),
        "overall": overall_metrics,
        "by_documentation_challenge": by_challenge,
    }

    run_config = {
        "record_type": "clinauthbench_score_run_config_v1",
        "created_utc": utc_now_iso(),
        "model_id": model_id,
        "source": source,
        "case_set": args.case_set_name,
        "run_name": args.run_name,
        "raw_output_path": str(args.raw_output_path),
        "dataset_path": str(args.dataset_path),
        "case_config": str(args.case_config),
        "schema_path": str(args.schema_path),
        "output_dir": str(args.output_dir),
        "expected_case_count": args.expected_case_count,
        "scored_case_ids": case_ids,
        "notes": [
            "Raw outputs were retrieved before this script was run.",
            "This script does not call model APIs.",
            "Gold labels are loaded locally only for scoring.",
            "Parsed-output JSONL excludes per-case gold labels.",
        ],
    }

    parsed_public_records = [
        strip_private_fields_for_parsed_output(record)
        for record in scored_records
    ]

    metrics_path = args.output_dir / f"{args.run_name}_metrics.json"
    parse_schema_path = args.output_dir / f"{args.run_name}_parse_schema_summary.json"
    parsed_outputs_path = args.output_dir / f"{args.run_name}_parsed_outputs.jsonl"
    run_config_path = args.output_dir / f"{args.run_name}_score_run_config.json"

    write_json(metrics_path, metrics_payload)
    write_json(parse_schema_path, parse_schema_summary)
    write_jsonl(parsed_outputs_path, parsed_public_records)
    write_json(run_config_path, run_config)

    print(f"Model: {model_id}")
    print(f"Source: {source}")
    print(f"Cases scored: {len(scored_records)}")
    print(f"Valid JSON: {parse_schema_summary['n_valid_json']}/{parse_schema_summary['n_cases']}")
    print(f"Schema valid: {parse_schema_summary['n_schema_valid']}/{parse_schema_summary['n_cases']}")
    print(f"safe_for_lloc_accuracy: {overall_metrics['safe_for_lloc_accuracy']}")
    print(f"expected_los_exact_match: {overall_metrics['expected_los_exact_match']}")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {parse_schema_path}")
    print(f"Wrote: {parsed_outputs_path}")
    print(f"Wrote: {run_config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
