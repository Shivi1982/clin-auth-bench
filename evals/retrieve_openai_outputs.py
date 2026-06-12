#!/usr/bin/env python3
"""
Retrieve raw OpenAI model outputs for ClinAuthBench v1 smoke cases.

This script:
- reads the fixed smoke case IDs
- loads only case["content"]
- applies the existing blind prompt and output schema
- calls one OpenAI model
- saves raw model outputs to JSONL

It does NOT:
- parse model output
- validate schema
- score outputs
- read metadata.gold or hidden labels
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Missing dependency: openai\nInstall it with: pip install openai") from exc


HIDDEN_FIELD_MARKERS = (
    "metadata.gold",
    "evidence_anchors",
    "do_not_claim",
    "documentation_challenge",
    "documentation_challenge_tags",
)


def parse_version_tuple(value: str) -> Tuple[int, ...]:
    parts = re.split(r"[.\-+]", value)
    numbers: List[int] = []

    for part in parts:
        if not part.isdigit():
            break
        numbers.append(int(part))

    return tuple(numbers)


def check_openai_sdk_compatibility() -> None:
    try:
        openai_version = importlib.metadata.version("openai")
        httpx_version = importlib.metadata.version("httpx")
    except importlib.metadata.PackageNotFoundError:
        return

    openai_tuple = parse_version_tuple(openai_version)
    httpx_tuple = parse_version_tuple(httpx_version)

    if openai_tuple < (1, 68, 0):
        raise SystemExit(
            "Installed OpenAI SDK is too old for this Responses API retrieval script: "
            f"openai=={openai_version}. Run:\n"
            "  python -m pip install --upgrade 'openai>=1.68.0'\n"
        )

    if openai_tuple < (1, 55, 3) and httpx_tuple >= (0, 28, 0):
        raise SystemExit(
            "Installed OpenAI/httpx versions are incompatible: "
            f"openai=={openai_version}, httpx=={httpx_version}. Run:\n"
            "  python -m pip install --upgrade 'openai>=1.68.0'\n"
            "or temporarily downgrade httpx:\n"
            "  python -m pip install 'httpx<0.28'\n"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id.strip()).strip("_")


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_env_file(path: Path, override: bool = False) -> List[str]:
    """
    Load KEY=VALUE lines from a local .env file into os.environ.

    This intentionally supports only simple dotenv syntax so the retrieval
    script does not need an additional dependency.
    """
    if not path.exists():
        return []

    loaded: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export ") :].strip()

            if "=" not in line:
                raise ValueError(f"Invalid .env line {line_number} in {path}: expected KEY=VALUE")

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                raise ValueError(f"Invalid .env line {line_number} in {path}: empty key")

            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {'"', "'"}
            ):
                value = value[1:-1]

            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)

    return loaded


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
            raise ValueError(f"Unexpected case config entry type: {type(entry).__name__}")

    seen = set()
    deduped: List[str] = []

    for case_id in case_ids:
        if case_id not in seen:
            deduped.append(case_id)
            seen.add(case_id)

    return deduped


def load_smoke_case_ids(path: Path) -> List[str]:
    config = read_json(path)

    if isinstance(config, list):
        return normalize_case_id_entries(config)

    if not isinstance(config, dict):
        raise ValueError(f"Expected smoke config list or object, got {type(config).__name__}")

    for key in (
        "case_ids",
        "smoke_case_ids",
        "selected_case_ids",
        "ids",
        "cases",
        "smoke_cases",
        "selected_cases",
        "samples",
    ):
        if key in config:
            value = config[key]
            if not isinstance(value, list):
                raise ValueError(f"Config field {key!r} must be a list.")
            return normalize_case_id_entries(value)

    raise ValueError("Could not find case IDs in smoke config.")


def load_model_facing_cases(dataset_path: Path, wanted_case_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
    data = read_json(dataset_path)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if isinstance(data.get("cases"), list):
            records = data["cases"]
        elif isinstance(data.get("test"), list):
            records = data["test"]
        elif isinstance(data.get("data"), list):
            records = data["data"]
        else:
            raise ValueError("Dataset JSON must contain cases/test/data list or be a top-level list.")
    else:
        raise ValueError(f"Expected dataset JSON list or object, got {type(data).__name__}")

    wanted = set(wanted_case_ids)
    found: Dict[str, Dict[str, str]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        case_id = record.get("id") or record.get("case_id")
        if case_id not in wanted:
            continue

        content = record.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Case {case_id!r} is missing non-empty content.")

        found[case_id] = {
            "id": case_id,
            "content": content,
        }

    missing = [case_id for case_id in wanted_case_ids if case_id not in found]
    if missing:
        raise ValueError(f"Missing selected case IDs in dataset: {missing}")

    return found


def assert_no_hidden_markers(text: str, label: str) -> None:
    lowered = text.lower()
    found = [marker for marker in HIDDEN_FIELD_MARKERS if marker.lower() in lowered]

    if found:
        raise ValueError(
            f"{label} contains hidden-field marker(s): {found}. "
            "Refusing to send hidden-label fields to the model."
        )


def build_model_message(task_prompt: str, output_schema_text: str, case_content: str) -> str:
    message = task_prompt.strip()

    schema_replaced = False
    case_replaced = False

    for placeholder in ("{{OUTPUT_JSON_SCHEMA}}", "{{BASELINE_OUTPUT_SCHEMA}}", "{{SCHEMA}}"):
        if placeholder in message:
            message = message.replace(placeholder, output_schema_text.strip())
            schema_replaced = True

    for placeholder in ("{{CASE_CONTENT}}", "{{CONTENT}}"):
        if placeholder in message:
            message = message.replace(placeholder, case_content.strip())
            case_replaced = True

    if not schema_replaced:
        message += (
            "\n\n---\n"
            "OUTPUT JSON SCHEMA\n"
            "Return only JSON matching this schema. Do not include markdown.\n"
            "```json\n"
            f"{output_schema_text.strip()}\n"
            "```"
        )

    if not case_replaced:
        message += (
            "\n\n---\n"
            "CASE CONTENT\n"
            f"{case_content.strip()}"
        )

    assert_no_hidden_markers(message, "Constructed model message")
    return message


def existing_success_case_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    completed: set[str] = set()

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok" and isinstance(record.get("case_id"), str):
                completed.add(record["case_id"])

    return completed


def to_plain_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): to_plain_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain_jsonable(v) for v in value]

    if hasattr(value, "model_dump"):
        return to_plain_jsonable(value.model_dump())

    if hasattr(value, "dict"):
        return to_plain_jsonable(value.dict())

    return str(value)


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: List[str] = []
    output = getattr(response, "output", None)

    if output:
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue

            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)

    return "\n".join(parts).strip()


def call_openai_response(client: OpenAI, model_id: str, message: str, args: argparse.Namespace) -> Any:
    return client.responses.create(
        model=model_id,
        input=[
            {
                "role": "user",
                "content": message,
            }
        ],
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )


def retrieve_one_case(
    client: OpenAI,
    model_id: str,
    case_id: str,
    case_content: str,
    task_prompt: str,
    output_schema_text: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model_message = build_model_message(
        task_prompt=task_prompt,
        output_schema_text=output_schema_text,
        case_content=case_content,
    )

    record: Dict[str, Any] = {
        "record_type": "clinauthbench_openai_raw_model_output_v1",
        "status": None,
        "created_utc": utc_now_iso(),
        "source": "openai",
        "model_id": model_id,
        "case_set": args.case_set_name,
        "case_id": case_id,
        "prompt_path": str(args.prompt_path),
        "schema_path": str(args.schema_path),
        "prompt_sha256": sha256_text(task_prompt),
        "schema_sha256": sha256_text(output_schema_text),
        "case_content_sha256": sha256_text(case_content),
        "request": {
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
        },
        "response_id": None,
        "response_status": None,
        "raw_output": None,
        "usage": None,
        "error": None,
    }

    last_error: Optional[BaseException] = None

    for attempt in range(args.retries + 1):
        try:
            response = call_openai_response(
                client=client,
                model_id=model_id,
                message=model_message,
                args=args,
            )

            record.update(
                {
                    "status": "ok",
                    "response_id": getattr(response, "id", None),
                    "response_status": getattr(response, "status", None),
                    "raw_output": extract_response_text(response),
                    "usage": to_plain_jsonable(getattr(response, "usage", None)),
                    "error": None,
                }
            )
            return record

        except Exception as exc:
            last_error = exc

            if attempt < args.retries:
                sleep_s = min(args.retry_sleep_seconds * (2 ** attempt), args.max_retry_sleep_seconds)
                time.sleep(sleep_s)

    assert last_error is not None

    record.update(
        {
            "status": "error",
            "error": {
                "type": type(last_error).__name__,
                "message": str(last_error),
            },
        }
    )
    return record


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve raw OpenAI model outputs for ClinAuthBench v1 smoke cases."
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="OpenAI model ID to call, for example gpt-5.4 or the exact available model ID.",
    )
    parser.add_argument(
        "--case-config",
        type=Path,
        default=Path("evals/config/v1_smoke_cases.json"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/release/synthetic_bh_cases_v1_mdp_180.json"),
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("evals/prompts/v1_zero_shot_blind.md"),
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("evals/schema/baseline_output_schema.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Default: evals/model_outputs/raw/openai/<model_slug>/v1_smoke_outputs.jsonl",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional local env file to load before reading OPENAI_API_KEY. Default: .env",
    )
    parser.add_argument(
        "--case-set-name",
        default="v1_smoke_cases",
    )
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2500,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--max-retry-sleep-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--sleep-between-cases-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    check_openai_sdk_compatibility()

    loaded_env_keys = load_env_file(args.env_file)
    if loaded_env_keys:
        print(f"Loaded environment variable(s) from {args.env_file}: {', '.join(sorted(loaded_env_keys))}")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add OPENAI_API_KEY=sk-... to .env "
            "or export it in your shell."
        )

    model_id = args.model_id.strip()

    if args.output_path is None:
        args.output_path = (
            Path("evals/model_outputs/raw/openai")
            / safe_model_slug(model_id)
            / "v1_smoke_outputs.jsonl"
        )

    case_ids = load_smoke_case_ids(args.case_config)

    if args.expected_case_count and len(case_ids) != args.expected_case_count:
        raise SystemExit(
            f"Expected {args.expected_case_count} case IDs, found {len(case_ids)}"
        )

    task_prompt = read_text(args.prompt_path)
    output_schema_text = read_text(args.schema_path)

    assert_no_hidden_markers(task_prompt, "Prompt file")
    assert_no_hidden_markers(output_schema_text, "Schema file")

    cases = load_model_facing_cases(args.dataset_path, case_ids)

    if args.overwrite and args.output_path.exists():
        args.output_path.unlink()

    completed = existing_success_case_ids(args.output_path)

    client = OpenAI(timeout=args.timeout_seconds)

    print(f"Model: {model_id}")
    print(f"Selected cases: {len(case_ids)}")
    print(f"Output JSONL: {args.output_path}")

    for idx, case_id in enumerate(case_ids, start=1):
        if case_id in completed:
            print(f"[{idx}/{len(case_ids)}] SKIP existing ok: {case_id}")
            continue

        print(f"[{idx}/{len(case_ids)}] Retrieving: {case_id}")

        record = retrieve_one_case(
            client=client,
            model_id=model_id,
            case_id=case_id,
            case_content=cases[case_id]["content"],
            task_prompt=task_prompt,
            output_schema_text=output_schema_text,
            args=args,
        )

        append_jsonl(args.output_path, record)

        if record["status"] == "error":
            print(f"[{idx}/{len(case_ids)}] ERROR: {case_id} :: {record['error']}", file=sys.stderr)

            if args.stop_on_error:
                return 1

        else:
            print(f"[{idx}/{len(case_ids)}] OK: {case_id}")

        if args.sleep_between_cases_seconds > 0:
            time.sleep(args.sleep_between_cases_seconds)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
