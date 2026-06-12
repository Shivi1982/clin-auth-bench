#!/usr/bin/env python3
"""
Retrieve raw Hugging Face model outputs for ClinAuthBench v1 smoke cases.

This script intentionally:
- sends only task instructions + output JSON schema + case["content"] to the model
- does not send metadata.gold, evidence_anchors, do_not_claim, documentation_challenge,
  documentation_challenge_tags, or any hidden labels
- does not parse model output
- does not validate model output against the schema
- does not score model output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from huggingface_hub import InferenceClient
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub\n"
        "Install it with: pip install huggingface_hub"
    ) from exc


ALLOWED_HF_MODELS = {
    "openai/gpt-oss-120b",
    "meta-llama/Meta-Llama-3-8B-Instruct",
}

DEFAULT_MAX_TOKENS_BY_MODEL = {
    "openai/gpt-oss-120b": 1600,
    "meta-llama/Meta-Llama-3-8B-Instruct": 512,
}

UNSUPPORTED_HF_CHAT_MODELS = {
    "google/gemma-4-12B-it",
    "google/gemma-4-12B-it-assistant",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
}

HIDDEN_FIELD_MARKERS = (
    "metadata.gold",
    "evidence_anchors",
    "do_not_claim",
    "documentation_challenge",
    "documentation_challenge_tags",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_model_slug(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", model_id.strip())
    return slug.strip("_")


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
            raise ValueError(f"Unexpected case-id config entry type: {type(entry).__name__}")

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
        raise ValueError(f"Expected smoke config to be a list or object, got {type(config).__name__}")

    id_keys = (
        "case_ids",
        "smoke_case_ids",
        "selected_case_ids",
        "ids",
    )
    object_list_keys = (
        "cases",
        "smoke_cases",
        "selected_cases",
        "samples",
    )

    for key in id_keys:
        if key in config:
            value = config[key]
            if not isinstance(value, list):
                raise ValueError(f"Config field {key!r} must be a list.")
            return normalize_case_id_entries(value)

    for key in object_list_keys:
        if key in config:
            value = config[key]
            if not isinstance(value, list):
                raise ValueError(f"Config field {key!r} must be a list.")
            return normalize_case_id_entries(value)

    raise ValueError(
        "Could not find case IDs in smoke config. Expected one of: "
        "case_ids, smoke_case_ids, selected_case_ids, ids, cases, smoke_cases, "
        "selected_cases, samples."
    )


def load_model_facing_cases(dataset_path: Path, wanted_case_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """
    Load only the model-facing fields needed for retrieval.

    The code below intentionally copies only:
    - id
    - content

    It does not access record["metadata"] or any nested gold/challenge fields.
    """
    data = read_json(dataset_path)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if isinstance(data.get("cases"), list):
            records = data["cases"]
        elif isinstance(data.get("test"), list):
            records = data["test"]
        else:
            raise ValueError(
                "Dataset JSON object must contain a list under 'cases' or 'test', "
                "or be a top-level list of case records."
            )
    else:
        raise ValueError(f"Expected dataset JSON to be list or object, got {type(data).__name__}")

    wanted = set(wanted_case_ids)
    found: Dict[str, Dict[str, str]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        case_id = record.get("id")
        if case_id not in wanted:
            continue

        content = record.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Case {case_id!r} is missing non-empty model-facing content.")

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
            "Refusing to build a model prompt that may expose hidden labels."
        )


def build_model_message(task_prompt: str, output_schema_text: str, case_content: str) -> str:
    """
    Build the exact user-facing retrieval message.

    Supports simple placeholders if the prompt file already has them.
    Otherwise appends the schema and case content in fixed sections.
    """
    message = task_prompt.strip()

    schema_replaced = False
    case_replaced = False

    schema_placeholders = (
        "{{OUTPUT_JSON_SCHEMA}}",
        "{{BASELINE_OUTPUT_SCHEMA}}",
        "{{SCHEMA}}",
        "<<OUTPUT_JSON_SCHEMA>>",
        "<<BASELINE_OUTPUT_SCHEMA>>",
        "<<SCHEMA>>",
    )
    case_placeholders = (
        "{{CASE_CONTENT}}",
        "{{CONTENT}}",
        "<<CASE_CONTENT>>",
        "<<CONTENT>>",
    )

    for placeholder in schema_placeholders:
        if placeholder in message:
            message = message.replace(placeholder, output_schema_text.strip())
            schema_replaced = True

    for placeholder in case_placeholders:
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
    """
    Read only this script's wrapper JSONL records so reruns can resume.

    This does not parse or validate the raw model output string.
    """
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok" and isinstance(record.get("case_id"), str):
                completed.add(record["case_id"])

    return completed


def get_chat_content(response: Any) -> Tuple[str, Optional[str], Optional[Any], Optional[str]]:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Hugging Face response did not include choices.")

    choice0 = choices[0]
    finish_reason = getattr(choice0, "finish_reason", None)

    message = getattr(choice0, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if content is None:
        content = ""

    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    usage = getattr(response, "usage", None)
    response_model = getattr(response, "model", None)

    return content, finish_reason, usage, response_model


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


def call_hf_chat_completion(
    client: InferenceClient,
    model_id: str,
    message: str,
    max_tokens: int,
    temperature: float,
) -> Any:
    messages = [
        {
            "role": "user",
            "content": message,
        }
    ]

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

    return client.chat_completion(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )


def retrieve_one_case(
    client: InferenceClient,
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

    base_record: Dict[str, Any] = {
        "record_type": "clinauthbench_hf_raw_model_output_v1",
        "status": None,
        "created_utc": utc_now_iso(),
        "source": "huggingface",
        "model_id": model_id,
        "case_set": args.case_set_name,
        "case_id": case_id,
        "prompt_path": str(args.prompt_path),
        "schema_path": str(args.schema_path),
        "prompt_sha256": sha256_text(task_prompt),
        "schema_sha256": sha256_text(output_schema_text),
        "case_content_sha256": sha256_text(case_content),
        "request": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "raw_output": None,
        "finish_reason": None,
        "response_model": None,
        "usage": None,
        "error": None,
    }

    last_error: Optional[BaseException] = None

    for attempt in range(args.retries + 1):
        try:
            response = call_hf_chat_completion(
                client=client,
                model_id=model_id,
                message=model_message,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            raw_output, finish_reason, usage, response_model = get_chat_content(response)

            base_record.update(
                {
                    "status": "ok",
                    "raw_output": raw_output,
                    "finish_reason": finish_reason,
                    "response_model": response_model,
                    "usage": to_plain_jsonable(usage),
                    "error": None,
                }
            )
            return base_record

        except Exception as exc:  # Keep smoke retrieval moving unless --stop-on-error is used.
            last_error = exc
            if attempt < args.retries:
                sleep_s = min(args.retry_sleep_seconds * (2 ** attempt), args.max_retry_sleep_seconds)
                time.sleep(sleep_s)

    assert last_error is not None
    base_record.update(
        {
            "status": "error",
            "error": {
                "type": type(last_error).__name__,
                "message": str(last_error),
            },
        }
    )
    return base_record


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve raw Hugging Face model outputs for ClinAuthBench v1 smoke cases."
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help=(
            "Hugging Face chat-completions model ID, e.g. openai/gpt-oss-120b, "
            "or meta-llama/Meta-Llama-3-8B-Instruct."
        ),
    )
    parser.add_argument(
        "--case-config",
        type=Path,
        default=Path("evals/config/v1_smoke_cases.json"),
        help="JSON config containing the fixed smoke case IDs.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/release/synthetic_bh_cases_v1_mdp_180.json"),
        help="ClinAuthBench v1 release JSON.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("evals/prompts/v1_zero_shot_blind.md"),
        help="Blind zero-shot prompt file.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("evals/schema/baseline_output_schema.json"),
        help="Output JSON schema file shown to the model.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Default: "
            "evals/model_outputs/raw/hf/<model_slug>/v1_smoke_outputs.jsonl"
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional local env file to load before reading HF_TOKEN. Default: .env",
    )
    parser.add_argument(
        "--case-set-name",
        default="v1_smoke_cases",
        help="Name recorded in each JSONL row.",
    )
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=12,
        help="Expected number of case IDs in the smoke config. Use 0 to disable this check.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum output tokens requested from the model. "
            "Default: model-specific safe value."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for retrieval.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Client timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per case after the first failed attempt.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=5.0,
        help="Initial retry sleep in seconds.",
    )
    parser.add_argument(
        "--max-retry-sleep-seconds",
        type=float,
        default=30.0,
        help="Maximum retry sleep in seconds.",
    )
    parser.add_argument(
        "--sleep-between-cases-seconds",
        type=float,
        default=0.0,
        help="Optional pause between successful case requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output JSONL instead of resuming.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Exit immediately if a case retrieval fails.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_id = args.model_id.strip()
    if model_id in UNSUPPORTED_HF_CHAT_MODELS:
        raise SystemExit(
            f"Hugging Face router reports {model_id} is not a chat model for "
            "/v1/chat/completions. Use a chat-completions-compatible model for this "
            "HF retrieval script, or run this model family through a compatible hosted endpoint "
            "or local inference path under the same prompt/schema contract."
        )

    if model_id not in ALLOWED_HF_MODELS:
        allowed = ", ".join(sorted(ALLOWED_HF_MODELS))
        raise SystemExit(
            f"Refusing unexpected model ID: {model_id}\n"
            f"Allowed for this HF smoke retrieval script: {allowed}"
        )

    if args.max_tokens is None:
        args.max_tokens = DEFAULT_MAX_TOKENS_BY_MODEL[model_id]

    if args.output_path is None:
        args.output_path = (
            Path("evals/model_outputs/raw/hf")
            / safe_model_slug(model_id)
            / "v1_smoke_outputs.jsonl"
        )

    case_ids = load_smoke_case_ids(args.case_config)

    if args.expected_case_count and len(case_ids) != args.expected_case_count:
        raise SystemExit(
            f"Expected {args.expected_case_count} case IDs, found {len(case_ids)} in {args.case_config}"
        )

    task_prompt = read_text(args.prompt_path)
    output_schema_text = read_text(args.schema_path)

    assert_no_hidden_markers(task_prompt, "Prompt file")
    assert_no_hidden_markers(output_schema_text, "Schema file")

    cases = load_model_facing_cases(args.dataset_path, case_ids)

    if args.overwrite and args.output_path.exists():
        args.output_path.unlink()

    completed = existing_success_case_ids(args.output_path)

    loaded_env_keys = load_env_file(args.env_file)
    if loaded_env_keys:
        print(f"Loaded environment variable(s) from {args.env_file}: {', '.join(sorted(loaded_env_keys))}")

    token = os.environ.get("HF_TOKEN")
    if token:
        client = InferenceClient(api_key=token, timeout=args.timeout_seconds)
    else:
        client = InferenceClient(timeout=args.timeout_seconds)
        print(
            "HF_TOKEN is not set. Proceeding with local Hugging Face auth if available.",
            file=sys.stderr,
        )

    print(f"Model: {model_id}")
    print(f"Case config: {args.case_config}")
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
