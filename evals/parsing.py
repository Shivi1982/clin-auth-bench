import json
import re
from pathlib import Path

from jsonschema import validate, ValidationError


SCHEMA_PATH = Path(__file__).parent / "schema" / "baseline_output_schema.json"


def extract_json_text(raw_text: str) -> str:
    raw_text = raw_text.strip()

    # Handles ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Handles normal raw JSON
    return raw_text


def parse_model_response(raw_text: str):
    try:
        json_text = extract_json_text(raw_text)
        parsed = json.loads(json_text)
        return {
            "valid_json": True,
            "parsed": parsed,
            "parse_error": None,
        }
    except Exception as e:
        return {
            "valid_json": False,
            "parsed": None,
            "parse_error": str(e),
        }


def validate_prediction_schema(prediction: dict, schema_path: Path = SCHEMA_PATH):
    schema = json.loads(schema_path.read_text())

    try:
        validate(instance=prediction, schema=schema)
        return {
            "schema_valid": True,
            "schema_error": None,
        }
    except ValidationError as e:
        return {
            "schema_valid": False,
            "schema_error": e.message,
        }


def parse_and_validate(raw_text: str):
    parsed_result = parse_model_response(raw_text)

    if not parsed_result["valid_json"]:
        return {
            **parsed_result,
            "schema_valid": False,
            "schema_error": "Invalid JSON",
        }

    schema_result = validate_prediction_schema(parsed_result["parsed"])

    return {
        **parsed_result,
        **schema_result,
    }
