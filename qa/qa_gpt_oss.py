"""
ClinAuthBench QA runner using deterministic gates plus hosted GPT-OSS review.

The deterministic checks run first and do not require network access:
- schema and form-structure checks
- required clinical marker checks
- regex/sample-term PHI checks
- gold/evidence-anchor structural checks
- optional v1_30 distribution guardrails

The GPT-OSS pass uses Hugging Face's hosted Inference API through HF_TOKEN.
By default it does not pin a third-party inference provider in the model name.
If you intentionally use a provider suffix such as ":fireworks-ai", pass
--allow-third-party-provider.

Usage:
  export HF_TOKEN="hf_..."
  python clin_auth_bench/qa/qa_gpt_oss.py \
    --input clin_auth_bench/data/release/synthetic_bh_cases_v1_mdp_180.json \
    --guardrail-profile none

Deterministic-only smoke test:
  python clin_auth_bench/qa/qa_gpt_oss.py \
    --input clin_auth_bench/data/release/synthetic_bh_cases_v1_mdp_180.json \
    --guardrail-profile none \
    --deterministic-only
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from huggingface_hub import InferenceClient
except ImportError:  # Deterministic-only mode can still run without this package.
    InferenceClient = None


BENCH_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "openai/gpt-oss-120b"

FORM_HEADER = re.compile(
    r"FORM:\s*([^|]+?)\s*\|\s*CREATION_DATE:\s*([^\n]+)",
    re.MULTILINE,
)

REQUIRED_TOP_LEVEL_FIELDS = {"id", "title", "metadata", "content"}
REQUIRED_METADATA_FIELDS = {
    "benchmark",
    "version",
    "synthetic",
    "privacy_design",
    "level_of_care",
    "documentation_window_hours",
    "diagnosis_category",
    "trajectory",
    "documentation_challenge",
    "documentation_challenge_tags",
    "mdp_trajectory",
    "gold",
    "quality_checks",
    "content_gold_checks",
}
REQUIRED_GOLD_FIELDS = {
    "current_suicide_risk",
    "historical_suicide_risk",
    "primary_diagnosis",
    "psychosis_present",
    "substance_use",
    "safe_for_lloc",
    "expected_los_recommendation",
    "lower_level_of_care_barriers",
    "documentation_challenge",
    "documentation_challenge_tags",
    "key_evidence_forms",
    "evidence_anchors",
    "do_not_claim",
    "form_count",
}
REQUIRED_CONTENT_MARKERS = [
    "C-SSRS - Discharge Screener",
    "PHQ-9",
    "GAD-7",
    "Activities of Daily Living:",
    "Medication Compliance:",
    "Hours of sleep/Night:",
    "Mood:",
    "Behavior:",
    "Cognition/Thought Content/Thought Process:",
]

PHI_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "url": re.compile(r"https?://|www\.|aspx\?|pdfid=", re.I),
    "mrn_or_account": re.compile(
        r"\b(?:MRN|Medical Record|Account|Encounter|Claim)\s*[:#]\s*[A-Z0-9-]{4,}\b",
        re.I,
    ),
    "dob": re.compile(
        r"\b(?:DOB|Date of Birth)\s*[:#-]?\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{8})\b",
        re.I,
    ),
    "street_address": re.compile(
        r"\b\d{1,6}\s+[A-Z][A-Za-z0-9.'-]*(?:\s+[A-Za-z0-9.'-]+){0,4}\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct)\b"
    ),
}

FORBIDDEN_SAMPLE_TERMS = {
    "synthetic_source_patient_name",
    "synthetic_source_facility_name",
    "synthetic_source_staff_name",
    "synthetic_source_identifier",
    "aspx?pdfid",
}

GOLD_USER = """Check whether these evidence forms support the gold labels.

GOLD LABELS:
{gold}

DO-NOT-CLAIM RULES:
{do_not_claim}

KEY FORM MATCH WARNINGS:
{key_form_warnings}

KEY EVIDENCE FORMS:
{key_forms}

Return ONLY valid JSON, no markdown:
{{
  "current_suicide_risk": {{"gold":"{gold_sr}","supported":"YES|PARTIAL|NO","note":"<one line>"}},
  "historical_suicide_risk": {{"gold":"{gold_hist}","supported":"YES|PARTIAL|NO","note":"<one line>"}},
  "primary_diagnosis": {{"gold":"{gold_dx}","supported":"YES|PARTIAL|NO","note":"<one line>"}},
  "safe_for_lloc": {{"gold":{gold_lloc},"supported":"YES|PARTIAL|NO","note":"<one line>"}},
  "expected_los_recommendation": {{"gold":"{gold_los}","supported":"YES|PARTIAL|NO","note":"<one line>"}},
  "documentation_challenge": {{"gold":"{gold_challenge}","supported":"YES|PARTIAL|NO","note":"<one line>"}},
  "contradiction_present_when_labeled": {{"supported":"YES|PARTIAL|NO|NOT_APPLICABLE","note":"<one line>"}},
  "rating_score_handling": {{"supported":"YES|PARTIAL|NO","note":"<missing/refused/malformed scores handled without invention?>"}},
  "required_domains": {{
    "cssrs": "PRESENT|MISSING",
    "phq_gad": "PRESENT|MISSING",
    "adls": "PRESENT|MISSING",
    "sleep": "PRESENT|MISSING",
    "medication_compliance": "PRESENT|MISSING",
    "mood_behavior_cognition": "PRESENT|MISSING"
  }},
  "lloc_barriers_missing": ["<barrier in gold not found in forms>"],
  "evidence_anchor_issues": ["<gold evidence anchor not found or not supportive>"],
  "do_not_claim_violations": ["<quote form text breaking a rule, else empty>"],
  "verdict": "PASS|PASS_WITH_NOTES|FAIL",
  "fail_reason": null
}}

Set verdict FAIL if any major field is NO, if do_not_claim_violations exist, or if safe_for_lloc is contradicted.
Use PASS_WITH_NOTES for minor PARTIAL support that needs human review.
"""

GOLD_RESPONSE_SCHEMA = {
    "name": "clin_auth_bench_gold_alignment",
    "schema": {
        "type": "object",
        "properties": {
            "current_suicide_risk": {"type": "object"},
            "historical_suicide_risk": {"type": "object"},
            "primary_diagnosis": {"type": "object"},
            "safe_for_lloc": {"type": "object"},
            "expected_los_recommendation": {"type": "object"},
            "documentation_challenge": {"type": "object"},
            "contradiction_present_when_labeled": {"type": "object"},
            "rating_score_handling": {"type": "object"},
            "required_domains": {"type": "object"},
            "lloc_barriers_missing": {"type": "array", "items": {"type": "string"}},
            "evidence_anchor_issues": {"type": "array", "items": {"type": "string"}},
            "do_not_claim_violations": {"type": "array", "items": {"type": "string"}},
            "verdict": {"type": "string", "enum": ["PASS", "PASS_WITH_NOTES", "FAIL"]},
            "fail_reason": {"type": ["string", "null"]},
        },
        "required": [
            "current_suicide_risk",
            "historical_suicide_risk",
            "primary_diagnosis",
            "safe_for_lloc",
            "expected_los_recommendation",
            "documentation_challenge",
            "contradiction_present_when_labeled",
            "rating_score_handling",
            "required_domains",
            "lloc_barriers_missing",
            "evidence_anchor_issues",
            "do_not_claim_violations",
            "verdict",
            "fail_reason",
        ],
        "additionalProperties": True,
    },
}


def load_cases(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at the top level.")
    return data


def parse_forms(content: str) -> List[Dict[str, str]]:
    matches = list(FORM_HEADER.finditer(content or ""))
    forms = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        forms.append(
            {
                "name": match.group(1).strip(),
                "creation_date": match.group(2).strip(),
                "text": content[match.start() : end].strip(),
            }
        )
    return forms


def normalize_form_name(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip().lower()


def select_key_forms(content: str, key_form_names: Iterable[str]) -> Tuple[str, List[str], List[str]]:
    forms = parse_forms(content)
    by_name: Dict[str, List[Dict[str, str]]] = {}
    for form in forms:
        by_name.setdefault(normalize_form_name(form["name"]), []).append(form)

    selected = []
    found_names = []
    missing = []
    for name in key_form_names or []:
        matches = by_name.get(normalize_form_name(name), [])
        if matches:
            found_names.append(name)
            selected.extend(form["text"] for form in matches)
        else:
            missing.append(name)

    return "\n\n".join(selected), found_names, missing


def extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in model response: {text[:300]}")

    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError(f"Could not parse JSON object from model response: {text[:300]}")


def scan_phi_text(text: str) -> List[Dict[str, str]]:
    flags = []
    safe_text = text or ""
    for pattern_name, pattern in PHI_PATTERNS.items():
        for match in pattern.finditer(safe_text):
            value = match.group(0)
            if value.upper().startswith("SYNTHETIC_"):
                continue
            flags.append(
                {
                    "type": pattern_name,
                    "value": value,
                    "reason": f"Matched deterministic {pattern_name} PHI/PII pattern",
                }
            )

    lower_text = safe_text.lower()
    for term in sorted(FORBIDDEN_SAMPLE_TERMS):
        if term in lower_text:
            flags.append(
                {
                    "type": "forbidden_sample_term",
                    "value": term,
                    "reason": "Matched term from real/sample documentation that should not appear in synthetic release data",
                }
            )
    return flags


def deterministic_case_checks(case: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    warnings = []
    case_id = case.get("id", "unknown")
    metadata = case.get("metadata", {})
    gold = metadata.get("gold", {})
    content = case.get("content", "")
    forms = parse_forms(content)

    missing_top = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(case))
    if missing_top:
        issues.append(f"missing top-level fields: {missing_top}")

    missing_meta = sorted(REQUIRED_METADATA_FIELDS - set(metadata))
    if missing_meta:
        issues.append(f"{case_id}: missing metadata fields: {missing_meta}")

    missing_gold = sorted(REQUIRED_GOLD_FIELDS - set(gold))
    if missing_gold:
        issues.append(f"{case_id}: missing metadata.gold fields: {missing_gold}")

    if not forms:
        issues.append("no parseable FORM headers found")

    quality_checks = metadata.get("quality_checks", {})
    expected_form_count = quality_checks.get("form_count") or gold.get("form_count")
    if expected_form_count is not None and expected_form_count != len(forms):
        issues.append(
            f"form_count mismatch: metadata says {expected_form_count}, parsed {len(forms)}"
        )

    if quality_checks.get("passed") is not True:
        issues.append("metadata.quality_checks.passed is not true")
    if metadata.get("content_gold_checks", {}).get("passed") is not True:
        issues.append("metadata.content_gold_checks.passed is not true")

    for marker in REQUIRED_CONTENT_MARKERS:
        if marker not in content:
            issues.append(f"missing required content marker: {marker}")

    if gold.get("documentation_challenge") != metadata.get("documentation_challenge"):
        issues.append("metadata and gold documentation_challenge do not match")

    if set(gold.get("documentation_challenge_tags", [])) != set(
        metadata.get("documentation_challenge_tags", [])
    ):
        issues.append("metadata and gold documentation_challenge_tags do not match")

    if not isinstance(gold.get("safe_for_lloc"), bool):
        issues.append("metadata.gold.safe_for_lloc must be boolean")

    if metadata.get("documentation_window_hours") != 72:
        warnings.append("documentation_window_hours is not 72")

    if not gold.get("key_evidence_forms"):
        issues.append("metadata.gold.key_evidence_forms is empty")
    else:
        _, _, missing_key_forms = select_key_forms(content, gold.get("key_evidence_forms", []))
        if missing_key_forms:
            issues.append(f"key_evidence_forms not found in content: {missing_key_forms}")

    for anchor in gold.get("evidence_anchors", []):
        form_name = anchor.get("supporting_form")
        hint = anchor.get("evidence_hint")
        if form_name and normalize_form_name(form_name) not in {
            normalize_form_name(form["name"]) for form in forms
        }:
            issues.append(f"evidence anchor supporting_form not found: {form_name}")
        if hint and hint.lower() not in content.lower():
            warnings.append(f"evidence anchor hint not found as exact text: {hint}")

    phi_flags = []
    for form in forms:
        for flag in scan_phi_text(form["text"]):
            flag["form"] = form["name"]
            phi_flags.append(flag)

    status = "PASS" if not issues and not phi_flags else "FAIL"
    return {
        "case_id": case_id,
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "form_count": len(forms),
        "phi_status": "SAFE" if not phi_flags else "FLAG",
        "phi_flags": phi_flags,
    }


def dataset_guardrail_checks(cases: List[Dict[str, Any]], profile: str) -> Dict[str, Any]:
    if profile == "auto":
        profile = "v1_30" if len(cases) == 30 else "none"
    if profile == "none":
        return {"profile": "none", "status": "SKIPPED", "issues": [], "counts": {}}

    issues = []
    safe_count = sum(
        1 for case in cases if case.get("metadata", {}).get("gold", {}).get("safe_for_lloc") is True
    )
    continued_count = len(cases) - safe_count
    challenge_counts = Counter(
        case.get("metadata", {}).get("documentation_challenge") for case in cases
    )
    diagnosis_counts = Counter(case.get("metadata", {}).get("diagnosis_category") for case in cases)

    if profile == "v1_30":
        if len(cases) != 30:
            issues.append(f"expected 30 cases, got {len(cases)}")
        if continued_count != 18:
            issues.append(f"expected 18 continued-stay cases, got {continued_count}")
        if safe_count != 12:
            issues.append(f"expected 12 safe/LLOC-ready cases, got {safe_count}")
        contradiction_count = challenge_counts.get("contradiction", 0)
        if not 3 <= contradiction_count <= 4:
            issues.append(f"expected 3-4 contradiction cases, got {contradiction_count}")
        if diagnosis_counts.get("Dual diagnosis / OUD-related behavioral health crisis", 0) < 4:
            issues.append("expected at least 4 OUD/dual-diagnosis cases")
        if diagnosis_counts.get("Trauma/anxiety with suicidality", 0) < 4:
            issues.append("expected at least 4 trauma/anxiety cases")

    return {
        "profile": profile,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "counts": {
            "total": len(cases),
            "continued_stay": continued_count,
            "safe_for_lloc": safe_count,
            "documentation_challenges": dict(sorted(challenge_counts.items())),
            "diagnosis_categories": dict(sorted(diagnosis_counts.items())),
        },
    }


def object_to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [object_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [object_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): object_to_plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return object_to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return object_to_plain(value.dict())
    if hasattr(value, "__dict__"):
        return {
            key: object_to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def response_debug_string(response: Any) -> str:
    try:
        return json.dumps(object_to_plain(response), indent=2)[:2500]
    except Exception:
        return repr(response)[:2500]


def hf_response_candidates(response: Any) -> List[str]:
    candidates: List[str] = []
    if isinstance(response, str):
        return [response]

    plain = object_to_plain(response)
    choices = plain.get("choices", []) if isinstance(plain, dict) else []
    if choices:
        message = choices[0].get("message", {}) or {}
        for key in ("content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str) and text.strip():
                            candidates.append(text)
                    elif isinstance(item, str) and item.strip():
                        candidates.append(item)
    for key in ("generated_text", "output_text", "text"):
        value = plain.get(key) if isinstance(plain, dict) else None
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    return candidates


def call_hf_json(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    retries: int = 2,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_debug = ""
    for attempt in range(retries + 1):
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": GOLD_RESPONSE_SCHEMA,
                },
                extra_body={"reasoning_effort": "low"},
            )
            last_debug = response_debug_string(response)
            candidates = hf_response_candidates(response)
            if not candidates:
                raise ValueError("empty assistant content")
            candidate_errors = []
            for candidate in candidates:
                try:
                    return extract_json_object(candidate)
                except Exception as exc:
                    candidate_errors.append(str(exc))
            raise ValueError("; ".join(candidate_errors))
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        "HF model did not return valid JSON: "
        f"{last_error}. Raw response preview: {last_debug or '<no response captured>'}"
    )


def check_gold_with_hf(
    client: Any,
    model: str,
    case: Dict[str, Any],
    max_tokens: int,
    max_key_form_chars: int,
) -> Dict[str, Any]:
    gold = case.get("metadata", {}).get("gold", {})
    key_forms, found_names, missing_names = select_key_forms(
        case.get("content", ""),
        gold.get("key_evidence_forms", []),
    )
    if not key_forms:
        key_forms = case.get("content", "")[:max_key_form_chars]
    else:
        key_forms = key_forms[:max_key_form_chars]

    warnings = []
    if missing_names:
        warnings.append("Missing key evidence forms: " + ", ".join(missing_names))
    if found_names:
        warnings.append("Matched key evidence forms: " + ", ".join(found_names))

    messages = [
        {"role": "system", "content": "Dataset QA reviewer. Return only valid JSON."},
        {
            "role": "user",
            "content": GOLD_USER.format(
                gold=json.dumps(gold, indent=2),
                do_not_claim="\n".join(f"- {rule}" for rule in gold.get("do_not_claim", [])),
                key_form_warnings="\n".join(f"- {warning}" for warning in warnings) or "- none",
                key_forms=key_forms,
                gold_sr=gold.get("current_suicide_risk", ""),
                gold_hist=gold.get("historical_suicide_risk", ""),
                gold_dx=gold.get("primary_diagnosis", ""),
                gold_lloc=str(gold.get("safe_for_lloc", "null")).lower(),
                gold_los=gold.get("expected_los_recommendation", ""),
                gold_challenge=gold.get("documentation_challenge", ""),
            ),
        },
    ]
    result = call_hf_json(client, model, messages, max_tokens=max_tokens)
    result["case_id"] = case.get("id", "unknown")
    result["key_forms_found"] = found_names
    result["key_forms_missing"] = missing_names
    return result


def run_hf_gold_checks(
    cases: List[Dict[str, Any]],
    model: str,
    token: str,
    workers: int,
    max_tokens: int,
    max_key_form_chars: int,
) -> Dict[str, Dict[str, Any]]:
    if InferenceClient is None:
        raise ImportError("Install huggingface_hub to run GPT-OSS QA: pip install huggingface_hub")
    client = InferenceClient(token=token)

    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(check_gold_with_hf, client, model, case, max_tokens, max_key_form_chars): case[
                "id"
            ]
            for case in cases
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                results[case_id] = future.result()
                print(f"  [{idx}/{len(cases)}] {case_id} -> {results[case_id].get('verdict')}")
            except Exception as exc:
                results[case_id] = {"case_id": case_id, "verdict": "ERROR", "error": str(exc)}
                print(f"  [{idx}/{len(cases)}] {case_id} -> ERROR: {exc}")
    return results


def uses_explicit_provider_route(model: str) -> bool:
    """HF model routes may append ':provider-name' after org/model."""
    return ":" in model.rsplit("/", 1)[-1]


def final_case_verdict(deterministic: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> str:
    if deterministic.get("status") == "FAIL" or deterministic.get("phi_status") == "FLAG":
        return "FAIL"
    if not llm_result:
        return "PASS"
    if llm_result.get("verdict") == "PASS":
        return "PASS"
    if llm_result.get("verdict") == "PASS_WITH_NOTES":
        return "REVIEW"
    return "FAIL"


def write_summary(
    cases: List[Dict[str, Any]],
    deterministic_results: Dict[str, Dict[str, Any]],
    guardrails: Dict[str, Any],
    gold_results: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> Path:
    path = out_dir / "summary.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# ClinAuthBench QA - {out_dir.name}\n\n")
        f.write("## Run Mode\n\n")
        if gold_results:
            f.write("- GPT-OSS gold alignment: `RUN`\n")
        else:
            f.write("- GPT-OSS gold alignment: `SKIPPED`\n")
            f.write("- This report contains deterministic schema, form, PHI/PII, and guardrail checks only.\n")
        f.write("\n")

        f.write("## Dataset Guardrails\n\n")
        f.write(f"- Profile: `{guardrails['profile']}`\n")
        f.write(f"- Status: `{guardrails['status']}`\n")
        if guardrails.get("status") == "SKIPPED":
            f.write("- Dataset-level distribution checks were not applied for this run.\n")
        for issue in guardrails.get("issues", []):
            f.write(f"- Issue: {issue}\n")
        f.write("\n```json\n")
        f.write(json.dumps(guardrails.get("counts", {}), indent=2))
        f.write("\n```\n\n")

        f.write("## Case Summary\n\n")
        f.write("| Case ID | Deterministic | PHI/PII | GPT-OSS Gold | Final |\n")
        f.write("|---|---|---|---|---|\n")
        failures = []
        for case in cases:
            case_id = case["id"]
            deterministic = deterministic_results[case_id]
            llm = gold_results.get(case_id)
            llm_cell = llm.get("verdict", "SKIPPED") if llm else "SKIPPED"
            final = final_case_verdict(deterministic, llm)
            if final != "PASS":
                failures.append((case_id, deterministic, llm))
            f.write(
                f"| {case_id} | {deterministic['status']} | {deterministic['phi_status']} | "
                f"{llm_cell} | {final} |\n"
            )

        f.write("\n## Deterministic Details\n\n")
        f.write("| Case ID | Forms Parsed | Issues | Warnings | PHI/PII Flags |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for case in cases:
            case_id = case["id"]
            deterministic = deterministic_results[case_id]
            f.write(
                f"| {case_id} | {deterministic.get('form_count', 0)} | "
                f"{len(deterministic.get('issues', []))} | "
                f"{len(deterministic.get('warnings', []))} | "
                f"{len(deterministic.get('phi_flags', []))} |\n"
            )

        warning_cases = [
            (case["id"], deterministic_results[case["id"]])
            for case in cases
            if deterministic_results[case["id"]].get("warnings")
        ]
        if warning_cases:
            f.write("\n## Deterministic Warnings\n\n")
            for case_id, deterministic in warning_cases:
                f.write(f"### {case_id}\n")
                for warning in deterministic.get("warnings", []):
                    f.write(f"- {warning}\n")
                f.write("\n")

        if failures or guardrails.get("status") == "FAIL":
            f.write("\n## Cases Needing Review\n\n")
            for case_id, deterministic, llm in failures:
                f.write(f"### {case_id}\n")
                for issue in deterministic.get("issues", []):
                    f.write(f"- Deterministic issue: {issue}\n")
                for warning in deterministic.get("warnings", []):
                    f.write(f"- Deterministic warning: {warning}\n")
                for flag in deterministic.get("phi_flags", []):
                    f.write(
                        f"- PHI flag in `{flag.get('form')}`: {flag.get('type')} "
                        f"`{flag.get('value')}` - {flag.get('reason')}\n"
                    )
                if llm:
                    if llm.get("fail_reason"):
                        f.write(f"- GPT-OSS fail reason: {llm['fail_reason']}\n")
                    for field in [
                        "lloc_barriers_missing",
                        "evidence_anchor_issues",
                        "do_not_claim_violations",
                    ]:
                        for item in llm.get(field, []) or []:
                            f.write(f"- GPT-OSS {field}: {item}\n")
                    if llm.get("error"):
                        f.write(f"- GPT-OSS error: {llm['error']}\n")
                f.write("\n")

        final_counts = Counter(
            final_case_verdict(deterministic_results[case["id"]], gold_results.get(case["id"]))
            for case in cases
        )
        f.write("\n## Rollup\n\n")
        f.write(f"- Cases: {len(cases)}\n")
        f.write(f"- Final PASS: {final_counts.get('PASS', 0)}\n")
        f.write(f"- Final REVIEW: {final_counts.get('REVIEW', 0)}\n")
        f.write(f"- Final FAIL: {final_counts.get('FAIL', 0)}\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="ClinAuthBench deterministic + GPT-OSS QA")
    parser.add_argument("--input", required=True, help="Path to ClinAuthBench JSON dataset")
    parser.add_argument(
        "--output",
        default=str(BENCH_DIR / "outputs" / "qa_gpt_oss"),
        help="Directory where QA run folders are written",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument(
        "--allow-third-party-provider",
        action="store_true",
        help=(
            "Allow explicit HF provider routes such as openai/gpt-oss-120b:fireworks-ai. "
            "Without this flag, provider-suffixed model IDs are rejected."
        ),
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel GPT-OSS case calls")
    parser.add_argument("--max-tokens", type=int, default=1800, help="Max tokens for each GPT-OSS JSON response")
    parser.add_argument("--max-key-form-chars", type=int, default=12000)
    parser.add_argument(
        "--guardrail-profile",
        choices=["auto", "v1_30", "none"],
        default="auto",
        help="Dataset-level deterministic distribution guardrails",
    )
    parser.add_argument("--cases", default=None, help="Comma-separated case IDs to QA")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Skip hosted GPT-OSS calls and run deterministic gates only",
    )
    args = parser.parse_args()

    cases = load_cases(args.input)
    if args.cases:
        wanted = {case_id.strip() for case_id in args.cases.split(",") if case_id.strip()}
        cases = [case for case in cases if case.get("id") in wanted]
    print(f"Loaded {len(cases)} cases.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nPass 1: deterministic schema, form, PHI, and guardrail checks")
    deterministic_results = {case["id"]: deterministic_case_checks(case) for case in cases}
    guardrails = dataset_guardrail_checks(cases, args.guardrail_profile)
    deterministic_failures = sum(1 for item in deterministic_results.values() if item["status"] == "FAIL")
    phi_flags = sum(1 for item in deterministic_results.values() if item["phi_status"] == "FLAG")
    print(
        f"  Deterministic failures: {deterministic_failures}; "
        f"PHI flagged cases: {phi_flags}; guardrails: {guardrails['status']}"
    )

    gold_results: Dict[str, Dict[str, Any]] = {}
    if args.deterministic_only:
        print("\nPass 2: GPT-OSS gold alignment skipped (--deterministic-only)")
    else:
        if uses_explicit_provider_route(args.model) and not args.allow_third_party_provider:
            raise ValueError(
                f"Model '{args.model}' uses an explicit third-party provider route. "
                "Remove the provider suffix or pass --allow-third-party-provider after confirming "
                "you are comfortable sending prompts/case text to that provider."
            )
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError("Set HF_TOKEN before running GPT-OSS QA, or pass --deterministic-only.")
        print(f"\nPass 2: hosted GPT-OSS gold alignment ({args.model})")
        gold_results = run_hf_gold_checks(
            cases,
            model=args.model,
            token=token,
            workers=args.workers,
            max_tokens=args.max_tokens,
            max_key_form_chars=args.max_key_form_chars,
        )

    with open(out_dir / "deterministic_results.json", "w", encoding="utf-8") as f:
        json.dump(deterministic_results, f, indent=2)
    with open(out_dir / "dataset_guardrails.json", "w", encoding="utf-8") as f:
        json.dump(guardrails, f, indent=2)
    if gold_results:
        with open(out_dir / "gpt_oss_gold_results.json", "w", encoding="utf-8") as f:
            json.dump(gold_results, f, indent=2)

    summary_path = write_summary(cases, deterministic_results, guardrails, gold_results, out_dir)
    final_counts = Counter(
        final_case_verdict(deterministic_results[case["id"]], gold_results.get(case["id"]))
        for case in cases
    )
    print(
        f"\nDone. PASS={final_counts.get('PASS', 0)} "
        f"REVIEW={final_counts.get('REVIEW', 0)} FAIL={final_counts.get('FAIL', 0)}"
    )
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
