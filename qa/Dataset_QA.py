"""
ClinAuthBench — Pre-Publication Quality Review Runner
======================================================
Runs each case in isolation through a structured QA prompt.
Outputs a per-case markdown report + a summary roll-up.

Usage (from VSCode terminal):
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."
  python clin_auth_bench/qa/Dataset_QA.py --input clin_auth_bench/data/release/synthetic_bh_cases_v1_mdp_180.json
"""

import json
import argparse
import time
import os
from datetime import datetime
from pathlib import Path

import anthropic

BENCH_DIR = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You are a clinical informatics quality reviewer, not a clinician.
Your job is DATASET QUALITY ASSURANCE before publication — not clinical evaluation or
authorization decisions.

For each case you receive, produce a structured QA report that covers exactly the
sections listed in the user prompt. Be precise and evidence-based: quote the exact
form name and a short phrase from the form when citing evidence. Never invent or
infer information that is not explicitly present in the form content.

PHI/PII rule: Flag any value that looks like real patient data — real names,
real phone numbers, real addresses, real SSNs, real dates of birth, real MRNs.
Synthetic placeholders (SYNTHETIC_CLINICIAN, SYNTHETIC_SIGNATURE_ON_FILE, etc.)
are safe and should be noted as such. If a field is blank or marked as not
collected, note "not present" rather than flagging it.

Gold alignment rule: Compare what the forms say against the gold metadata fields.
Flag only meaningful discrepancies — e.g. the form narrative says "no current SI"
but gold says "moderate current risk". Small wording differences are not flags.

Output format: Use the exact markdown structure specified. Keep each section concise.
Bullet points only where the content is genuinely list-like. Do not pad."""

USER_PROMPT_TEMPLATE = """
## Case under review
- ID: {case_id}
- Title: {title}
- Diagnosis category: {diagnosis_category}
- Documentation challenge: {doc_challenge}
- Total forms: {form_count}

---

## Gold labels (for alignment check)
```json
{gold_json}
```

## Do-not-claim rules for this case
{do_not_claim}

---

## Full form content
{content}

---

## QA Report required — produce exactly this structure

### 1. Admission summary
In 3–4 sentences: what is the patient admitted for, what are the key precipitating
events, and what is the stated anticipated discharge plan.

### 2. Actions taken during stay
List the clinical actions documented (medication restarts, safety planning,
group attendance, discharge coordination, collateral contacts). Cite the form name
for each action.

### 3. Active symptoms identified
For each domain below, state what the forms document. If nothing is documented
for a domain, write "not documented":
- Mood
- Anxiety
- Psychosis / hallucinations
- Suicidal ideation (current, not historical)
- Behavioral disturbance
- Substance use / withdrawal

### 4. Sleep patterns
Summarise all sleep-related data across all forms. Note any inconsistencies
between shifts (e.g. one shift says 3–4 hrs, another says 6 hrs).

### 5. Activities of daily living (ADLs)
Summarise ADL status across shifts. Note any meaningful variation between forms.

### 6. Medication and treatment response
For each medication documented, state: medication name, adherence noted, and
response/side effects documented. Flag any form that documents a response
inconsistent with another form.

### 7. Contradictions and inconsistencies
List every meaningful contradiction found between forms. For each one:
- Form A (name + date): says [X]
- Form B (name + date): says [Y]
- Severity: minor | moderate | significant
If none found, write "No contradictions identified."

### 8. PHI / PII scan
For each form, state one of:
- SAFE — all identifiers are synthetic or absent
- FLAG — [form name]: [specific field] contains [describe concern]
If no flags across all forms, write a single line: "All {form_count} forms: SAFE"

### 9. Gold label alignment
For each gold field below, check whether the form content supports it:
- current_suicide_risk: [gold value] → [supported / partially supported / contradicted — cite form]
- primary_diagnosis: [gold value] → [supported / not mentioned in forms]
- safe_for_lloc: [gold value] → [supported / contradicted — cite form]
- lower_level_of_care_barriers: [gold list] → [each barrier: found in forms / not found]
- do_not_claim violations: check whether any form content accidentally violates
  the do_not_claim rules listed above. State NONE or describe each violation.

### 10. Overall QA verdict
One of: PASS | PASS WITH NOTES | FAIL
Followed by a single sentence explaining the verdict.
If FAIL or PASS WITH NOTES, list the specific issues that need fixing before publication.
"""


def load_cases(filepath: str) -> list:
    with open(filepath, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("Expected a JSON array at the top level.")


def build_user_prompt(case: dict) -> str:
    meta = case.get("metadata", {})
    gold = meta.get("gold", {})
    do_not_claim = meta.get("gold", {}).get("do_not_claim", [])

    return USER_PROMPT_TEMPLATE.format(
        case_id=case.get("id", "unknown"),
        title=case.get("title", ""),
        diagnosis_category=meta.get("diagnosis_category", ""),
        doc_challenge=meta.get("documentation_challenge", ""),
        form_count=meta.get("quality_checks", {}).get("form_count", "unknown"),
        gold_json=json.dumps(gold, indent=2),
        do_not_claim="\n".join(f"- {r}" for r in do_not_claim),
        content=case.get("content", ""),
    )


def review_case(client: anthropic.Anthropic, case: dict, case_num: int, total: int) -> dict:
    case_id = case.get("id", f"case_{case_num}")
    print(f"  [{case_num}/{total}] Reviewing {case_id}...", end=" ", flush=True)

    user_prompt = build_user_prompt(case)

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    review_text = message.content[0].text
    print("done")

    verdict = "UNKNOWN"
    for line in review_text.splitlines():
        if line.strip().startswith("PASS") or line.strip().startswith("FAIL"):
            verdict = line.strip().split()[0]
            break

    return {
        "case_id": case_id,
        "title": case.get("title", ""),
        "diagnosis_category": case.get("metadata", {}).get("diagnosis_category", ""),
        "doc_challenge": case.get("metadata", {}).get("documentation_challenge", ""),
        "verdict": verdict,
        "review": review_text,
    }


def write_case_report(result: dict, output_dir: Path):
    filename = output_dir / f"{result['case_id']}_qa.md"
    with open(filename, "w") as f:
        f.write(f"# QA Report — {result['case_id']}\n")
        f.write(f"**Title:** {result['title']}  \n")
        f.write(f"**Diagnosis:** {result['diagnosis_category']}  \n")
        f.write(f"**Challenge:** {result['doc_challenge']}  \n")
        f.write(f"**Verdict:** `{result['verdict']}`  \n\n")
        f.write("---\n\n")
        f.write(result["review"])
    return filename


def write_summary_report(results: list, output_dir: Path, run_ts: str):
    summary_path = output_dir / "QA_SUMMARY.md"

    pass_count = sum(1 for r in results if r["verdict"].startswith("PASS") and "NOTES" not in r["verdict"])
    notes_count = sum(1 for r in results if "NOTES" in r["verdict"])
    fail_count = sum(1 for r in results if r["verdict"] == "FAIL")

    with open(summary_path, "w") as f:
        f.write(f"# ClinAuthBench QA Summary — {run_ts}\n\n")
        f.write(f"| | |\n|---|---|\n")
        f.write(f"| Total cases reviewed | {len(results)} |\n")
        f.write(f"| PASS | {pass_count} |\n")
        f.write(f"| PASS WITH NOTES | {notes_count} |\n")
        f.write(f"| FAIL | {fail_count} |\n\n")

        f.write("---\n\n## Case-by-case verdicts\n\n")
        f.write("| Case ID | Title | Verdict |\n")
        f.write("|---|---|---|\n")
        for r in results:
            icon = "✅" if r["verdict"] == "PASS" else ("⚠️" if "NOTES" in r["verdict"] else "❌")
            f.write(f"| {r['case_id']} | {r['title']} | {icon} {r['verdict']} |\n")

        if fail_count > 0 or notes_count > 0:
            f.write("\n---\n\n## Cases requiring attention\n\n")
            for r in results:
                if r["verdict"] != "PASS":
                    f.write(f"### {r['case_id']} — {r['verdict']}\n")
                    lines = r["review"].splitlines()
                    in_verdict = False
                    for line in lines:
                        if "Overall QA verdict" in line:
                            in_verdict = True
                        if in_verdict:
                            f.write(line + "\n")
                        if in_verdict and line.strip() == "" and len(line) == 0:
                            break
                    f.write("\n")

    return summary_path


def main():
    parser = argparse.ArgumentParser(description="ClinAuthBench QA Review Runner")
    parser.add_argument("--input", required=True, help="Path to the JSON dataset file")
    parser.add_argument("--output", default=str(BENCH_DIR / "outputs" / "qa_reports"), help="Output directory for reports")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument("--cases", default=None, help="Comma-separated case IDs to run (default: all)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable before running.")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"\nLoading cases from {args.input}...")
    cases = load_cases(args.input)

    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.get("id") in wanted]
        print(f"Filtered to {len(cases)} case(s): {args.cases}")
    else:
        print(f"Loaded {len(cases)} cases.")

    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = Path(args.output) / run_ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    results = []
    for i, case in enumerate(cases, start=1):
        result = review_case(client, case, i, len(cases))
        results.append(result)
        write_case_report(result, output_dir)
        if i < len(cases):
            time.sleep(args.delay)

    summary_path = write_summary_report(results, output_dir, run_ts)

    print(f"\n{'='*50}")
    print(f"QA complete. {len(results)} cases reviewed.")
    print(f"Summary: {summary_path}")
    print(f"Per-case reports: {output_dir}/")

    pass_c = sum(1 for r in results if r["verdict"] == "PASS")
    notes_c = sum(1 for r in results if "NOTES" in r["verdict"])
    fail_c = sum(1 for r in results if r["verdict"] == "FAIL")
    print(f"\nVerdicts:  PASS={pass_c}  PASS WITH NOTES={notes_c}  FAIL={fail_c}")


if __name__ == "__main__":
    main()
