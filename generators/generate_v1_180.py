"""
Generate the ClinAuthBench v1 180-case release candidate and the 171-180 QA batch.

Cases 171-180 continue the explicit probabilistic MDP transition path used in
cases 121-170. The batch closes v1 at the planned 60/40 disposition mix.

Outputs:
- clin_auth_bench/data/release/synthetic_bh_cases_v1_mdp_180.json
- clin_auth_bench/data/review_batches/synthetic_bh_cases_v1_mdp_171_180.json
"""

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from generate_v1_70 import required_content_markers
from generate_v1_170 import build_specs_170, p_case
from mdp_case_builder import build_case


BENCH_DIR = Path(__file__).resolve().parents[1]
CUMULATIVE_OUT_PATH = BENCH_DIR / "data" / "release" / "synthetic_bh_cases_v1_mdp_180.json"
BATCH_OUT_PATH = BENCH_DIR / "data" / "review_batches" / "synthetic_bh_cases_v1_mdp_171_180.json"

EXPECTED_TOTAL_CASES = 180
EXPECTED_BATCH_CASES = 10
EXPECTED_TOTAL_CONTINUED_STAY = 108
EXPECTED_TOTAL_SAFE_FOR_LLOC = 72
EXPECTED_BATCH_CONTINUED_STAY = 6
EXPECTED_BATCH_SAFE_FOR_LLOC = 4
EXPECTED_TOTAL_CONTRADICTION_MIN = 18
EXPECTED_TOTAL_CONTRADICTION_MAX = 27

EXPECTED_DIAGNOSIS_COUNTS = {
    "Bipolar mixed episode": 26,
    "Borderline personality disorder self-harm crisis": 26,
    "Dual diagnosis / OUD-related behavioral health crisis": 25,
    "Major depressive disorder with psychotic features": 26,
    "Schizoaffective disorder with command hallucinations": 26,
    "Substance-induced mood or psychotic symptoms": 26,
    "Trauma/anxiety with suicidality": 25,
}


def specs_171_180():
    base = datetime(2027, 2, 1)
    return [
        p_case(171, "Final schizoaffective negative screener conflict", "schizoaffective", "contradiction", "final_schizo_negative_screener_conflict", base, False, "3 days", ["psychiatry note says not ready", "safety plan incomplete", "collateral monitoring not confirmed"], "moderate current risk because internal preoccupation and incomplete safety planning persist despite a negative discharge screener", "admission command hallucinations and historical interrupted attempt documented", ["negative_discharge_screener", "readiness_conflict"]),
        p_case(172, "Final schizoaffective command hallucination barrier", "schizoaffective", "lower_level_of_care_barrier_reasoning", "final_schizo_recent_command_ah_barrier", base + timedelta(days=4), False, "3 days", ["recent command hallucinations", "medication response partial", "safety plan incomplete"], "moderate-to-high current risk due to recent command hallucinations and incomplete safety plan", "historical command hallucinations and interrupted attempt documented", ["recent_command_ah"]),
        p_case(173, "Final psychotic depression malformed score barrier", "mdd_psychosis", "missing_invalid_or_stale_evidence", "final_mdd_malformed_score_barrier", base + timedelta(days=8), False, "2 days", ["malformed PHQ-9 requires verification", "sleep not stabilized", "safety plan incomplete"], "moderate current risk based on narrative evidence; malformed PHQ-9 should not be scored as valid", "historical severe depression, hallucinations, and suicidal thoughts documented", ["malformed_score"], rating_mode="malformed"),
        p_case(174, "Final psychotic depression current low after PHP confirmation", "mdd_psychosis", "current_vs_historical_risk", "final_mdd_current_low_php_ready", base + timedelta(days=12), True, "0 days", [], "low current risk after medication response, sleep improvement, and PHP intake confirmation", "prior psychotic depression and recurrent crisis presentation documented", ["current_denial_vs_history"]),
        p_case(175, "Final BPD independent ADLs but incomplete safety plan", "bpd", "lower_level_of_care_barrier_reasoning", "final_bpd_adl_safety_mismatch", base + timedelta(days=16), False, "2 days", ["independent ADLs do not resolve suicide risk", "overnight coping plan incomplete", "support person not confirmed"], "moderate current risk despite independent ADLs because crisis planning remains unreliable", "history of self-harm urges during interpersonal conflict documented", ["adl_safety_mismatch"]),
        p_case(176, "Final BPD current low after support confirmation", "bpd", "current_vs_historical_risk", "final_bpd_current_low_support_confirmed", base + timedelta(days=20), True, "0 days", [], "low current risk after completed crisis plan and overnight support confirmation", "historical affective instability and self-harm crisis documented", ["current_denial_vs_history"]),
        p_case(177, "Final substance-induced refused C-SSRS barrier", "substance_induced", "missing_invalid_or_stale_evidence", "final_substance_refused_cssrs_barrier", base + timedelta(days=24), False, "3 days", ["refused suicide intensity questions", "dual diagnosis follow-up pending", "housing not confirmed"], "assumed high current risk because suicide intensity items were refused while paranoia and placement gaps persist", "stimulant-associated paranoia and unsafe thoughts documented", ["refused_cssrs"], rating_mode="missing"),
        p_case(178, "Final substance-induced recovery supports resolved", "substance_induced", "lower_level_of_care_barrier_reasoning", "final_substance_barriers_resolved", base + timedelta(days=28), True, "0 days", [], "low current risk after dual-diagnosis follow-up, housing, and safety plan were confirmed", "substance-induced mood and psychotic symptoms documented earlier in stay", ["lloc_barriers_resolved"]),
        p_case(179, "Final bipolar sleep instability blocks step-down", "bipolar", "lower_level_of_care_barrier_reasoning", "final_bipolar_sleep_instability", base + timedelta(days=32), False, "3 days", ["sleep not stabilized", "medication response partial", "follow-up appointment pending"], "moderate current risk because sleep, medication response, and follow-up remain unstable", "mixed mood episode with impulsive unsafe statements documented", ["lloc_barriers"]),
        p_case(180, "Final bipolar current low after monitoring confirmed", "bipolar", "current_vs_historical_risk", "final_bipolar_current_low_monitoring_confirmed", base + timedelta(days=36), True, "0 days", [], "low current risk after sleep improved and outpatient monitoring was confirmed", "historical mixed-episode impulsivity documented for context only", ["current_denial_vs_history"]),
    ]


def build_specs_180():
    return build_specs_170() + specs_171_180()


def validate_cases(cases, batch_cases):
    failures = []
    if len(cases) != EXPECTED_TOTAL_CASES:
        failures.append(f"expected {EXPECTED_TOTAL_CASES} total cases, got {len(cases)}")
    if len(batch_cases) != EXPECTED_BATCH_CASES:
        failures.append(f"expected {EXPECTED_BATCH_CASES} batch cases, got {len(batch_cases)}")

    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        failures.append("case IDs are not unique")
    expected_batch_ids = {f"clin_auth_bench_v1_{case_no:04d}" for case_no in range(171, 181)}
    actual_batch_ids = {case["id"] for case in batch_cases}
    if actual_batch_ids != expected_batch_ids:
        failures.append("171-180 batch IDs do not match expected case range")

    safe_count = sum(case["metadata"]["gold"]["safe_for_lloc"] for case in cases)
    continued_count = len(cases) - safe_count
    if continued_count != EXPECTED_TOTAL_CONTINUED_STAY:
        failures.append(f"expected {EXPECTED_TOTAL_CONTINUED_STAY} total continued-stay cases, got {continued_count}")
    if safe_count != EXPECTED_TOTAL_SAFE_FOR_LLOC:
        failures.append(f"expected {EXPECTED_TOTAL_SAFE_FOR_LLOC} total safe/LLOC-ready cases, got {safe_count}")

    batch_safe_count = sum(case["metadata"]["gold"]["safe_for_lloc"] for case in batch_cases)
    batch_continued_count = len(batch_cases) - batch_safe_count
    if batch_continued_count != EXPECTED_BATCH_CONTINUED_STAY:
        failures.append(f"expected {EXPECTED_BATCH_CONTINUED_STAY} batch continued-stay cases, got {batch_continued_count}")
    if batch_safe_count != EXPECTED_BATCH_SAFE_FOR_LLOC:
        failures.append(f"expected {EXPECTED_BATCH_SAFE_FOR_LLOC} batch safe/LLOC-ready cases, got {batch_safe_count}")

    diagnosis_counts = Counter(case["metadata"]["diagnosis_category"] for case in cases)
    if dict(sorted(diagnosis_counts.items())) != dict(sorted(EXPECTED_DIAGNOSIS_COUNTS.items())):
        failures.append(f"diagnosis counts do not match expected distribution: {dict(sorted(diagnosis_counts.items()))}")

    challenge_counts = Counter(case["metadata"]["documentation_challenge"] for case in cases)
    contradiction_count = challenge_counts["contradiction"]
    if not EXPECTED_TOTAL_CONTRADICTION_MIN <= contradiction_count <= EXPECTED_TOTAL_CONTRADICTION_MAX:
        failures.append(
            f"expected {EXPECTED_TOTAL_CONTRADICTION_MIN}-{EXPECTED_TOTAL_CONTRADICTION_MAX} total contradiction cases, got {contradiction_count}"
        )

    for case in cases:
        if not case["metadata"]["quality_checks"]["passed"]:
            failures.append(f"{case['id']} failed quality checks")
        if not case["metadata"]["content_gold_checks"]["passed"]:
            failures.append(f"{case['id']} failed content/gold checks")
        for marker in required_content_markers():
            if marker not in case["content"]:
                failures.append(f"{case['id']} missing required content marker: {marker}")

    for case in batch_cases:
        if case["metadata"].get("mdp_model") != "probabilistic_v1":
            failures.append(f"{case['id']} is not marked as probabilistic_v1")
        for step in case["metadata"].get("mdp_trajectory", []):
            if step.get("transition_model") != "probabilistic_v1":
                failures.append(f"{case['id']} has non-probabilistic transition step")
            if "transition_options" not in step or "selected_outcome" not in step:
                failures.append(f"{case['id']} missing probability trace fields")

    return failures


def print_rollup(cases, batch_cases):
    safe_count = sum(case["metadata"]["gold"]["safe_for_lloc"] for case in cases)
    continued_count = len(cases) - safe_count
    batch_safe_count = sum(case["metadata"]["gold"]["safe_for_lloc"] for case in batch_cases)
    batch_continued_count = len(batch_cases) - batch_safe_count
    challenge_counts = Counter(case["metadata"]["documentation_challenge"] for case in cases)
    diagnosis_counts = Counter(case["metadata"]["diagnosis_category"] for case in cases)
    batch_challenge_counts = Counter(case["metadata"]["documentation_challenge"] for case in batch_cases)
    batch_diagnosis_counts = Counter(case["metadata"]["diagnosis_category"] for case in batch_cases)
    mdp_models = Counter(case["metadata"].get("mdp_model") for case in batch_cases)
    print(f"Wrote {len(cases)} ClinAuthBench v1 release-candidate cases to {CUMULATIVE_OUT_PATH}")
    print(f"Wrote {len(batch_cases)} QA batch cases to {BATCH_OUT_PATH}")
    print(f"Cumulative continued stay: {continued_count}; safe/LLOC-ready: {safe_count}")
    print(f"Batch continued stay: {batch_continued_count}; safe/LLOC-ready: {batch_safe_count}")
    print("Batch MDP models:", dict(sorted(mdp_models.items())))
    print("Cumulative documentation challenges:", dict(sorted(challenge_counts.items())))
    print("Batch documentation challenges:", dict(sorted(batch_challenge_counts.items())))
    print("Cumulative diagnosis families:", dict(sorted(diagnosis_counts.items())))
    print("Batch diagnosis families:", dict(sorted(batch_diagnosis_counts.items())))


def main():
    cases = [build_case(spec) for spec in build_specs_180()]
    batch_cases = [case for case in cases if 171 <= int(case["id"].rsplit("_", 1)[-1]) <= 180]
    failures = validate_cases(cases, batch_cases)
    if failures:
        raise RuntimeError("Generated 180-case dataset failed validation:\n" + "\n".join(failures))

    CUMULATIVE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUMULATIVE_OUT_PATH.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    BATCH_OUT_PATH.write_text(json.dumps(batch_cases, indent=2), encoding="utf-8")
    print_rollup(cases, batch_cases)


if __name__ == "__main__":
    main()
