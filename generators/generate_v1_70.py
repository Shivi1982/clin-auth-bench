"""
Generate the ClinAuthBench v1 70-case dataset and the 31-70 QA batch.

This generator keeps the MDP-style construction path:
CaseSpec -> hidden MDP trajectory -> rendered forms -> gold labels -> validation.

Outputs:
- clin_auth_bench/outputs/generated_artifacts/synthetic_bh_cases_v1_mdp_70.json
- clin_auth_bench/data/review_batches/synthetic_bh_cases_v1_mdp_31_70.json
"""

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from generate_v1_30 import build_specs_30
from mdp_case_builder import CaseSpec, build_case


BENCH_DIR = Path(__file__).resolve().parents[1]
CUMULATIVE_OUT_PATH = BENCH_DIR / "outputs" / "generated_artifacts" / "synthetic_bh_cases_v1_mdp_70.json"
BATCH_OUT_PATH = BENCH_DIR / "data" / "review_batches" / "synthetic_bh_cases_v1_mdp_31_70.json"

EXPECTED_TOTAL_CASES = 70
EXPECTED_BATCH_CASES = 40
EXPECTED_TOTAL_CONTINUED_STAY = 42
EXPECTED_TOTAL_SAFE_FOR_LLOC = 28
EXPECTED_BATCH_CONTINUED_STAY = 24
EXPECTED_BATCH_SAFE_FOR_LLOC = 16
EXPECTED_DIAGNOSIS_TOTAL_EACH = 10
EXPECTED_TOTAL_CONTRADICTION_MIN = 7
EXPECTED_TOTAL_CONTRADICTION_MAX = 10


def specs_31_70():
    base = datetime(2025, 9, 1)
    return [
        CaseSpec(31, "Continued stay for psychosis readiness conflict", "schizoaffective", "contradiction", "negative_screener_but_psychosis_not_ready", base, False, "3 days", ["psychiatry note says not ready", "safety plan incomplete", "collateral monitoring not confirmed"], "moderate current risk because recent internal preoccupation and incomplete safety planning outweigh a negative discharge screener", "admission command hallucinations and historical interrupted attempt documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(32, "Continued stay with copied-forward schizoaffective risk", "schizoaffective", "missing_invalid_or_stale_evidence", "copied_forward_psychosis_barriers", base + timedelta(days=4), False, "2 days", ["copied-forward risk text requires reconciliation", "medication response partial", "housing not confirmed"], "moderate current risk; copied-forward high-risk language is stale but current psychosis and planning barriers remain", "prior command hallucinations and treatment nonadherence documented", ["copied_forward"]),
        CaseSpec(33, "Continued stay for command hallucination barrier review", "schizoaffective", "lower_level_of_care_barrier_reasoning", "psychosis_partial_response_supports_unready", base + timedelta(days=8), False, "3 days", ["recent command hallucinations", "safety plan incomplete", "medication response partial"], "moderate-to-high current risk due to recent command hallucinations and incomplete safety plan", "historical interrupted attempt and command hallucinations documented", ["recent_command_ah"]),
        CaseSpec(34, "Safe step-down after current denial with psychosis history", "schizoaffective", "current_vs_historical_risk", "historical_psychosis_current_ready", base + timedelta(days=12), True, "0 days", [], "low current risk after final review, with historical psychosis risk documented for context only", "admission command hallucinations and historical interrupted attempt documented", ["current_denial_vs_history"]),
        CaseSpec(35, "Safe step-down after schizoaffective support confirmation", "schizoaffective", "current_vs_historical_risk", "current_denial_supports_confirmed_psychosis", base + timedelta(days=16), True, "0 days", [], "low current SI by final screener after supports and PHP intake were confirmed", "prior command hallucinations and inpatient monitoring need documented earlier in stay", ["current_denial_vs_history"]),
        CaseSpec(36, "Continued stay for depressive psychosis discharge conflict", "mdd_psychosis", "contradiction", "negative_screener_mdd_practitioner_not_ready", base + timedelta(days=20), False, "2 days", ["psychiatry note says not ready", "sleep not stabilized", "collateral monitoring not confirmed"], "moderate current risk because severe depression, poor sleep, and incomplete collateral monitoring persist despite negative screener", "prior ED presentation for suicidal ideation and psychotic depression documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(37, "Continued stay with malformed PHQ in psychotic depression", "mdd_psychosis", "missing_invalid_or_stale_evidence", "malformed_phq_mdd_current_barriers", base + timedelta(days=24), False, "2 days", ["malformed PHQ-9 requires verification", "safety plan incomplete", "sleep not stabilized"], "moderate current risk based on narrative evidence; malformed PHQ-9 should not be scored as valid", "severe depression with psychotic features and suicidal ideation documented", ["malformed_score"], rating_mode="malformed"),
        CaseSpec(38, "Continued stay for psychotic depression support gap", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "mdd_improved_but_no_monitoring", base + timedelta(days=28), False, "3 days", ["collateral monitoring not confirmed", "safety plan incomplete", "follow-up appointment pending"], "moderate current risk because symptom improvement is partial and monitoring supports remain incomplete", "history of medication nonadherence and recurrent crisis presentation documented", ["lloc_barriers"]),
        CaseSpec(39, "Safe step-down after psychotic depression stabilizes", "mdd_psychosis", "current_vs_historical_risk", "mdd_history_current_low_risk", base + timedelta(days=32), True, "0 days", [], "low current risk after sleep, medication response, and crisis plan improved", "admission severe depression and suicidal ideation documented for historical context", ["current_denial_vs_history"]),
        CaseSpec(40, "Safe step-down after MDD barrier resolution", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "mdd_barriers_resolved_ready", base + timedelta(days=36), True, "0 days", [], "low current risk after medication response and confirmed outpatient monitoring", "prior psychotic depression and recurrent crisis presentation documented", ["lloc_barriers_resolved"]),
        CaseSpec(41, "Continued stay for BPD readiness conflict", "bpd", "contradiction", "bpd_discharge_screen_negative_practitioner_not_ready", base + timedelta(days=40), False, "3 days", ["psychiatry note says not ready", "overnight support not confirmed", "crisis plan incomplete"], "moderate current risk because affective instability and incomplete overnight support persist despite negative screener", "historical self-harm crisis after interpersonal conflict documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(42, "Continued stay with missing scores in self-harm crisis", "bpd", "missing_invalid_or_stale_evidence", "bpd_missing_scores_barriers", base + timedelta(days=44), False, "2 days", ["PHQ-9 refused", "crisis plan incomplete", "support person not confirmed"], "moderate current risk based on narrative evidence; missing scores should not be invented", "recent self-harm urges and medication nonadherence documented", ["missing_scores"], rating_mode="missing"),
        CaseSpec(43, "Continued stay for BPD safety plan unreliability", "bpd", "lower_level_of_care_barrier_reasoning", "bpd_adl_ok_safety_not_ok", base + timedelta(days=48), False, "2 days", ["independent ADLs do not resolve suicide risk", "overnight coping plan incomplete", "support person not confirmed"], "moderate current risk despite independent ADLs because safety planning remains unreliable", "history of self-harm urges during interpersonal conflict documented", ["adl_safety_mismatch"]),
        CaseSpec(44, "Safe step-down after BPD current risk resolves", "bpd", "current_vs_historical_risk", "bpd_historical_self_harm_current_ready", base + timedelta(days=52), True, "0 days", [], "low current risk after completed crisis plan and confirmed supports", "historical self-harm crisis documented during admission context", ["current_denial_vs_history"]),
        CaseSpec(45, "Safe step-down after BPD crisis plan completion", "bpd", "lower_level_of_care_barrier_reasoning", "bpd_barriers_resolved_ready", base + timedelta(days=56), True, "0 days", [], "low current risk with completed crisis plan and step-down support confirmed", "interpersonal conflict and self-harm urges documented earlier in stay", ["lloc_barriers_resolved"]),
        CaseSpec(46, "Safe step-down despite ADL documentation noise", "bpd", "lower_level_of_care_barrier_reasoning", "bpd_adl_noise_currently_ready", base + timedelta(days=60), True, "0 days", [], "low current risk after support confirmation; ADL independence alone is not the reason for readiness", "historical self-harm crisis and affective instability documented", ["adl_safety_mismatch"]),
        CaseSpec(47, "Continued stay after refused C-SSRS in substance-induced crisis", "substance_induced", "missing_invalid_or_stale_evidence", "substance_refused_cssrs_barriers", base + timedelta(days=64), False, "3 days", ["refused suicide intensity questions", "dual diagnosis follow-up pending", "housing not confirmed"], "assumed high current risk because suicide intensity items were refused and dual-diagnosis supports remain incomplete", "stimulant-associated paranoia and unsafe thoughts documented", ["refused_cssrs"], rating_mode="missing"),
        CaseSpec(48, "Continued stay for persistent paranoia after intoxication", "substance_induced", "lower_level_of_care_barrier_reasoning", "persistent_paranoia_after_observation", base + timedelta(days=68), False, "4 days", ["persistent paranoia after observation", "dual diagnosis follow-up pending", "housing not confirmed"], "moderate-to-high current risk because psychotic symptoms persist after intoxication clears", "stimulant-associated paranoia and missed treatment documented", ["substance_vs_primary_symptoms"]),
        CaseSpec(49, "Safe step-down after substance-induced symptoms clear", "substance_induced", "current_vs_historical_risk", "substance_history_current_ready", base + timedelta(days=72), True, "0 days", [], "low current risk after paranoia resolved and recovery follow-up was confirmed", "historical stimulant-associated unsafe thoughts documented", ["current_denial_vs_history"]),
        CaseSpec(50, "Safe step-down after current denial in substance crisis", "substance_induced", "current_vs_historical_risk", "substance_current_denial_recovery_confirmed", base + timedelta(days=76), True, "0 days", [], "low current risk with historical intoxication-related paranoia documented for context only", "prior stimulant-associated paranoia and unsafe thoughts documented", ["current_denial_vs_history"]),
        CaseSpec(51, "Safe step-down after dual diagnosis follow-up confirmed", "substance_induced", "lower_level_of_care_barrier_reasoning", "substance_lloc_barriers_resolved", base + timedelta(days=80), True, "0 days", [], "low current risk after dual-diagnosis follow-up, housing, and safety plan were confirmed", "substance-induced mood and psychotic symptoms documented earlier in stay", ["lloc_barriers_resolved"]),
        CaseSpec(52, "Safe step-down after substance recovery plan completion", "substance_induced", "lower_level_of_care_barrier_reasoning", "substance_recovery_plan_ready", base + timedelta(days=84), True, "0 days", [], "low current risk after recovery plan and outpatient monitoring were documented as complete", "prior unsafe thoughts during stimulant use documented", ["lloc_barriers_resolved"]),
        CaseSpec(53, "Continued stay for bipolar readiness contradiction", "bipolar", "contradiction", "bipolar_negative_screener_not_ready", base + timedelta(days=88), False, "3 days", ["psychiatry note says not ready", "sleep not stabilized", "medication monitoring not arranged"], "moderate current risk because mixed mood symptoms and sleep instability persist despite negative screener", "history of impulsive unsafe behavior during mixed mood episode documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(54, "Continued stay with stale bipolar admission risk", "bipolar", "missing_invalid_or_stale_evidence", "bipolar_copied_forward_current_barriers", base + timedelta(days=92), False, "2 days", ["copied-forward risk text requires reconciliation", "medication response partial", "housing not confirmed"], "moderate current risk; stale admission risk should be reconciled but current mixed symptoms and barriers remain", "prior impulsive unsafe behavior and medication nonadherence documented", ["copied_forward"]),
        CaseSpec(55, "Continued stay for fragmented bipolar shift risk", "bipolar", "lower_level_of_care_barrier_reasoning", "bipolar_fragmented_shift_risk", base + timedelta(days=96), False, "3 days", ["night shift pacing", "day shift denial conflicts with later unsafe statement", "medication monitoring not arranged"], "ongoing moderate risk requiring integration of fragmented nursing shifts", "history of impulsive behavior during mixed mood episodes documented", ["fragmented_shift_notes"]),
        CaseSpec(56, "Continued stay for bipolar sleep and monitoring barriers", "bipolar", "lower_level_of_care_barrier_reasoning", "bipolar_sleep_not_stable", base + timedelta(days=100), False, "2 days", ["sleep not stabilized", "medication response partial", "follow-up appointment pending"], "moderate current risk because sleep, medication response, and follow-up remain unstable", "mixed mood episode with impulsive unsafe statements documented", ["lloc_barriers"]),
        CaseSpec(57, "Safe step-down after bipolar mixed episode improves", "bipolar", "current_vs_historical_risk", "bipolar_history_current_ready", base + timedelta(days=104), True, "0 days", [], "low current risk after sleep improved and step-down monitoring was confirmed", "historical mixed-episode impulsivity documented for context only", ["current_denial_vs_history"]),
        CaseSpec(58, "Safe step-down after bipolar barrier resolution", "bipolar", "lower_level_of_care_barrier_reasoning", "bipolar_barriers_resolved_ready", base + timedelta(days=108), True, "0 days", [], "low current risk after medication response and outpatient follow-up were confirmed", "prior mixed mood crisis and decreased sleep documented earlier in stay", ["lloc_barriers_resolved"]),
        CaseSpec(59, "Continued stay for OUD readiness contradiction", "oud_dual", "contradiction", "oud_negative_screener_cravings_not_ready", base + timedelta(days=112), False, "3 days", ["practitioner clarification pending", "residential SUD placement pending", "opioid cravings not stabilized"], "moderate-to-high current risk because opioid cravings and placement barriers persist despite negative discharge screener", "opioid relapse crisis and missed treatment documented", ["negative_discharge_screener", "readiness_conflict", "oud_crisis"]),
        CaseSpec(60, "Continued stay after refused C-SSRS in OUD crisis", "oud_dual", "missing_invalid_or_stale_evidence", "oud_refused_cssrs_recovery_unready", base + timedelta(days=116), False, "3 days", ["refused suicide intensity questions", "residential SUD placement pending", "housing not confirmed"], "assumed high current risk because suicide intensity items were refused and OUD recovery supports remain incomplete", "prior opioid overdose, cravings, and fentanyl exposure documented", ["refused_cssrs", "oud_crisis"], rating_mode="missing"),
        CaseSpec(61, "Continued stay for OUD recovery placement barrier", "oud_dual", "lower_level_of_care_barrier_reasoning", "oud_residential_pending", base + timedelta(days=120), False, "4 days", ["residential SUD placement pending", "opioid cravings not stabilized", "housing not confirmed"], "moderate-to-high current risk because cravings and recovery placement barriers remain unresolved", "opioid relapse crisis and suicidal thoughts during relapse documented", ["oud_crisis"]),
        CaseSpec(62, "Continued stay for OUD cravings and support gap", "oud_dual", "lower_level_of_care_barrier_reasoning", "oud_cravings_support_gap", base + timedelta(days=124), False, "3 days", ["opioid cravings not stabilized", "collateral monitoring not confirmed", "medication response partial"], "moderate current risk because cravings and monitoring supports remain incomplete", "historical fentanyl exposure and unsafe thoughts documented", ["oud_crisis", "lloc_barriers"]),
        CaseSpec(63, "Safe step-down after historical OUD risk resolves", "oud_dual", "current_vs_historical_risk", "oud_history_current_ready", base + timedelta(days=128), True, "0 days", [], "low current risk after cravings improved and recovery supports were confirmed", "historical fentanyl exposure, opioid cravings, and unsafe thoughts documented", ["current_denial_vs_history", "oud_crisis"]),
        CaseSpec(64, "Safe step-down after OUD residential plan confirmed", "oud_dual", "lower_level_of_care_barrier_reasoning", "oud_residential_confirmed_ready", base + timedelta(days=132), True, "0 days", [], "low current risk after residential recovery placement and medication support were confirmed", "opioid relapse crisis and cravings documented earlier in stay", ["lloc_barriers_resolved", "oud_crisis"]),
        CaseSpec(65, "Continued stay for trauma nightmares and support gap", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "trauma_nightmares_support_gap", base + timedelta(days=136), False, "3 days", ["overnight support not confirmed", "crisis plan incomplete", "sleep not stabilized"], "moderate current risk because nightmares, sleep disruption, and overnight support remain unreliable", "trauma-related passive suicidal thoughts and panic symptoms documented", ["lloc_barriers", "trauma_anxiety"]),
        CaseSpec(66, "Continued stay for trauma panic and incomplete crisis plan", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "trauma_panic_plan_incomplete", base + timedelta(days=140), False, "2 days", ["crisis plan incomplete", "support person not confirmed", "follow-up appointment pending"], "moderate current risk because panic symptoms and crisis planning remain incomplete", "trauma-related anxiety and passive suicidal thoughts documented", ["lloc_barriers", "trauma_anxiety"]),
        CaseSpec(67, "Continued stay for trauma sleep instability", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "trauma_sleep_unstable", base + timedelta(days=144), False, "3 days", ["sleep not stabilized", "overnight support not confirmed", "collateral monitoring not confirmed"], "moderate current risk because sleep disturbance and support reliability remain unresolved", "history of nightmares, hypervigilance, and passive SI documented", ["lloc_barriers", "trauma_anxiety"]),
        CaseSpec(68, "Continued stay for trauma hypervigilance and housing gap", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "trauma_hypervigilance_housing_gap", base + timedelta(days=148), False, "2 days", ["housing not confirmed", "support person not confirmed", "crisis plan incomplete"], "moderate current risk because hypervigilance and step-down supports remain incomplete", "trauma-related panic and suicidal thoughts documented earlier in stay", ["lloc_barriers", "trauma_anxiety"]),
        CaseSpec(69, "Continued stay for trauma follow-up uncertainty", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "trauma_followup_pending", base + timedelta(days=152), False, "2 days", ["follow-up appointment pending", "overnight support not confirmed", "sleep not stabilized"], "moderate current risk because follow-up and overnight support remain unreliable despite partial improvement", "trauma-related anxiety and passive suicidal thoughts documented", ["lloc_barriers", "trauma_anxiety"]),
        CaseSpec(70, "Safe step-down after trauma-related SI resolves", "trauma_anxiety", "current_vs_historical_risk", "trauma_history_current_ready", base + timedelta(days=156), True, "0 days", [], "low current risk after nightmares improved, safety plan completed, and supports were confirmed", "historical trauma-related passive suicidal thoughts and panic symptoms documented", ["current_denial_vs_history", "trauma_anxiety"]),
    ]


def build_specs_70():
    return build_specs_30() + specs_31_70()


def required_content_markers():
    return [
        "C-SSRS - Discharge Screener",
        "Activities of Daily Living:",
        "Medication Compliance:",
        "Hours of sleep/Night:",
        "Mood:",
        "Behavior:",
        "Cognition/Thought Content/Thought Process:",
        "PHQ-9",
        "GAD-7",
    ]


def validate_cases(cases, batch_cases):
    failures = []
    if len(cases) != EXPECTED_TOTAL_CASES:
        failures.append(f"expected {EXPECTED_TOTAL_CASES} total cases, got {len(cases)}")
    if len(batch_cases) != EXPECTED_BATCH_CASES:
        failures.append(f"expected {EXPECTED_BATCH_CASES} batch cases, got {len(batch_cases)}")

    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        failures.append("case IDs are not unique")
    expected_batch_ids = {f"clin_auth_bench_v1_{case_no:04d}" for case_no in range(31, 71)}
    actual_batch_ids = {case["id"] for case in batch_cases}
    if actual_batch_ids != expected_batch_ids:
        failures.append("31-70 batch IDs do not match expected case range")

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
    for diagnosis, count in sorted(diagnosis_counts.items()):
        if count != EXPECTED_DIAGNOSIS_TOTAL_EACH:
            failures.append(f"expected {EXPECTED_DIAGNOSIS_TOTAL_EACH} cases for {diagnosis}, got {count}")

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
    print(f"Wrote {len(cases)} cumulative ClinAuthBench v1 cases to {CUMULATIVE_OUT_PATH}")
    print(f"Wrote {len(batch_cases)} QA batch cases to {BATCH_OUT_PATH}")
    print(f"Cumulative continued stay: {continued_count}; safe/LLOC-ready: {safe_count}")
    print(f"Batch continued stay: {batch_continued_count}; safe/LLOC-ready: {batch_safe_count}")
    print("Cumulative documentation challenges:", dict(sorted(challenge_counts.items())))
    print("Batch documentation challenges:", dict(sorted(batch_challenge_counts.items())))
    print("Cumulative diagnosis families:", dict(sorted(diagnosis_counts.items())))
    print("Batch diagnosis families:", dict(sorted(batch_diagnosis_counts.items())))


def main():
    cases = [build_case(spec) for spec in build_specs_70()]
    batch_cases = [case for case in cases if 31 <= int(case["id"].rsplit("_", 1)[-1]) <= 70]
    failures = validate_cases(cases, batch_cases)
    if failures:
        raise RuntimeError("Generated 70-case dataset failed validation:\n" + "\n".join(failures))

    CUMULATIVE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUMULATIVE_OUT_PATH.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    BATCH_OUT_PATH.write_text(json.dumps(batch_cases, indent=2), encoding="utf-8")
    print_rollup(cases, batch_cases)


if __name__ == "__main__":
    main()
