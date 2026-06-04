"""
Generate the ClinAuthBench v1 120-case dataset and the 71-120 QA batch.

This generator preserves the same MDP construction path used for earlier
batches:
CaseSpec -> hidden MDP trajectory -> rendered forms -> gold labels -> validation.

Outputs:
- clin_auth_bench/outputs/generated_artifacts/synthetic_bh_cases_v1_mdp_120.json
- clin_auth_bench/data/review_batches/synthetic_bh_cases_v1_mdp_71_120.json
"""

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from generate_v1_70 import build_specs_70, required_content_markers
from mdp_case_builder import CaseSpec, build_case


BENCH_DIR = Path(__file__).resolve().parents[1]
CUMULATIVE_OUT_PATH = BENCH_DIR / "outputs" / "generated_artifacts" / "synthetic_bh_cases_v1_mdp_120.json"
BATCH_OUT_PATH = BENCH_DIR / "data" / "review_batches" / "synthetic_bh_cases_v1_mdp_71_120.json"

EXPECTED_TOTAL_CASES = 120
EXPECTED_BATCH_CASES = 50
EXPECTED_TOTAL_CONTINUED_STAY = 72
EXPECTED_TOTAL_SAFE_FOR_LLOC = 48
EXPECTED_BATCH_CONTINUED_STAY = 30
EXPECTED_BATCH_SAFE_FOR_LLOC = 20
EXPECTED_TOTAL_CONTRADICTION_MIN = 14
EXPECTED_TOTAL_CONTRADICTION_MAX = 18

EXPECTED_DIAGNOSIS_COUNTS = {
    "Bipolar mixed episode": 17,
    "Borderline personality disorder self-harm crisis": 17,
    "Dual diagnosis / OUD-related behavioral health crisis": 18,
    "Major depressive disorder with psychotic features": 17,
    "Schizoaffective disorder with command hallucinations": 17,
    "Substance-induced mood or psychotic symptoms": 17,
    "Trauma/anxiety with suicidality": 17,
}


def specs_71_120():
    base = datetime(2026, 1, 1)
    return [
        CaseSpec(71, "OUD negative screener with unresolved cravings", "oud_dual", "contradiction", "oud_negative_screener_cravings_unready", base, False, "3 days", ["practitioner clarification pending", "residential SUD placement pending", "opioid cravings not stabilized"], "moderate-to-high current risk because cravings and residential placement barriers persist despite a negative discharge screener", "prior opioid overdose, fentanyl exposure, and suicidal thoughts during relapse documented", ["negative_discharge_screener", "readiness_conflict", "oud_crisis"]),
        CaseSpec(72, "OUD refused suicide intensity questions", "oud_dual", "missing_invalid_or_stale_evidence", "oud_refused_cssrs_unstable_recovery", base + timedelta(days=4), False, "3 days", ["refused suicide intensity questions", "residential SUD placement pending", "housing not confirmed"], "assumed high current risk because suicide intensity items were refused and recovery placement is unresolved", "historical fentanyl exposure, overdose risk, and medication nonadherence documented", ["refused_cssrs", "oud_crisis"], rating_mode="missing"),
        CaseSpec(73, "OUD relapse crisis with housing gap", "oud_dual", "lower_level_of_care_barrier_reasoning", "oud_relapse_housing_gap", base + timedelta(days=8), False, "4 days", ["housing not confirmed", "opioid cravings not stabilized", "dual diagnosis follow-up pending"], "moderate-to-high current risk because cravings and housing instability make step-down unsafe", "recent opioid relapse crisis with suicidal thoughts documented", ["oud_crisis", "lloc_barriers"]),
        CaseSpec(74, "OUD medication adherence partial after relapse", "oud_dual", "lower_level_of_care_barrier_reasoning", "oud_partial_adherence_residential_pending", base + timedelta(days=12), False, "3 days", ["medication response partial", "residential SUD placement pending", "collateral monitoring not confirmed"], "moderate current risk because buprenorphine adherence and recovery monitoring are not yet stable", "history of fentanyl exposure, cravings, and rapid relapse documented", ["oud_crisis", "lloc_barriers"]),
        CaseSpec(75, "OUD cravings with fragmented shift risk", "oud_dual", "lower_level_of_care_barrier_reasoning", "oud_fragmented_shift_cravings", base + timedelta(days=16), False, "3 days", ["night shift pacing", "opioid cravings not stabilized", "support person not confirmed"], "moderate current risk that becomes clear only after integrating night shift cravings and support gaps", "prior opioid relapse crisis and unsafe thoughts documented", ["fragmented_shift_notes", "oud_crisis"]),
        CaseSpec(76, "OUD recovery placement still pending", "oud_dual", "lower_level_of_care_barrier_reasoning", "oud_placement_pending_not_ready", base + timedelta(days=20), False, "4 days", ["residential SUD placement pending", "housing not confirmed", "follow-up appointment pending"], "moderate current risk because recovery placement and step-down follow-up are incomplete", "historical opioid relapse crisis and missed treatment documented", ["oud_crisis", "lloc_barriers"]),
        CaseSpec(77, "OUD current risk low with historical overdose context", "oud_dual", "current_vs_historical_risk", "oud_current_low_history_high", base + timedelta(days=24), True, "0 days", [], "low current risk after cravings improved and supports were confirmed; historical overdose risk remains context only", "prior opioid overdose, fentanyl exposure, and suicidal thoughts during relapse documented", ["current_denial_vs_history", "oud_crisis"]),
        CaseSpec(78, "OUD safe step-down after residential confirmation", "oud_dual", "lower_level_of_care_barrier_reasoning", "oud_residential_confirmed_safe", base + timedelta(days=28), True, "0 days", [], "low current risk after residential recovery placement and medication support were confirmed", "opioid relapse crisis and cravings documented earlier in stay", ["lloc_barriers_resolved", "oud_crisis"]),
        CaseSpec(79, "Schizoaffective discharge conflict with recent internal preoccupation", "schizoaffective", "contradiction", "schizo_negative_screener_not_ready", base + timedelta(days=32), False, "3 days", ["psychiatry note says not ready", "safety plan incomplete", "collateral monitoring not confirmed"], "moderate current risk because internal preoccupation and incomplete safety planning persist despite a negative screener", "admission command hallucinations and historical interrupted attempt documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(80, "Schizoaffective copied-forward risk requires reconciliation", "schizoaffective", "missing_invalid_or_stale_evidence", "schizo_copied_forward_current_barriers", base + timedelta(days=36), False, "2 days", ["copied-forward risk text requires reconciliation", "medication response partial", "housing not confirmed"], "moderate current risk; copied-forward high-risk text is stale but current barriers remain", "prior command hallucinations and interrupted attempt documented", ["copied_forward"]),
        CaseSpec(81, "Schizoaffective command hallucination monitoring barrier", "schizoaffective", "lower_level_of_care_barrier_reasoning", "schizo_command_ah_monitoring_gap", base + timedelta(days=40), False, "3 days", ["recent command hallucinations", "medication response partial", "safety plan incomplete"], "moderate-to-high current risk due to recent command hallucinations and incomplete safety plan", "historical command hallucinations and interrupted attempt documented", ["recent_command_ah"]),
        CaseSpec(82, "Schizoaffective support not confirmed despite partial improvement", "schizoaffective", "lower_level_of_care_barrier_reasoning", "schizo_partial_improvement_support_gap", base + timedelta(days=44), False, "2 days", ["collateral monitoring not confirmed", "housing not confirmed", "follow-up appointment pending"], "moderate current risk because partial symptom improvement does not resolve missing support confirmation", "prior psychosis, unsafe thoughts, and treatment nonadherence documented", ["lloc_barriers"]),
        CaseSpec(83, "Schizoaffective current denial with historical risk only", "schizoaffective", "current_vs_historical_risk", "schizo_history_current_safe", base + timedelta(days=48), True, "0 days", [], "low current risk by final review with historical command hallucination risk documented only as context", "admission command hallucinations and historical interrupted attempt documented", ["current_denial_vs_history"]),
        CaseSpec(84, "Schizoaffective PHP confirmed after psychosis improves", "schizoaffective", "current_vs_historical_risk", "schizo_php_confirmed_current_low", base + timedelta(days=52), True, "0 days", [], "low current risk after psychosis improved and PHP intake was confirmed", "prior command hallucinations and inpatient safety monitoring documented", ["current_denial_vs_history"]),
        CaseSpec(85, "Schizoaffective LLOC barriers resolved", "schizoaffective", "lower_level_of_care_barrier_reasoning", "schizo_barriers_resolved_ready", base + timedelta(days=56), True, "0 days", [], "low current risk after safety plan, housing, and collateral monitoring were confirmed", "historical psychosis and unsafe thoughts documented earlier in stay", ["lloc_barriers_resolved"]),
        CaseSpec(86, "Psychotic depression negative screener but not clinically ready", "mdd_psychosis", "contradiction", "mdd_negative_screener_sleep_barrier", base + timedelta(days=60), False, "2 days", ["psychiatry note says not ready", "sleep not stabilized", "collateral monitoring not confirmed"], "moderate current risk because severe depression, poor sleep, and collateral gaps persist despite a negative screener", "prior severe depression with psychotic features and suicidal ideation documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(87, "Psychotic depression malformed PHQ with barriers", "mdd_psychosis", "missing_invalid_or_stale_evidence", "mdd_malformed_phq_current_barriers", base + timedelta(days=64), False, "2 days", ["malformed PHQ-9 requires verification", "sleep not stabilized", "safety plan incomplete"], "moderate current risk based on narrative evidence; malformed PHQ-9 should not be scored as valid", "historical severe depression, hallucinations, and suicidal thoughts documented", ["malformed_score"], rating_mode="malformed"),
        CaseSpec(88, "Psychotic depression safety plan unreliable", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "mdd_safety_plan_unreliable", base + timedelta(days=68), False, "3 days", ["safety plan incomplete", "collateral monitoring not confirmed", "follow-up appointment pending"], "moderate current risk because safety planning and follow-up reliability remain incomplete", "recurrent crisis presentation after medication nonadherence documented", ["lloc_barriers"]),
        CaseSpec(89, "Psychotic depression partial medication response", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "mdd_partial_med_response", base + timedelta(days=72), False, "2 days", ["medication response partial", "sleep not stabilized", "housing not confirmed"], "moderate current risk because medication response, sleep, and housing supports remain unstable", "prior psychotic depression with suicidal ideation documented", ["lloc_barriers"]),
        CaseSpec(90, "Psychotic depression current low risk with history", "mdd_psychosis", "current_vs_historical_risk", "mdd_current_low_history_high", base + timedelta(days=76), True, "0 days", [], "low current risk after sleep improved and crisis plan was completed", "admission severe depression and suicidal ideation documented for historical context", ["current_denial_vs_history"]),
        CaseSpec(91, "Psychotic depression stable for PHP", "mdd_psychosis", "current_vs_historical_risk", "mdd_php_ready_current_low", base + timedelta(days=80), True, "0 days", [], "low current risk after medication response and PHP intake confirmation", "prior psychotic depression and recurrent crisis presentation documented", ["current_denial_vs_history"]),
        CaseSpec(92, "Psychotic depression barriers resolved", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "mdd_barriers_resolved_for_stepdown", base + timedelta(days=84), True, "0 days", [], "low current risk after sleep, collateral monitoring, and follow-up were confirmed", "history of severe depression with psychotic features documented earlier in stay", ["lloc_barriers_resolved"]),
        CaseSpec(93, "BPD negative screener but crisis plan incomplete", "bpd", "contradiction", "bpd_negative_screener_crisis_plan_gap", base + timedelta(days=88), False, "3 days", ["psychiatry note says not ready", "crisis plan incomplete", "overnight support not confirmed"], "moderate current risk because affective instability and overnight support gaps persist despite a negative screener", "historical self-harm crisis after interpersonal conflict documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(94, "BPD missing scores with self-harm support gap", "bpd", "missing_invalid_or_stale_evidence", "bpd_missing_scores_support_gap", base + timedelta(days=92), False, "2 days", ["PHQ-9 refused", "support person not confirmed", "crisis plan incomplete"], "moderate current risk based on narrative evidence; refused scores should not be invented", "recent self-harm urges and medication nonadherence documented", ["missing_scores"], rating_mode="missing"),
        CaseSpec(95, "BPD independent ADLs but safety not reliable", "bpd", "lower_level_of_care_barrier_reasoning", "bpd_adl_safety_mismatch_again", base + timedelta(days=96), False, "2 days", ["independent ADLs do not resolve suicide risk", "overnight coping plan incomplete", "support person not confirmed"], "moderate current risk despite independent ADLs because crisis planning remains unreliable", "history of self-harm urges during interpersonal conflict documented", ["adl_safety_mismatch"]),
        CaseSpec(96, "BPD fragmented shifts reveal unresolved risk", "bpd", "lower_level_of_care_barrier_reasoning", "bpd_fragmented_shift_unready", base + timedelta(days=100), False, "3 days", ["night shift pacing", "day shift denial conflicts with later unsafe statement", "medication response partial"], "ongoing moderate risk requiring integration of fragmented nursing notes across shifts", "history of impulsive unsafe behavior and self-harm urges documented", ["fragmented_shift_notes"]),
        CaseSpec(97, "BPD current risk low after support confirmation", "bpd", "current_vs_historical_risk", "bpd_current_low_history_self_harm", base + timedelta(days=104), True, "0 days", [], "low current risk after completed crisis plan and support confirmation", "historical self-harm crisis documented during admission context", ["current_denial_vs_history"]),
        CaseSpec(98, "BPD current denial with DBT step-down plan", "bpd", "current_vs_historical_risk", "bpd_dbt_stepdown_current_safe", base + timedelta(days=108), True, "0 days", [], "low current risk after DBT-oriented step-down support and crisis contacts were confirmed", "interpersonal conflict and self-harm urges documented earlier in stay", ["current_denial_vs_history"]),
        CaseSpec(99, "BPD LLOC barriers resolved after crisis planning", "bpd", "lower_level_of_care_barrier_reasoning", "bpd_barriers_resolved_after_plan", base + timedelta(days=112), True, "0 days", [], "low current risk with completed crisis plan and overnight support confirmed", "historical affective instability and self-harm crisis documented", ["lloc_barriers_resolved"]),
        CaseSpec(100, "Substance-induced refused C-SSRS with paranoia", "substance_induced", "missing_invalid_or_stale_evidence", "substance_refused_cssrs_paranoia", base + timedelta(days=116), False, "3 days", ["refused suicide intensity questions", "dual diagnosis follow-up pending", "housing not confirmed"], "assumed high current risk because suicide intensity items were refused while paranoia and placement gaps persist", "stimulant-associated paranoia and unsafe thoughts documented", ["refused_cssrs"], rating_mode="missing"),
        CaseSpec(101, "Substance-induced copied-forward intoxication risk", "substance_induced", "missing_invalid_or_stale_evidence", "substance_copied_forward_risk", base + timedelta(days=120), False, "2 days", ["copied-forward risk text requires reconciliation", "persistent paranoia after observation", "dual diagnosis follow-up pending"], "moderate current risk; copied-forward intoxication risk is stale but persistent paranoia remains active", "recent stimulant-associated paranoia and missed treatment documented", ["copied_forward", "substance_vs_primary_symptoms"]),
        CaseSpec(102, "Substance-induced paranoia persists after observation", "substance_induced", "lower_level_of_care_barrier_reasoning", "substance_persistent_paranoia_observation", base + timedelta(days=124), False, "4 days", ["persistent paranoia after observation", "dual diagnosis follow-up pending", "housing not confirmed"], "moderate-to-high current risk because paranoia persists after intoxication clears", "stimulant-associated paranoia and unsafe thoughts documented", ["substance_vs_primary_symptoms"]),
        CaseSpec(103, "Substance-induced recovery follow-up pending", "substance_induced", "lower_level_of_care_barrier_reasoning", "substance_recovery_followup_pending", base + timedelta(days=128), False, "3 days", ["dual diagnosis follow-up pending", "housing not confirmed", "medication monitoring not arranged"], "moderate current risk because recovery follow-up and step-down monitoring are not confirmed", "history of stimulant-associated unsafe thoughts documented", ["lloc_barriers", "substance_vs_primary_symptoms"]),
        CaseSpec(104, "Substance-induced current risk low after clearance", "substance_induced", "current_vs_historical_risk", "substance_current_low_after_clearance", base + timedelta(days=132), True, "0 days", [], "low current risk after paranoia resolved and recovery follow-up was confirmed", "historical stimulant-associated paranoia and unsafe thoughts documented", ["current_denial_vs_history"]),
        CaseSpec(105, "Substance-induced current denial with recovery supports", "substance_induced", "current_vs_historical_risk", "substance_current_denial_supports", base + timedelta(days=136), True, "0 days", [], "low current risk with historical intoxication-related unsafe thoughts documented only as context", "prior stimulant-associated paranoia and crisis presentation documented", ["current_denial_vs_history"]),
        CaseSpec(106, "Substance-induced barriers resolved for step-down", "substance_induced", "lower_level_of_care_barrier_reasoning", "substance_barriers_resolved_for_lloc", base + timedelta(days=140), True, "0 days", [], "low current risk after dual-diagnosis follow-up, housing, and safety plan were confirmed", "substance-induced mood and psychotic symptoms documented earlier in stay", ["lloc_barriers_resolved"]),
        CaseSpec(107, "Bipolar negative screener but mixed symptoms persist", "bipolar", "contradiction", "bipolar_negative_screener_mixed_unready", base + timedelta(days=144), False, "3 days", ["psychiatry note says not ready", "sleep not stabilized", "medication monitoring not arranged"], "moderate current risk because mixed mood symptoms and sleep instability persist despite a negative screener", "history of impulsive unsafe behavior during mixed mood episode documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(108, "Bipolar stale admission risk with current barriers", "bipolar", "missing_invalid_or_stale_evidence", "bipolar_copied_forward_barriers", base + timedelta(days=148), False, "2 days", ["copied-forward risk text requires reconciliation", "medication response partial", "housing not confirmed"], "moderate current risk; copied-forward admission high-risk text is stale but mixed symptoms and barriers remain", "prior impulsive unsafe behavior and medication nonadherence documented", ["copied_forward"]),
        CaseSpec(109, "Bipolar sleep instability blocks step-down", "bipolar", "lower_level_of_care_barrier_reasoning", "bipolar_sleep_instability_unready", base + timedelta(days=152), False, "3 days", ["sleep not stabilized", "medication response partial", "follow-up appointment pending"], "moderate current risk because sleep, medication response, and follow-up remain unstable", "mixed mood episode with impulsive unsafe statements documented", ["lloc_barriers"]),
        CaseSpec(110, "Bipolar fragmented shift notes reveal risk", "bipolar", "lower_level_of_care_barrier_reasoning", "bipolar_fragmented_shift_notes", base + timedelta(days=156), False, "3 days", ["night shift pacing", "day shift denial conflicts with later unsafe statement", "medication monitoring not arranged"], "ongoing moderate risk requiring integration of fragmented shift notes", "history of impulsive behavior during mixed mood episodes documented", ["fragmented_shift_notes"]),
        CaseSpec(111, "Bipolar current low risk after sleep improves", "bipolar", "current_vs_historical_risk", "bipolar_sleep_improved_current_low", base + timedelta(days=160), True, "0 days", [], "low current risk after sleep improved and monitoring was confirmed", "historical mixed-episode impulsivity documented for context only", ["current_denial_vs_history"]),
        CaseSpec(112, "Bipolar PHP confirmed after medication response", "bipolar", "current_vs_historical_risk", "bipolar_php_confirmed_ready", base + timedelta(days=164), True, "0 days", [], "low current risk after medication response and PHP intake confirmation", "prior mixed mood crisis and decreased sleep documented earlier in stay", ["current_denial_vs_history"]),
        CaseSpec(113, "Bipolar barriers resolved for step-down", "bipolar", "lower_level_of_care_barrier_reasoning", "bipolar_barriers_resolved_for_lloc", base + timedelta(days=168), True, "0 days", [], "low current risk after medication response, sleep, and outpatient monitoring were confirmed", "history of mixed mood episode with unsafe impulsivity documented", ["lloc_barriers_resolved"]),
        CaseSpec(114, "Trauma negative screener but overnight support gap", "trauma_anxiety", "contradiction", "trauma_negative_screener_support_gap", base + timedelta(days=172), False, "3 days", ["psychiatry note says not ready", "overnight support not confirmed", "crisis plan incomplete"], "moderate current risk because nightmares and overnight support gaps persist despite a negative screener", "trauma-related passive suicidal thoughts and panic symptoms documented", ["negative_discharge_screener", "readiness_conflict", "trauma_anxiety"]),
        CaseSpec(115, "Trauma missing scores with panic symptoms", "trauma_anxiety", "missing_invalid_or_stale_evidence", "trauma_missing_scores_panic", base + timedelta(days=176), False, "2 days", ["PHQ-9 refused", "crisis plan incomplete", "support person not confirmed"], "moderate current risk based on narrative panic and sleep evidence; missing scores should not be invented", "history of nightmares, hypervigilance, and passive suicidal thoughts documented", ["missing_scores", "trauma_anxiety"], rating_mode="missing"),
        CaseSpec(116, "Trauma nightmares and sleep instability", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "trauma_nightmares_sleep_instability", base + timedelta(days=180), False, "3 days", ["sleep not stabilized", "overnight support not confirmed", "collateral monitoring not confirmed"], "moderate current risk because nightmares, sleep disruption, and support reliability remain unresolved", "trauma-related anxiety and passive suicidal thoughts documented", ["lloc_barriers", "trauma_anxiety"]),
        CaseSpec(117, "Trauma hypervigilance with housing uncertainty", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "trauma_hypervigilance_housing_uncertain", base + timedelta(days=184), False, "2 days", ["housing not confirmed", "support person not confirmed", "crisis plan incomplete"], "moderate current risk because hypervigilance and step-down supports remain incomplete", "trauma-related panic and suicidal thoughts documented earlier in stay", ["lloc_barriers", "trauma_anxiety"]),
        CaseSpec(118, "Trauma current low risk after supports confirmed", "trauma_anxiety", "current_vs_historical_risk", "trauma_current_low_history_si", base + timedelta(days=188), True, "0 days", [], "low current risk after nightmares improved, safety plan completed, and supports were confirmed", "historical trauma-related passive suicidal thoughts and panic symptoms documented", ["current_denial_vs_history", "trauma_anxiety"]),
        CaseSpec(119, "Trauma current denial with therapy follow-up", "trauma_anxiety", "current_vs_historical_risk", "trauma_therapy_followup_ready", base + timedelta(days=192), True, "0 days", [], "low current risk after therapy follow-up and overnight support were confirmed", "prior panic episode with passive suicidal thoughts documented", ["current_denial_vs_history", "trauma_anxiety"]),
        CaseSpec(120, "Trauma barriers resolved after sleep planning", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "trauma_barriers_resolved_ready", base + timedelta(days=196), True, "0 days", [], "low current risk after sleep plan, crisis plan, and support contacts were confirmed", "historical trauma-related anxiety and passive suicidal thoughts documented", ["lloc_barriers_resolved", "trauma_anxiety"]),
    ]


def build_specs_120():
    return build_specs_70() + specs_71_120()


def validate_cases(cases, batch_cases):
    failures = []
    if len(cases) != EXPECTED_TOTAL_CASES:
        failures.append(f"expected {EXPECTED_TOTAL_CASES} total cases, got {len(cases)}")
    if len(batch_cases) != EXPECTED_BATCH_CASES:
        failures.append(f"expected {EXPECTED_BATCH_CASES} batch cases, got {len(batch_cases)}")

    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        failures.append("case IDs are not unique")
    expected_batch_ids = {f"clin_auth_bench_v1_{case_no:04d}" for case_no in range(71, 121)}
    actual_batch_ids = {case["id"] for case in batch_cases}
    if actual_batch_ids != expected_batch_ids:
        failures.append("71-120 batch IDs do not match expected case range")

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
    cases = [build_case(spec) for spec in build_specs_120()]
    batch_cases = [case for case in cases if 71 <= int(case["id"].rsplit("_", 1)[-1]) <= 120]
    failures = validate_cases(cases, batch_cases)
    if failures:
        raise RuntimeError("Generated 120-case dataset failed validation:\n" + "\n".join(failures))

    CUMULATIVE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUMULATIVE_OUT_PATH.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    BATCH_OUT_PATH.write_text(json.dumps(batch_cases, indent=2), encoding="utf-8")
    print_rollup(cases, batch_cases)


if __name__ == "__main__":
    main()
