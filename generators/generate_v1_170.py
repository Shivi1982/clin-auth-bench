"""
Generate the ClinAuthBench v1 170-case dataset and the 121-170 QA batch.

Cases 121-170 use explicit probabilistic MDP transitions. The generator still
conditions on the intended case target so the accepted batch preserves the
planned disposition mix while recording probabilities, eligible outcomes, RNG
rolls, and selected outcomes in metadata.mdp_trajectory.

Outputs:
- clin_auth_bench/outputs/generated_artifacts/synthetic_bh_cases_v1_mdp_170.json
- clin_auth_bench/data/review_batches/synthetic_bh_cases_v1_mdp_121_170.json
"""

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from generate_v1_70 import required_content_markers
from generate_v1_120 import build_specs_120
from mdp_case_builder import CaseSpec, build_case


BENCH_DIR = Path(__file__).resolve().parents[1]
CUMULATIVE_OUT_PATH = BENCH_DIR / "outputs" / "generated_artifacts" / "synthetic_bh_cases_v1_mdp_170.json"
BATCH_OUT_PATH = BENCH_DIR / "data" / "review_batches" / "synthetic_bh_cases_v1_mdp_121_170.json"

EXPECTED_TOTAL_CASES = 170
EXPECTED_BATCH_CASES = 50
EXPECTED_TOTAL_CONTINUED_STAY = 102
EXPECTED_TOTAL_SAFE_FOR_LLOC = 68
EXPECTED_BATCH_CONTINUED_STAY = 30
EXPECTED_BATCH_SAFE_FOR_LLOC = 20
EXPECTED_TOTAL_CONTRADICTION_MIN = 17
EXPECTED_TOTAL_CONTRADICTION_MAX = 25

EXPECTED_DIAGNOSIS_COUNTS = {
    "Bipolar mixed episode": 24,
    "Borderline personality disorder self-harm crisis": 24,
    "Dual diagnosis / OUD-related behavioral health crisis": 25,
    "Major depressive disorder with psychotic features": 24,
    "Schizoaffective disorder with command hallucinations": 24,
    "Substance-induced mood or psychotic symptoms": 24,
    "Trauma/anxiety with suicidality": 25,
}


def p_case(*args, **kwargs):
    kwargs.setdefault("mdp_model", "probabilistic_v1")
    return CaseSpec(*args, **kwargs)


def specs_121_170():
    base = datetime(2026, 8, 1)
    return [
        p_case(121, "Probabilistic OUD negative screener with craving barrier", "oud_dual", "contradiction", "prob_oud_negative_screener_cravings", base, False, "3 days", ["practitioner clarification pending", "residential SUD placement pending", "opioid cravings not stabilized"], "moderate-to-high current risk because opioid cravings and residential placement barriers persist despite a negative discharge screener", "prior opioid overdose, fentanyl exposure, and suicidal thoughts during relapse documented", ["negative_discharge_screener", "readiness_conflict", "oud_crisis"]),
        p_case(122, "Probabilistic OUD refused suicide intensity questions", "oud_dual", "missing_invalid_or_stale_evidence", "prob_oud_refused_cssrs", base + timedelta(days=4), False, "3 days", ["refused suicide intensity questions", "residential SUD placement pending", "housing not confirmed"], "assumed high current risk because suicide intensity items were refused and OUD recovery placement remains unresolved", "historical fentanyl exposure, overdose risk, and medication nonadherence documented", ["refused_cssrs", "oud_crisis"], rating_mode="missing"),
        p_case(123, "Probabilistic OUD relapse with housing uncertainty", "oud_dual", "lower_level_of_care_barrier_reasoning", "prob_oud_housing_uncertain", base + timedelta(days=8), False, "4 days", ["housing not confirmed", "opioid cravings not stabilized", "dual diagnosis follow-up pending"], "moderate-to-high current risk because cravings and housing instability make step-down unsafe", "recent opioid relapse crisis with suicidal thoughts documented", ["oud_crisis", "lloc_barriers"]),
        p_case(124, "Probabilistic OUD partial medication response", "oud_dual", "lower_level_of_care_barrier_reasoning", "prob_oud_partial_med_response", base + timedelta(days=12), False, "3 days", ["medication response partial", "residential SUD placement pending", "collateral monitoring not confirmed"], "moderate current risk because medication response and recovery monitoring are not yet stable", "history of fentanyl exposure, cravings, and rapid relapse documented", ["oud_crisis", "lloc_barriers"]),
        p_case(125, "Probabilistic OUD night-shift craving signal", "oud_dual", "lower_level_of_care_barrier_reasoning", "prob_oud_fragmented_craving_signal", base + timedelta(days=16), False, "3 days", ["night shift pacing", "opioid cravings not stabilized", "support person not confirmed"], "moderate current risk that becomes clear only after integrating night shift cravings and support gaps", "prior opioid relapse crisis and unsafe thoughts documented", ["fragmented_shift_notes", "oud_crisis"]),
        p_case(126, "Probabilistic OUD current risk low after support confirmation", "oud_dual", "current_vs_historical_risk", "prob_oud_current_low_history_high", base + timedelta(days=20), True, "0 days", [], "low current risk after cravings improved and supports were confirmed; historical overdose risk remains context only", "prior opioid overdose, fentanyl exposure, and suicidal thoughts during relapse documented", ["current_denial_vs_history", "oud_crisis"]),
        p_case(127, "Probabilistic OUD residential plan confirmed", "oud_dual", "lower_level_of_care_barrier_reasoning", "prob_oud_residential_confirmed", base + timedelta(days=24), True, "0 days", [], "low current risk after residential recovery placement and medication support were confirmed", "opioid relapse crisis and cravings documented earlier in stay", ["lloc_barriers_resolved", "oud_crisis"]),
        p_case(128, "Probabilistic schizoaffective readiness conflict", "schizoaffective", "contradiction", "prob_schizo_negative_screener_not_ready", base + timedelta(days=28), False, "3 days", ["psychiatry note says not ready", "safety plan incomplete", "collateral monitoring not confirmed"], "moderate current risk because internal preoccupation and incomplete safety planning persist despite a negative screener", "admission command hallucinations and historical interrupted attempt documented", ["negative_discharge_screener", "readiness_conflict"]),
        p_case(129, "Probabilistic schizoaffective stale risk reconciliation", "schizoaffective", "missing_invalid_or_stale_evidence", "prob_schizo_stale_risk", base + timedelta(days=32), False, "2 days", ["copied-forward risk text requires reconciliation", "medication response partial", "housing not confirmed"], "moderate current risk; copied-forward high-risk text is stale but current barriers remain", "prior command hallucinations and interrupted attempt documented", ["copied_forward"]),
        p_case(130, "Probabilistic schizoaffective command hallucination monitoring", "schizoaffective", "lower_level_of_care_barrier_reasoning", "prob_schizo_recent_command_ah", base + timedelta(days=36), False, "3 days", ["recent command hallucinations", "medication response partial", "safety plan incomplete"], "moderate-to-high current risk due to recent command hallucinations and incomplete safety plan", "historical command hallucinations and interrupted attempt documented", ["recent_command_ah"]),
        p_case(131, "Probabilistic schizoaffective support gap", "schizoaffective", "lower_level_of_care_barrier_reasoning", "prob_schizo_support_gap", base + timedelta(days=40), False, "2 days", ["collateral monitoring not confirmed", "housing not confirmed", "follow-up appointment pending"], "moderate current risk because partial symptom improvement does not resolve missing support confirmation", "prior psychosis, unsafe thoughts, and treatment nonadherence documented", ["lloc_barriers"]),
        p_case(132, "Probabilistic schizoaffective current denial with history", "schizoaffective", "current_vs_historical_risk", "prob_schizo_current_denial_history", base + timedelta(days=44), True, "0 days", [], "low current risk by final review with historical command hallucination risk documented only as context", "admission command hallucinations and historical interrupted attempt documented", ["current_denial_vs_history"]),
        p_case(133, "Probabilistic schizoaffective PHP support confirmed", "schizoaffective", "current_vs_historical_risk", "prob_schizo_php_confirmed", base + timedelta(days=48), True, "0 days", [], "low current risk after psychosis improved and PHP intake was confirmed", "prior command hallucinations and inpatient safety monitoring documented", ["current_denial_vs_history"]),
        p_case(134, "Probabilistic schizoaffective barriers resolved", "schizoaffective", "lower_level_of_care_barrier_reasoning", "prob_schizo_barriers_resolved", base + timedelta(days=52), True, "0 days", [], "low current risk after safety plan, housing, and collateral monitoring were confirmed", "historical psychosis and unsafe thoughts documented earlier in stay", ["lloc_barriers_resolved"]),
        p_case(135, "Probabilistic psychotic depression discharge conflict", "mdd_psychosis", "contradiction", "prob_mdd_negative_screener_not_ready", base + timedelta(days=56), False, "2 days", ["psychiatry note says not ready", "sleep not stabilized", "collateral monitoring not confirmed"], "moderate current risk because severe depression, poor sleep, and collateral gaps persist despite a negative screener", "prior severe depression with psychotic features and suicidal ideation documented", ["negative_discharge_screener", "readiness_conflict"]),
        p_case(136, "Probabilistic psychotic depression malformed PHQ", "mdd_psychosis", "missing_invalid_or_stale_evidence", "prob_mdd_malformed_phq", base + timedelta(days=60), False, "2 days", ["malformed PHQ-9 requires verification", "sleep not stabilized", "safety plan incomplete"], "moderate current risk based on narrative evidence; malformed PHQ-9 should not be scored as valid", "historical severe depression, hallucinations, and suicidal thoughts documented", ["malformed_score"], rating_mode="malformed"),
        p_case(137, "Probabilistic psychotic depression safety plan gap", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "prob_mdd_safety_plan_gap", base + timedelta(days=64), False, "3 days", ["safety plan incomplete", "collateral monitoring not confirmed", "follow-up appointment pending"], "moderate current risk because safety planning and follow-up reliability remain incomplete", "recurrent crisis presentation after medication nonadherence documented", ["lloc_barriers"]),
        p_case(138, "Probabilistic psychotic depression sleep instability", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "prob_mdd_sleep_instability", base + timedelta(days=68), False, "2 days", ["medication response partial", "sleep not stabilized", "housing not confirmed"], "moderate current risk because medication response, sleep, and housing supports remain unstable", "prior psychotic depression with suicidal ideation documented", ["lloc_barriers"]),
        p_case(139, "Probabilistic psychotic depression current low risk", "mdd_psychosis", "current_vs_historical_risk", "prob_mdd_current_low_history", base + timedelta(days=72), True, "0 days", [], "low current risk after sleep improved and crisis plan was completed", "admission severe depression and suicidal ideation documented for historical context", ["current_denial_vs_history"]),
        p_case(140, "Probabilistic psychotic depression PHP ready", "mdd_psychosis", "current_vs_historical_risk", "prob_mdd_php_ready", base + timedelta(days=76), True, "0 days", [], "low current risk after medication response and PHP intake confirmation", "prior psychotic depression and recurrent crisis presentation documented", ["current_denial_vs_history"]),
        p_case(141, "Probabilistic psychotic depression LLOC barriers resolved", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "prob_mdd_barriers_resolved", base + timedelta(days=80), True, "0 days", [], "low current risk after sleep, collateral monitoring, and follow-up were confirmed", "history of severe depression with psychotic features documented earlier in stay", ["lloc_barriers_resolved"]),
        p_case(142, "Probabilistic BPD negative screener with crisis-plan gap", "bpd", "contradiction", "prob_bpd_negative_screener_plan_gap", base + timedelta(days=84), False, "3 days", ["psychiatry note says not ready", "crisis plan incomplete", "overnight support not confirmed"], "moderate current risk because affective instability and overnight support gaps persist despite a negative screener", "historical self-harm crisis after interpersonal conflict documented", ["negative_discharge_screener", "readiness_conflict"]),
        p_case(143, "Probabilistic BPD missing scores", "bpd", "missing_invalid_or_stale_evidence", "prob_bpd_missing_scores", base + timedelta(days=88), False, "2 days", ["PHQ-9 refused", "support person not confirmed", "crisis plan incomplete"], "moderate current risk based on narrative evidence; refused scores should not be invented", "recent self-harm urges and medication nonadherence documented", ["missing_scores"], rating_mode="missing"),
        p_case(144, "Probabilistic BPD independent ADLs but unsafe plan", "bpd", "lower_level_of_care_barrier_reasoning", "prob_bpd_adl_safety_mismatch", base + timedelta(days=92), False, "2 days", ["independent ADLs do not resolve suicide risk", "overnight coping plan incomplete", "support person not confirmed"], "moderate current risk despite independent ADLs because crisis planning remains unreliable", "history of self-harm urges during interpersonal conflict documented", ["adl_safety_mismatch"]),
        p_case(145, "Probabilistic BPD fragmented shift risk", "bpd", "lower_level_of_care_barrier_reasoning", "prob_bpd_fragmented_shift", base + timedelta(days=96), False, "3 days", ["night shift pacing", "day shift denial conflicts with later unsafe statement", "medication response partial"], "ongoing moderate risk requiring integration of fragmented nursing notes across shifts", "history of impulsive unsafe behavior and self-harm urges documented", ["fragmented_shift_notes"]),
        p_case(146, "Probabilistic BPD current low risk after support confirmation", "bpd", "current_vs_historical_risk", "prob_bpd_current_low_history", base + timedelta(days=100), True, "0 days", [], "low current risk after completed crisis plan and support confirmation", "historical self-harm crisis documented during admission context", ["current_denial_vs_history"]),
        p_case(147, "Probabilistic BPD DBT step-down plan", "bpd", "current_vs_historical_risk", "prob_bpd_dbt_ready", base + timedelta(days=104), True, "0 days", [], "low current risk after DBT-oriented step-down support and crisis contacts were confirmed", "interpersonal conflict and self-harm urges documented earlier in stay", ["current_denial_vs_history"]),
        p_case(148, "Probabilistic BPD crisis-plan barriers resolved", "bpd", "lower_level_of_care_barrier_reasoning", "prob_bpd_barriers_resolved", base + timedelta(days=108), True, "0 days", [], "low current risk with completed crisis plan and overnight support confirmed", "historical affective instability and self-harm crisis documented", ["lloc_barriers_resolved"]),
        p_case(149, "Probabilistic substance-induced refused C-SSRS", "substance_induced", "missing_invalid_or_stale_evidence", "prob_substance_refused_cssrs", base + timedelta(days=112), False, "3 days", ["refused suicide intensity questions", "dual diagnosis follow-up pending", "housing not confirmed"], "assumed high current risk because suicide intensity items were refused while paranoia and placement gaps persist", "stimulant-associated paranoia and unsafe thoughts documented", ["refused_cssrs"], rating_mode="missing"),
        p_case(150, "Probabilistic substance-induced stale intoxication note", "substance_induced", "missing_invalid_or_stale_evidence", "prob_substance_copied_forward", base + timedelta(days=116), False, "2 days", ["copied-forward risk text requires reconciliation", "persistent paranoia after observation", "dual diagnosis follow-up pending"], "moderate current risk; copied-forward intoxication risk is stale but persistent paranoia remains active", "recent stimulant-associated paranoia and missed treatment documented", ["copied_forward", "substance_vs_primary_symptoms"]),
        p_case(151, "Probabilistic substance-induced persistent paranoia", "substance_induced", "lower_level_of_care_barrier_reasoning", "prob_substance_persistent_paranoia", base + timedelta(days=120), False, "4 days", ["persistent paranoia after observation", "dual diagnosis follow-up pending", "housing not confirmed"], "moderate-to-high current risk because paranoia persists after intoxication clears", "stimulant-associated paranoia and unsafe thoughts documented", ["substance_vs_primary_symptoms"]),
        p_case(152, "Probabilistic substance-induced recovery follow-up gap", "substance_induced", "lower_level_of_care_barrier_reasoning", "prob_substance_followup_gap", base + timedelta(days=124), False, "3 days", ["dual diagnosis follow-up pending", "housing not confirmed", "medication monitoring not arranged"], "moderate current risk because recovery follow-up and step-down monitoring are not confirmed", "history of stimulant-associated unsafe thoughts documented", ["lloc_barriers", "substance_vs_primary_symptoms"]),
        p_case(153, "Probabilistic substance-induced current low after clearance", "substance_induced", "current_vs_historical_risk", "prob_substance_current_low", base + timedelta(days=128), True, "0 days", [], "low current risk after paranoia resolved and recovery follow-up was confirmed", "historical stimulant-associated paranoia and unsafe thoughts documented", ["current_denial_vs_history"]),
        p_case(154, "Probabilistic substance-induced recovery supports confirmed", "substance_induced", "current_vs_historical_risk", "prob_substance_supports_confirmed", base + timedelta(days=132), True, "0 days", [], "low current risk with historical intoxication-related unsafe thoughts documented only as context", "prior stimulant-associated paranoia and crisis presentation documented", ["current_denial_vs_history"]),
        p_case(155, "Probabilistic substance-induced LLOC barriers resolved", "substance_induced", "lower_level_of_care_barrier_reasoning", "prob_substance_barriers_resolved", base + timedelta(days=136), True, "0 days", [], "low current risk after dual-diagnosis follow-up, housing, and safety plan were confirmed", "substance-induced mood and psychotic symptoms documented earlier in stay", ["lloc_barriers_resolved"]),
        p_case(156, "Probabilistic bipolar negative screener with mixed symptoms", "bipolar", "contradiction", "prob_bipolar_negative_screener_unready", base + timedelta(days=140), False, "3 days", ["psychiatry note says not ready", "sleep not stabilized", "medication monitoring not arranged"], "moderate current risk because mixed mood symptoms and sleep instability persist despite a negative screener", "history of impulsive unsafe behavior during mixed mood episode documented", ["negative_discharge_screener", "readiness_conflict"]),
        p_case(157, "Probabilistic bipolar copied-forward admission risk", "bipolar", "missing_invalid_or_stale_evidence", "prob_bipolar_copied_forward", base + timedelta(days=144), False, "2 days", ["copied-forward risk text requires reconciliation", "medication response partial", "housing not confirmed"], "moderate current risk; copied-forward admission high-risk text is stale but mixed symptoms and barriers remain", "prior impulsive unsafe behavior and medication nonadherence documented", ["copied_forward"]),
        p_case(158, "Probabilistic bipolar sleep instability", "bipolar", "lower_level_of_care_barrier_reasoning", "prob_bipolar_sleep_instability", base + timedelta(days=148), False, "3 days", ["sleep not stabilized", "medication response partial", "follow-up appointment pending"], "moderate current risk because sleep, medication response, and follow-up remain unstable", "mixed mood episode with impulsive unsafe statements documented", ["lloc_barriers"]),
        p_case(159, "Probabilistic bipolar fragmented shift notes", "bipolar", "lower_level_of_care_barrier_reasoning", "prob_bipolar_fragmented_shift", base + timedelta(days=152), False, "3 days", ["night shift pacing", "day shift denial conflicts with later unsafe statement", "medication monitoring not arranged"], "ongoing moderate risk requiring integration of fragmented shift notes", "history of impulsive behavior during mixed mood episodes documented", ["fragmented_shift_notes"]),
        p_case(160, "Probabilistic bipolar current low after sleep improves", "bipolar", "current_vs_historical_risk", "prob_bipolar_current_low", base + timedelta(days=156), True, "0 days", [], "low current risk after sleep improved and monitoring was confirmed", "historical mixed-episode impulsivity documented for context only", ["current_denial_vs_history"]),
        p_case(161, "Probabilistic bipolar PHP confirmed", "bipolar", "current_vs_historical_risk", "prob_bipolar_php_confirmed", base + timedelta(days=160), True, "0 days", [], "low current risk after medication response and PHP intake confirmation", "prior mixed mood crisis and decreased sleep documented earlier in stay", ["current_denial_vs_history"]),
        p_case(162, "Probabilistic bipolar LLOC barriers resolved", "bipolar", "lower_level_of_care_barrier_reasoning", "prob_bipolar_barriers_resolved", base + timedelta(days=164), True, "0 days", [], "low current risk after medication response, sleep, and outpatient monitoring were confirmed", "history of mixed mood episode with unsafe impulsivity documented", ["lloc_barriers_resolved"]),
        p_case(163, "Probabilistic trauma negative screener with support gap", "trauma_anxiety", "contradiction", "prob_trauma_negative_screener_support_gap", base + timedelta(days=168), False, "3 days", ["psychiatry note says not ready", "overnight support not confirmed", "crisis plan incomplete"], "moderate current risk because nightmares and overnight support gaps persist despite a negative screener", "trauma-related passive suicidal thoughts and panic symptoms documented", ["negative_discharge_screener", "readiness_conflict", "trauma_anxiety"]),
        p_case(164, "Probabilistic trauma missing scores with panic symptoms", "trauma_anxiety", "missing_invalid_or_stale_evidence", "prob_trauma_missing_scores", base + timedelta(days=172), False, "2 days", ["PHQ-9 refused", "crisis plan incomplete", "support person not confirmed"], "moderate current risk based on narrative panic and sleep evidence; missing scores should not be invented", "history of nightmares, hypervigilance, and passive suicidal thoughts documented", ["missing_scores", "trauma_anxiety"], rating_mode="missing"),
        p_case(165, "Probabilistic trauma nightmares and sleep instability", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "prob_trauma_sleep_instability", base + timedelta(days=176), False, "3 days", ["sleep not stabilized", "overnight support not confirmed", "collateral monitoring not confirmed"], "moderate current risk because nightmares, sleep disruption, and support reliability remain unresolved", "trauma-related anxiety and passive suicidal thoughts documented", ["lloc_barriers", "trauma_anxiety"]),
        p_case(166, "Probabilistic trauma hypervigilance and housing uncertainty", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "prob_trauma_housing_uncertain", base + timedelta(days=180), False, "2 days", ["housing not confirmed", "support person not confirmed", "crisis plan incomplete"], "moderate current risk because hypervigilance and step-down supports remain incomplete", "trauma-related panic and suicidal thoughts documented earlier in stay", ["lloc_barriers", "trauma_anxiety"]),
        p_case(167, "Probabilistic trauma follow-up still pending", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "prob_trauma_followup_pending", base + timedelta(days=184), False, "2 days", ["follow-up appointment pending", "overnight support not confirmed", "sleep not stabilized"], "moderate current risk because follow-up and overnight support remain unreliable despite partial improvement", "trauma-related anxiety and passive suicidal thoughts documented", ["lloc_barriers", "trauma_anxiety"]),
        p_case(168, "Probabilistic trauma current low after supports confirmed", "trauma_anxiety", "current_vs_historical_risk", "prob_trauma_current_low", base + timedelta(days=188), True, "0 days", [], "low current risk after nightmares improved, safety plan completed, and supports were confirmed", "historical trauma-related passive suicidal thoughts and panic symptoms documented", ["current_denial_vs_history", "trauma_anxiety"]),
        p_case(169, "Probabilistic trauma therapy follow-up ready", "trauma_anxiety", "current_vs_historical_risk", "prob_trauma_therapy_ready", base + timedelta(days=192), True, "0 days", [], "low current risk after therapy follow-up and overnight support were confirmed", "prior panic episode with passive suicidal thoughts documented", ["current_denial_vs_history", "trauma_anxiety"]),
        p_case(170, "Probabilistic trauma sleep plan barriers resolved", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "prob_trauma_barriers_resolved", base + timedelta(days=196), True, "0 days", [], "low current risk after sleep plan, crisis plan, and support contacts were confirmed", "historical trauma-related anxiety and passive suicidal thoughts documented", ["lloc_barriers_resolved", "trauma_anxiety"]),
    ]


def build_specs_170():
    return build_specs_120() + specs_121_170()


def validate_cases(cases, batch_cases):
    failures = []
    if len(cases) != EXPECTED_TOTAL_CASES:
        failures.append(f"expected {EXPECTED_TOTAL_CASES} total cases, got {len(cases)}")
    if len(batch_cases) != EXPECTED_BATCH_CASES:
        failures.append(f"expected {EXPECTED_BATCH_CASES} batch cases, got {len(batch_cases)}")

    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        failures.append("case IDs are not unique")
    expected_batch_ids = {f"clin_auth_bench_v1_{case_no:04d}" for case_no in range(121, 171)}
    actual_batch_ids = {case["id"] for case in batch_cases}
    if actual_batch_ids != expected_batch_ids:
        failures.append("121-170 batch IDs do not match expected case range")

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
    print(f"Wrote {len(cases)} cumulative ClinAuthBench v1 cases to {CUMULATIVE_OUT_PATH}")
    print(f"Wrote {len(batch_cases)} QA batch cases to {BATCH_OUT_PATH}")
    print(f"Cumulative continued stay: {continued_count}; safe/LLOC-ready: {safe_count}")
    print(f"Batch continued stay: {batch_continued_count}; safe/LLOC-ready: {batch_safe_count}")
    print("Batch MDP models:", dict(sorted(mdp_models.items())))
    print("Cumulative documentation challenges:", dict(sorted(challenge_counts.items())))
    print("Batch documentation challenges:", dict(sorted(batch_challenge_counts.items())))
    print("Cumulative diagnosis families:", dict(sorted(diagnosis_counts.items())))
    print("Batch diagnosis families:", dict(sorted(batch_diagnosis_counts.items())))


def main():
    cases = [build_case(spec) for spec in build_specs_170()]
    batch_cases = [case for case in cases if 121 <= int(case["id"].rsplit("_", 1)[-1]) <= 170]
    failures = validate_cases(cases, batch_cases)
    if failures:
        raise RuntimeError("Generated 170-case dataset failed validation:\n" + "\n".join(failures))

    CUMULATIVE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUMULATIVE_OUT_PATH.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    BATCH_OUT_PATH.write_text(json.dumps(batch_cases, indent=2), encoding="utf-8")
    print_rollup(cases, batch_cases)


if __name__ == "__main__":
    main()
