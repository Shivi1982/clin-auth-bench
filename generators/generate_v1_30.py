"""
Generate the ClinAuthBench v1 30-case calibration set.

This expands the base 10 case specs from mdp_case_builder into a stratified
calibration batch while preserving the finalized v1 JSON field names. The
calibration set enforces the main disposition guardrail: 18/30 cases recommend
continued stay and 12/30 are safe or near-safe for lower level of care.
"""

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from mdp_case_builder import CaseSpec, build_base_specs, build_case


BENCH_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BENCH_DIR / "outputs" / "generated_artifacts" / "synthetic_bh_cases_v1_mdp_30.json"

EXPECTED_CASES = 30
EXPECTED_CONTINUED_STAY = 18
EXPECTED_SAFE_FOR_LLOC = 12
EXPECTED_CONTRADICTION_MIN = 3
EXPECTED_CONTRADICTION_MAX = 4
EXPECTED_OUD_MIN = 4
EXPECTED_TRAUMA_ANXIETY_MIN = 4


def additional_specs():
    base = datetime(2025, 6, 1)
    return [
        CaseSpec(11, "Safe step-down after historical opioid overdose risk", "oud_dual", "current_vs_historical_risk", "historical_high_to_current_safe_stepdown", base, True, "0 days", [], "low current suicide risk with historical opioid overdose risk documented for context only", "prior opioid overdose, fentanyl exposure, and medication nonadherence documented", ["current_denial_vs_history", "oud_crisis"]),
        CaseSpec(12, "Safe step-down with stale admission risk copied forward", "bipolar", "missing_invalid_or_stale_evidence", "stale_copy_forward_but_currently_ready", base + timedelta(days=4), True, "0 days", [], "low current risk after completed safety planning; copied-forward admission risk is stale", "admission high-risk text documented during mixed mood crisis", ["copied_forward"]),
        CaseSpec(13, "Continued stay for psychiatry-social work contradiction", "schizoaffective", "contradiction", "social_work_optimism_psychiatry_not_ready", base + timedelta(days=8), False, "2 days", ["psychiatry note says not ready", "housing not confirmed", "safety plan incomplete"], "moderate current risk because psychiatry documents incomplete safety planning despite optimistic discharge discussion", "prior command hallucinations and interrupted attempt documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(14, "Continued stay for fragmented shift risk and recovery barrier", "substance_induced", "lower_level_of_care_barrier_reasoning", "day_denial_evening_unsafe_statement", base + timedelta(days=12), False, "3 days", ["day shift denial conflicts with later unsafe statement", "sleep not stabilized", "medication monitoring not arranged"], "ongoing moderate risk emerges from later shift documentation after earlier denial", "history of impulsive unsafe behavior during substance-related mood episodes documented", ["fragmented_shift_notes"]),
        CaseSpec(15, "Safe step-down despite malformed PHQ entry", "trauma_anxiety", "missing_invalid_or_stale_evidence", "malformed_score_but_narrative_ready", base + timedelta(days=16), True, "0 days", [], "low current risk based on narrative and completed crisis plan; malformed PHQ-9 should not be scored", "trauma-related passive suicidal thoughts and panic symptoms documented historically", ["malformed_score", "trauma_anxiety"], rating_mode="malformed"),
        CaseSpec(16, "Continued stay after refused C-SSRS and OUD recovery barrier", "oud_dual", "missing_invalid_or_stale_evidence", "refused_cssrs_recovery_barrier", base + timedelta(days=20), False, "3 days", ["refused suicide intensity questions", "residential SUD placement pending", "housing not confirmed"], "assumed high current risk because suicide intensity items were refused and OUD recovery supports remain incomplete", "prior opioid overdose, cravings, and fentanyl exposure documented", ["refused_cssrs", "oud_crisis"], rating_mode="missing"),
        CaseSpec(17, "Safe step-down after current denial and confirmed supports", "schizoaffective", "current_vs_historical_risk", "current_denial_confirmed_supports", base + timedelta(days=24), True, "0 days", [], "low current SI by final review with historical command hallucination risk documented only as context", "admission command hallucinations and interrupted attempt documented", ["current_denial_vs_history"]),
        CaseSpec(18, "Continued stay for trauma nightmares and unresolved overnight support", "trauma_anxiety", "lower_level_of_care_barrier_reasoning", "improved_affect_unresolved_overnight_support", base + timedelta(days=28), False, "2 days", ["overnight support not confirmed", "crisis plan incomplete", "support person not confirmed"], "moderate current risk despite improved affect because nightmares, overnight support, and crisis planning remain unreliable", "trauma-related panic and passive suicidal thoughts documented", ["lloc_barriers", "trauma_anxiety"]),
        CaseSpec(19, "Safe step-down after medication response established", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "symptom_improvement_barriers_resolved", base + timedelta(days=32), True, "0 days", [], "low current risk after medication response, sleep, and collateral supports were documented as adequate for step-down", "recurrent crisis presentation after medication nonadherence documented", ["lloc_barriers_resolved"]),
        CaseSpec(20, "Safe step-down with historical trauma-related SI only", "trauma_anxiety", "current_vs_historical_risk", "historical_si_currently_ready", base + timedelta(days=36), True, "0 days", [], "low current risk with historical trauma-related suicidal thoughts documented but not current", "prior panic episode with passive suicidal thoughts documented", ["current_denial_vs_history", "trauma_anxiety"]),
        CaseSpec(21, "Contradictory discharge readiness after negative OUD screener", "oud_dual", "contradiction", "negative_screener_but_practitioner_not_ready", base + timedelta(days=40), False, "3 days", ["practitioner clarification pending", "residential SUD placement pending", "opioid cravings not stabilized"], "moderate-to-high current risk because OUD cravings and placement barriers persist despite negative discharge screener", "opioid relapse crisis and missed treatment documented", ["negative_discharge_screener", "readiness_conflict", "oud_crisis"]),
        CaseSpec(22, "Safe step-down with copied-forward stale risk", "mdd_psychosis", "missing_invalid_or_stale_evidence", "stale_high_risk_current_low_risk", base + timedelta(days=44), True, "0 days", [], "low current risk after completed crisis plan; copied-forward high-risk note is stale", "admission severe depression with psychotic features documented", ["copied_forward"]),
        CaseSpec(23, "Continued stay for substance symptoms after intoxication clears", "substance_induced", "lower_level_of_care_barrier_reasoning", "persistent_symptoms_after_intoxication", base + timedelta(days=48), False, "4 days", ["persistent paranoia after observation", "dual diagnosis follow-up pending", "housing not confirmed"], "moderate-to-high current risk because psychotic symptoms persist after intoxication clears", "stimulant-associated paranoia and unsafe thoughts documented", ["substance_vs_primary_symptoms"]),
        CaseSpec(24, "Safe step-down after current denial and PHP confirmation", "bipolar", "current_vs_historical_risk", "mixed_episode_history_current_ready", base + timedelta(days=52), True, "0 days", [], "low current risk with historical mixed-episode impulsivity documented for context", "prior impulsive unsafe behavior during mixed mood episode documented", ["current_denial_vs_history"]),
        CaseSpec(25, "Continued stay for readiness barriers across forms", "mdd_psychosis", "lower_level_of_care_barrier_reasoning", "optimistic_note_but_barriers_remain", base + timedelta(days=56), False, "2 days", ["safety plan incomplete", "collateral monitoring not confirmed", "sleep not stabilized"], "moderate current risk because discharge planning remains incomplete despite some improvement", "prior ED presentation for suicidal ideation documented", ["negative_discharge_screener", "lloc_barriers"]),
        CaseSpec(26, "Continued stay for night shift risk after daytime denial", "bpd", "lower_level_of_care_barrier_reasoning", "fragmented_shift_barriers", base + timedelta(days=60), False, "3 days", ["night shift pacing", "day shift denial conflicts with later unsafe statement", "medication response partial"], "ongoing moderate risk requiring integration of fragmented nursing shifts", "history of impulsive unsafe behavior and medication nonadherence documented", ["fragmented_shift_notes"]),
        CaseSpec(27, "Safe step-down with missing anxiety/depression scores but adequate narrative", "trauma_anxiety", "missing_invalid_or_stale_evidence", "missing_scores_currently_ready", base + timedelta(days=64), True, "0 days", [], "low current risk by narrative review and completed safety plan; missing PHQ-9/GAD-7 scores should not be invented", "trauma-related anxiety and historical suicidal thoughts documented", ["missing_scores", "trauma_anxiety"], rating_mode="missing"),
        CaseSpec(28, "Safe step-down despite ADL/safety documentation noise", "bpd", "lower_level_of_care_barrier_reasoning", "adl_independent_and_currently_ready", base + timedelta(days=68), True, "0 days", [], "low current risk with completed safety plan and confirmed supports", "historical self-harm crisis documented", ["adl_safety_mismatch"]),
        CaseSpec(29, "Continued stay after negative screener but unresolved supports", "schizoaffective", "lower_level_of_care_barrier_reasoning", "negative_screener_unresolved_supports", base + timedelta(days=72), False, "3 days", ["safety plan incomplete", "collateral monitoring not confirmed", "medication response partial"], "moderate current risk because negative discharge screener does not resolve incomplete safety supports", "prior interrupted attempt and treatment nonadherence documented", ["negative_discharge_screener", "lloc_barriers"]),
        CaseSpec(30, "Safe step-down after historical opioid and psychosis risk", "oud_dual", "current_vs_historical_risk", "historical_oud_related_psychosis_current_ready", base + timedelta(days=76), True, "0 days", [], "low current risk after cravings improved and step-down recovery supports were confirmed", "historical fentanyl exposure, opioid cravings, and unsafe thoughts documented", ["current_denial_vs_history", "oud_crisis"]),
    ]


def build_specs_30():
    return build_base_specs() + additional_specs()


def validate_batch(cases):
    failures = []
    if len(cases) != EXPECTED_CASES:
        failures.append(f"expected {EXPECTED_CASES} cases, got {len(cases)}")

    safe_count = sum(case["metadata"]["gold"]["safe_for_lloc"] for case in cases)
    continued_count = len(cases) - safe_count
    if continued_count != EXPECTED_CONTINUED_STAY:
        failures.append(f"expected {EXPECTED_CONTINUED_STAY} continued-stay cases, got {continued_count}")
    if safe_count != EXPECTED_SAFE_FOR_LLOC:
        failures.append(f"expected {EXPECTED_SAFE_FOR_LLOC} safe/LLOC-ready cases, got {safe_count}")

    challenge_counts = Counter(case["metadata"]["documentation_challenge"] for case in cases)
    contradiction_count = challenge_counts["contradiction"]
    if not EXPECTED_CONTRADICTION_MIN <= contradiction_count <= EXPECTED_CONTRADICTION_MAX:
        failures.append(
            f"expected {EXPECTED_CONTRADICTION_MIN}-{EXPECTED_CONTRADICTION_MAX} contradiction cases, got {contradiction_count}"
        )

    diagnosis_counts = Counter(case["metadata"]["diagnosis_category"] for case in cases)
    oud_count = diagnosis_counts["Dual diagnosis / OUD-related behavioral health crisis"]
    trauma_count = diagnosis_counts["Trauma/anxiety with suicidality"]
    if oud_count < EXPECTED_OUD_MIN:
        failures.append(f"expected at least {EXPECTED_OUD_MIN} OUD/dual-diagnosis cases, got {oud_count}")
    if trauma_count < EXPECTED_TRAUMA_ANXIETY_MIN:
        failures.append(f"expected at least {EXPECTED_TRAUMA_ANXIETY_MIN} trauma/anxiety cases, got {trauma_count}")

    required_content_markers = [
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
    for case in cases:
        if not case["metadata"]["quality_checks"]["passed"]:
            failures.append(f"{case['id']} failed quality checks")
        if not case["metadata"]["content_gold_checks"]["passed"]:
            failures.append(f"{case['id']} failed content/gold checks")
        for marker in required_content_markers:
            if marker not in case["content"]:
                failures.append(f"{case['id']} missing required content marker: {marker}")

    return failures


def main():
    cases = [build_case(spec) for spec in build_specs_30()]
    failures = validate_batch(cases)
    if failures:
        raise RuntimeError("Generated 30-case calibration set failed validation:\n" + "\n".join(failures))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    safe_count = sum(case["metadata"]["gold"]["safe_for_lloc"] for case in cases)
    continued_count = len(cases) - safe_count
    challenge_counts = Counter(case["metadata"]["documentation_challenge"] for case in cases)
    diagnosis_counts = Counter(case["metadata"]["diagnosis_category"] for case in cases)
    print(f"Wrote {len(cases)} ClinAuthBench v1 calibration cases to {OUT_PATH}")
    print(f"Continued stay: {continued_count}; safe/LLOC-ready: {safe_count}")
    print("Documentation challenges:", dict(sorted(challenge_counts.items())))
    print("Diagnosis families:", dict(sorted(diagnosis_counts.items())))


if __name__ == "__main__":
    main()
