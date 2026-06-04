"""Shared MDP-style case builder for ClinAuthBench.

This module contains the case specs, hidden trajectory transitions, form
rendering, gold-label construction, and consistency checks used by the active
dataset generators. It does not write dataset files directly.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random

from form_templates import (
    discharge_screener,
    final_summary,
    form,
    group_note,
    hp_exam,
    initial_treatment_plan,
    lab_results_summary,
    medication_consent,
    medication_response,
    nursing_assessment,
    nursing_note,
    psych_progress,
    quality_checks,
    rating_forms,
    safet_step1,
    safet_steps25,
    social_work_note,
    treatment_plan_review,
)


@dataclass(frozen=True)
class Profile:
    category: str
    dx: str
    problem: str
    aeb: str
    reason: str
    withdrawal: str
    psychosis_score: str
    has_psychosis: bool
    substances: str
    substance_list: list
    scheduled_medication: str
    prn_medication: str
    medication_consent: str


@dataclass(frozen=True)
class CaseSpec:
    case_no: int
    title: str
    profile_key: str
    documentation_challenge: str
    trajectory: str
    start: datetime
    safe_for_lloc: bool
    expected_los: str
    barriers: list
    current_risk: str
    historical_risk: str
    documentation_challenge_tags: list
    rating_mode: str = "valid"
    mdp_model: str = "rule_based_v1"
    transition_seed: int = 0


@dataclass
class PatientState:
    risk: str = "high"
    sleep: str = "poor"
    medication_adherence: str = "partial"
    medication_response: str = "early"
    engagement: str = "limited"
    safety_plan: str = "incomplete"
    collateral: str = "unconfirmed"
    housing: str = "unconfirmed"
    lloc_readiness: str = "not_ready"


PROFILES = {
    "schizoaffective": Profile(
        category="Schizoaffective disorder with command hallucinations",
        dx="Schizoaffective disorder, depressive type",
        problem="Depressed Mood WITH Psychosis",
        aeb="unsafe thoughts with intermittent command auditory hallucinations and poor sleep",
        reason="worsening depression, command auditory hallucinations, and inability to maintain safety outside the unit",
        withdrawal="Absent; toxicology history reviewed in synthetic record",
        psychosis_score="4 - Moderate impairment with intermittent internal preoccupation",
        has_psychosis=True,
        substances="THC-Marijuana by history; no acute withdrawal observed",
        substance_list=["THC-Marijuana"],
        scheduled_medication="risperidone 2mg BID",
        prn_medication="lorazepam 1mg PRN",
        medication_consent="risperidone 2 MG, 2 mg by mouth twice daily Indication: psychosis/mood stabilization; trazodone 50 MG by mouth at bedtime PRN Indication: insomnia",
    ),
    "mdd_psychosis": Profile(
        category="Major depressive disorder with psychotic features",
        dx="Major Depressive Disorder, recurrent, severe with psychotic features",
        problem="Depressed Mood WITH Psychosis",
        aeb="severe depression, intermittent auditory hallucinations, and limited safety-plan reliability",
        reason="depressed mood with psychotic features and inability to maintain safety without staff support",
        withdrawal="Absent",
        psychosis_score="3 - Moderate impairment with nighttime internal preoccupation",
        has_psychosis=True,
        substances="None reported",
        substance_list=[],
        scheduled_medication="olanzapine 5mg HS",
        prn_medication="hydroxyzine 25mg PRN",
        medication_consent="olanzapine 5 MG by mouth at bedtime Indication: psychosis/mood stabilization; hydroxyzine 25 MG by mouth PRN Indication: anxiety",
    ),
    "bipolar": Profile(
        category="Bipolar mixed episode",
        dx="Bipolar I disorder, current episode mixed, severe",
        problem="Mood Instability WITH Safety Risk",
        aeb="decreased sleep, agitation, impulsive unsafe statements, and inconsistent discharge planning",
        reason="mixed mood symptoms, decreased sleep, impulsivity, and threats of self-harm during conflict",
        withdrawal="Moderate impairment; stimulant and cannabis use reported in synthetic history",
        psychosis_score="2 - Mild impairment; no persistent hallucinations observed after admission",
        has_psychosis=False,
        substances="COC-Cocaine by history; THC-Marijuana by history",
        substance_list=["COC-Cocaine", "THC-Marijuana"],
        scheduled_medication="divalproex ER 1250mg HS and quetiapine 300mg HS",
        prn_medication="hydroxyzine 50mg PRN",
        medication_consent="divalproex ER 1000 MG plus divalproex ER 250 MG by mouth at bedtime Indication: mood stabilization; quetiapine 300 MG by mouth at bedtime Indication: mood stabilization/sleep",
    ),
    "substance_induced": Profile(
        category="Substance-induced mood or psychotic symptoms",
        dx="Substance-induced mood disorder with psychotic symptoms",
        problem="Substance Use WITH Mood/Psychotic Symptoms",
        aeb="recent stimulant use, paranoia, unsafe thoughts, and incomplete recovery plan",
        reason="suicidal ideation and paranoia after stimulant use with symptoms persisting after observation",
        withdrawal="Moderate impairment; cravings and post-acute withdrawal symptoms monitored",
        psychosis_score="3 - Moderate impairment with paranoia after intoxication cleared",
        has_psychosis=True,
        substances="COC-Cocaine by history; ETOH-Alcohol by history",
        substance_list=["COC-Cocaine", "ETOH-Alcohol"],
        scheduled_medication="quetiapine 100mg HS",
        prn_medication="clonidine 0.1mg PRN",
        medication_consent="quetiapine 100 MG by mouth at bedtime Indication: psychosis/mood stabilization; clonidine 0.1 MG PRN Indication: withdrawal/anxiety symptoms",
    ),
    "bpd": Profile(
        category="Borderline personality disorder self-harm crisis",
        dx="Borderline Personality Disorder with acute self-harm crisis",
        problem="Affective Instability WITH Self-Harm Risk",
        aeb="self-harm urges, affective instability, and inability to complete a reliable crisis plan",
        reason="recent self-harm urges, intense interpersonal conflict, and inability to safety plan",
        withdrawal="Absent",
        psychosis_score="1 - No sustained psychosis; stress-related suspiciousness noted",
        has_psychosis=False,
        substances="None reported",
        substance_list=[],
        scheduled_medication="sertraline 100mg daily",
        prn_medication="trazodone 50mg PRN sleep",
        medication_consent="sertraline 100 MG by mouth daily Indication: depression/anxiety; trazodone 50 MG by mouth at bedtime PRN Indication: insomnia",
    ),
    "trauma_anxiety": Profile(
        category="Trauma/anxiety with suicidality",
        dx="Posttraumatic Stress Disorder with severe anxiety and suicidal ideation",
        problem="Anxiety/Trauma Symptoms WITH Safety Risk",
        aeb="nightmares, hypervigilance, panic symptoms, passive suicidal thoughts, and poor sleep",
        reason="trauma-related anxiety, insomnia, and suicidal ideation with unreliable crisis plan use",
        withdrawal="Absent",
        psychosis_score="1 - No sustained psychosis; hypervigilance and intrusive trauma memories noted",
        has_psychosis=False,
        substances="None reported",
        substance_list=[],
        scheduled_medication="sertraline 50mg daily and prazosin 1mg HS",
        prn_medication="hydroxyzine 25mg PRN",
        medication_consent="sertraline 50 MG by mouth daily Indication: PTSD/anxiety; prazosin 1 MG by mouth at bedtime Indication: nightmares; hydroxyzine 25 MG PRN Indication: anxiety",
    ),
    "oud_dual": Profile(
        category="Dual diagnosis / OUD-related behavioral health crisis",
        dx="Opioid Use Disorder, severe, with substance-induced mood symptoms",
        problem="Substance Abuse related to Opioids WITH Mood/Safety Risk",
        aeb="opioid cravings, recent fentanyl exposure, unstable recovery placement, and suicidal thoughts during relapse crisis",
        reason="opioid relapse crisis with suicidal ideation, cravings, and need for residential recovery placement",
        withdrawal="Moderate impairment; opioid cravings and post-acute withdrawal symptoms monitored",
        psychosis_score="2 - Mild impairment; intermittent paranoia during withdrawal stress",
        has_psychosis=False,
        substances="OUD-Opioids by history; fentanyl exposure by history; THC-Marijuana by history",
        substance_list=["OUD-Opioids", "Fentanyl by history", "THC-Marijuana"],
        scheduled_medication="buprenorphine 8mg SL TID and quetiapine 200mg HS",
        prn_medication="clonidine 0.1mg PRN withdrawal symptoms",
        medication_consent="buprenorphine 8 MG sublingual three times daily Indication: opioid use disorder; quetiapine 200 MG by mouth at bedtime Indication: mood/sleep stabilization; clonidine 0.1 MG PRN Indication: withdrawal symptoms",
    ),
}


def profile_to_dict(profile):
    return {
        "category": profile.category,
        "dx": profile.dx,
        "problem": profile.problem,
        "aeb": profile.aeb,
        "reason": profile.reason,
        "withdrawal": profile.withdrawal,
        "psychosis_score": profile.psychosis_score,
        "has_psychosis": profile.has_psychosis,
        "substances": profile.substances,
        "substance_list": profile.substance_list,
        "scheduled_medication": profile.scheduled_medication,
        "prn_medication": profile.prn_medication,
        "medication_consent": profile.medication_consent,
    }


def transition(state, action, spec):
    next_state = PatientState(**state.__dict__)
    if action == "admit_and_observe":
        next_state.risk = "high"
        next_state.sleep = "poor"
        next_state.safety_plan = "not_started"
    elif action == "restart_medication":
        next_state.medication_adherence = "partial"
        next_state.medication_response = "early"
    elif action == "group_and_safety_work":
        next_state.engagement = "variable"
        next_state.safety_plan = "partial"
        if spec.safe_for_lloc:
            next_state.risk = "low_current_high_history"
        elif "refused_cssrs" in spec.documentation_challenge_tags:
            next_state.risk = "assumed_high_refusal"
        elif "recent_command_ah" in spec.documentation_challenge_tags:
            next_state.risk = "moderate_to_high_recent_command_ah"
        else:
            next_state.risk = "moderate"
    elif action == "discharge_planning":
        if spec.safe_for_lloc:
            next_state.safety_plan = "complete"
            next_state.collateral = "confirmed"
            next_state.housing = "confirmed"
            next_state.lloc_readiness = "ready"
        else:
            next_state.collateral = "unconfirmed"
            next_state.housing = "unconfirmed"
            next_state.lloc_readiness = "not_ready"
    return next_state


def apply_state_updates(state, updates):
    next_state = PatientState(**state.__dict__)
    for field_name, value in updates.items():
        setattr(next_state, field_name, value)
    return next_state


def transition_option(outcome, probability, updates, compatible=True):
    return {
        "outcome": outcome,
        "probability": probability,
        "updates": updates,
        "compatible_with_case_target": compatible,
    }


def probabilistic_transition_options(action, spec):
    tags = set(spec.documentation_challenge_tags)
    if action == "admit_and_observe":
        return [
            transition_option(
                "acute_authorization_window_opens",
                1.0,
                {
                    "risk": "high",
                    "sleep": "poor",
                    "safety_plan": "not_started",
                    "lloc_readiness": "not_ready",
                },
            )
        ]

    if action == "restart_medication":
        med_refusal = "med_refusal" in tags
        return [
            transition_option(
                "accepts_medication_with_early_response",
                0.60,
                {"medication_adherence": "partial", "medication_response": "early"},
            ),
            transition_option(
                "accepts_after_staff_education",
                0.25,
                {"medication_adherence": "partial", "medication_response": "early"},
                compatible=not med_refusal,
            ),
            transition_option(
                "initial_refusal_then_partial_adherence",
                0.15,
                {"medication_adherence": "partial", "medication_response": "early"},
                compatible=med_refusal,
            ),
        ]

    if action == "group_and_safety_work":
        refused = "refused_cssrs" in tags
        recent_command_ah = "recent_command_ah" in tags
        barrier_case = not spec.safe_for_lloc and not refused and not recent_command_ah
        return [
            transition_option(
                "current_risk_low_with_historical_risk_context",
                0.52,
                {
                    "risk": "low_current_high_history",
                    "engagement": "variable",
                    "safety_plan": "partial",
                },
                compatible=spec.safe_for_lloc,
            ),
            transition_option(
                "moderate_risk_with_unresolved_barriers",
                0.28,
                {
                    "risk": "moderate",
                    "engagement": "variable",
                    "safety_plan": "partial",
                },
                compatible=barrier_case,
            ),
            transition_option(
                "assumed_high_risk_after_refused_suicide_items",
                0.12,
                {
                    "risk": "assumed_high_refusal",
                    "engagement": "variable",
                    "safety_plan": "partial",
                },
                compatible=refused,
            ),
            transition_option(
                "moderate_to_high_risk_after_recent_command_ah",
                0.08,
                {
                    "risk": "moderate_to_high_recent_command_ah",
                    "engagement": "variable",
                    "safety_plan": "partial",
                },
                compatible=recent_command_ah,
            ),
        ]

    if action == "discharge_planning":
        if spec.safe_for_lloc:
            return [
                transition_option(
                    "ready_after_supports_confirmed",
                    0.72,
                    {
                        "safety_plan": "complete",
                        "collateral": "confirmed",
                        "housing": "confirmed",
                        "lloc_readiness": "ready",
                    },
                ),
                transition_option(
                    "ready_with_stepdown_monitoring",
                    0.20,
                    {
                        "safety_plan": "complete",
                        "collateral": "confirmed",
                        "housing": "confirmed",
                        "lloc_readiness": "ready",
                    },
                ),
                transition_option(
                    "counterfactual_unready_if_supports_failed",
                    0.08,
                    {
                        "collateral": "unconfirmed",
                        "housing": "unconfirmed",
                        "lloc_readiness": "not_ready",
                    },
                    compatible=False,
                ),
            ]
        return [
            transition_option(
                "not_ready_because_barriers_persist",
                0.70,
                {
                    "collateral": "unconfirmed",
                    "housing": "unconfirmed",
                    "lloc_readiness": "not_ready",
                },
            ),
            transition_option(
                "not_ready_because_supports_uncertain",
                0.22,
                {
                    "collateral": "unconfirmed",
                    "housing": "unconfirmed",
                    "lloc_readiness": "not_ready",
                },
            ),
            transition_option(
                "counterfactual_ready_if_supports_confirmed",
                0.08,
                {
                    "safety_plan": "complete",
                    "collateral": "confirmed",
                    "housing": "confirmed",
                    "lloc_readiness": "ready",
                },
                compatible=False,
            ),
        ]

    raise ValueError(f"Unknown MDP action: {action}")


def choose_transition_option(options, rng):
    compatible_options = [option for option in options if option["compatible_with_case_target"]]
    if not compatible_options:
        raise ValueError("No compatible probabilistic transition options available")

    probability_mass = sum(option["probability"] for option in compatible_options)
    roll = rng.random()
    threshold = roll * probability_mass
    cumulative = 0.0
    for option in compatible_options:
        cumulative += option["probability"]
        if threshold <= cumulative:
            return option, roll, probability_mass
    return compatible_options[-1], roll, probability_mass


def build_probabilistic_mdp_trajectory(spec):
    state = PatientState()
    trajectory = []
    seed = spec.transition_seed or (spec.case_no * 7919)
    rng = Random(seed)
    for day, action in enumerate([
        "admit_and_observe",
        "restart_medication",
        "group_and_safety_work",
        "discharge_planning",
    ]):
        state_before = dict(state.__dict__)
        options = probabilistic_transition_options(action, spec)
        selected, roll, probability_mass = choose_transition_option(options, rng)
        state = apply_state_updates(state, selected["updates"])
        trajectory.append({
            "day": day,
            "action": action,
            "transition_model": spec.mdp_model,
            "transition_seed": seed,
            "state_before": state_before,
            "transition_options": options,
            "conditioned_probability_mass": round(probability_mass, 6),
            "sample_roll": round(roll, 6),
            "selected_outcome": selected["outcome"],
            "selected_probability": selected["probability"],
            "state": dict(state.__dict__),
        })
    return trajectory


def build_mdp_trajectory(spec):
    if spec.mdp_model == "probabilistic_v1":
        return build_probabilistic_mdp_trajectory(spec)

    state = PatientState()
    trajectory = []
    for day, action in enumerate([
        "admit_and_observe",
        "restart_medication",
        "group_and_safety_work",
        "discharge_planning",
    ]):
        state = transition(state, action, spec)
        trajectory.append({
            "day": day,
            "action": action,
            "state": dict(state.__dict__),
        })
    return trajectory


def trajectory_state(trajectory, action):
    for step in trajectory:
        if step["action"] == action:
            return step["state"]
    raise ValueError(f"Missing MDP action in trajectory: {action}")


def state_for_spec(spec):
    profile = profile_to_dict(PROFILES[spec.profile_key])
    return {
        "case_no": spec.case_no,
        "id": f"clin_auth_bench_v1_{spec.case_no:04d}",
        "title": spec.title,
        "patient_code": f"CAB-V1-{spec.case_no:04d}",
        "start": spec.start,
        "profile": profile,
        "trajectory": spec.trajectory,
        "current_risk": spec.current_risk,
        "historical_risk": spec.historical_risk,
        "barriers": spec.barriers,
        "expected_los": spec.expected_los,
        "safe_for_lloc": spec.safe_for_lloc,
        "documentation_challenge_tags": spec.documentation_challenge_tags,
        "mdp_model": spec.mdp_model,
    }


def psych_progress_for_spec(state, dt, spec, discharge_state):
    if discharge_state["lloc_readiness"] == "ready":
        return psych_progress(state, dt, ready_conflict=False)
    if spec.documentation_challenge == "contradiction":
        return psych_progress(state, dt, ready_conflict=True)
    return psych_progress(state, dt, ready_conflict="readiness_conflict" in spec.documentation_challenge_tags)


def rating_forms_for_spec(state, dt, spec):
    return rating_forms(
        state,
        dt,
        missing=spec.rating_mode == "missing",
        malformed=spec.rating_mode == "malformed",
    )


def social_work_ready_note(state, dt):
    return form("Discharge Planning / Social Work Barrier Note", dt, " | ".join([
        "Discharge Planning Contact: synthetic support person confirmed for medication storage and overnight monitoring",
        "Housing: confirmed safe step-down setting",
        "Transportation: confirmed",
        "Outpatient follow-up: PHP intake confirmed",
        "Patient preference: agrees to step-down and can describe crisis contacts",
        "Social Work Narrative: Patient completed discharge safety review, identified two supports, and confirmed follow-up plan. No current inpatient-level discharge barrier documented on this review.",
        "Insurance/UR note: step-down appears appropriate after final practitioner review",
    ]))


def nursing_flowsheet_snapshot(state, dt, spec, group_state, discharge_state):
    profile = state["profile"]
    ready_for_lloc = discharge_state["lloc_readiness"] == "ready"
    is_oud = "OUD" in profile["category"] or "Opioid" in profile["dx"]
    is_trauma = "Trauma" in profile["category"]
    has_psychosis = profile["has_psychosis"]
    group_refused = spec.documentation_challenge == "missing_invalid_or_stale_evidence"

    if "refused_cssrs" in spec.documentation_challenge_tags:
        suicide_ideation = "Patient Refused - Assume High Risk"
        suicidal_behavior = "Patient Refused - Assume High Risk"
    elif ready_for_lloc:
        suicide_ideation = "Denies"
        suicidal_behavior = "Denies/None reported"
    else:
        suicide_ideation = "Denies this shift; prior unsafe thoughts remain under review"
        suicidal_behavior = "Denies/None reported"

    if is_trauma:
        mood = "Anxious"
        cognition = "Distracted"
        sleep = "Hours of sleep/Night:3-4"
        sleep_doc = "Interrupted by nightmares and hypervigilance"
    elif "Bipolar" in profile["dx"]:
        mood = "Irritable"
        cognition = "Tangential"
        sleep = "Hours of sleep/Night:2-4"
        sleep_doc = "Slept in short intervals with pacing"
    elif is_oud:
        mood = "Anxious"
        cognition = "Distracted"
        sleep = "Hours of sleep/Night:4-5"
        sleep_doc = "Restless sleep with cravings reported"
    elif has_psychosis:
        mood = "Depressed"
        cognition = "Disorganized"
        sleep = "Hours of sleep/Night:3-5"
        sleep_doc = "Fragmented sleep; internally preoccupied at times"
    else:
        mood = "Irritable"
        cognition = "Distracted"
        sleep = "Hours of sleep/Night:5-6" if ready_for_lloc else "Hours of sleep/Night:3-4"
        sleep_doc = "Appeared to rest with staff checks" if ready_for_lloc else "Restless sleep and frequent awakening"

    adls = "Independent" if ready_for_lloc else "Independent with prompting for hygiene"
    medication_compliance = (
        "Initially refused scheduled dose; accepted after education"
        if "med_refusal" in spec.documentation_challenge_tags
        else "Compliant with all medications"
    )
    hallucination_field = (
        "Describe the Hallucinations: Auditory; command content earlier in stay" if has_psychosis
        else "Describe the Hallucinations: Denies AVH this shift"
    )
    substance_plan = (
        "Problem: Substance Abuse related to Opioids as evidenced by opioid use disorder, cravings, and buprenorphine treatment | Recovery placement: residential SUD placement pending"
        if is_oud
        else "Problem: Substance Use reviewed in synthetic record; no acute withdrawal requiring medical transfer"
    )
    trauma_plan = (
        "Problem: Anxiety/Trauma Symptoms as evidenced by nightmares, hypervigilance, panic symptoms, and passive suicidal thoughts"
        if is_trauma
        else ""
    )
    psych_plan = (
        f"Problem: {profile['problem']} as evidenced by {profile['aeb']}"
    )
    narrative_parts = [
        psych_plan,
        substance_plan,
        trauma_plan,
        "Problem: Altered Sleep Pattern related to insomnia as evidenced by medication regimen and disrupted overnight sleep",
        "Discharge/Living Situation: " + ("confirmed step-down support" if ready_for_lloc else "housing or overnight support not confirmed"),
    ]
    narrative = " | ".join(part for part in narrative_parts if part)

    fields = [
        "Any Falls for This Shift?: No",
        "Active Medical Problems on MTP: Yes, medical problems present",
        "Active Medical Problems: Stable with no acute medical complaint voiced",
        "Breathing: No observed or reported complaints",
        "Cardiovascular: No observed or reported complaints",
        "Urine: Normal",
        "Bladder Functions: Continent",
        "Bowel: Normal",
        "Last BM: Last BM" + dt.strftime("%Y%m%d"),
        "Ambulation: Steady",
        "Activities of Daily Living: " + adls,
        "Fall Assessment: Gait steady",
        "Suicidal Behavior: " + suicidal_behavior,
        "Falls Precautions: No",
        "Appetite: Good" if ready_for_lloc else "Appetite: Fair",
        "Skin: Normal",
        "Education Provided: coping skills, medication education, and sleep hygiene",
        "Approximately what time did the patient go to sleep?: Time:2300",
        "Hours of sleep/Night:: " + sleep,
        "Document Sleep: " + sleep_doc,
        "Does the patient have pain?: No" if not is_oud else "Does the patient have pain?: Yes, withdrawal-related body aches reported",
        "Medication Compliance: " + medication_compliance,
        "Medication Side Effects Observed: No",
        "MD Notified: No",
        "Document findings:: visibility, medication compliance, and cooperation with staff redirection",
        "Progress toward treatment (Medical Problems): Yes, making progress, as evidenced by:",
        "Participating in groups: " + ("No" if group_refused else "Yes"),
        "Not Participating in Groups: Reason For Not Participatingrefused" if group_refused else "Group participation: attended with prompting",
        "Narrative Note Instructions:: Plan of Care" + narrative,
        "Current Shift Assignment: Night Shift <br> 12 hours (7pm- 7am)",
        "Appearance: " + ("Disheveled" if not ready_for_lloc else "Neat, Well-Groomed"),
        "Orientation: Situation",
        "Progress toward treatment (Psych Problem): Yes, making progress, as evidenced by:",
        "Speech:: " + ("Poor articulation" if is_oud else "Normal rate/volume"),
        "Eye Contact: Fair" if not ready_for_lloc else "Eye Contact: Good",
        "Affect: " + ("Flat" if mood == "Anxious" else "Blunted"),
        "Mood: " + mood,
        "Behavior: " + ("Isolative/Withdrawn" if not ready_for_lloc else "Appropriate"),
        "Cognition/Thought Content/Thought Process: " + cognition,
        hallucination_field,
        "Aggression:: No aggression",
        "Suicide Ideation: " + suicide_ideation,
    ]
    return form("Nursing Note - Structured Shift Flowsheet", dt, " | ".join(fields))


def render_forms(spec, state, trajectory):
    start = spec.start
    group_state = trajectory_state(trajectory, "group_and_safety_work")
    discharge_state = trajectory_state(trajectory, "discharge_planning")
    ready_for_lloc = discharge_state["lloc_readiness"] == "ready"
    refused = "refused_cssrs" in spec.documentation_challenge_tags
    negative_discharge_screen = (
        ready_for_lloc
        or "negative_discharge_screener" in spec.documentation_challenge_tags
        or "current_denial_vs_history" in spec.documentation_challenge_tags
    )
    current_denial = group_state["risk"] == "low_current_high_history"
    forms = [
        initial_treatment_plan(state, start.replace(hour=18, minute=29)),
        nursing_assessment(state, start.replace(hour=21, minute=10)),
        safet_step1(state, start.replace(hour=22, minute=0), refused=refused, current_denial=current_denial),
        safet_steps25(state, start.replace(hour=22, minute=18)),
        hp_exam(state, start + timedelta(days=1, hours=6, minutes=20)),
        lab_results_summary(state, start + timedelta(days=1, hours=7, minutes=5)),
        medication_consent(state, start + timedelta(days=1, hours=9, minutes=35)),
        nursing_note(
            state,
            start + timedelta(days=1, hours=7),
            "Day Shift 12 hours (7am-7pm)",
            "Depressed",
            "Patient ate breakfast in room, accepted check-in, and stated distress was lower than admission. Staff documented that symptom improvement did not yet prove independent safety outside the unit.",
        ),
        group_note(
            state,
            start + timedelta(days=1, hours=14),
            "activity",
            group_state["engagement"] == "variable" and spec.documentation_challenge != "missing_invalid_or_stale_evidence",
            "Patient participated with prompting and identified one coping step." if spec.documentation_challenge != "missing_invalid_or_stale_evidence" else "Patient was invited twice and declined. Alternative worksheet left with patient.",
            mood="Irritable" if spec.documentation_challenge == "missing_invalid_or_stale_evidence" else "Appropriate",
        ),
        medication_response(state, start + timedelta(days=1, hours=16), accepted="med_refusal" not in spec.documentation_challenge_tags, prn=True),
        nursing_note(
            state,
            start + timedelta(days=1, hours=19),
            "Night Shift 12 hours (7pm-7am)",
            "Depressed",
            "Patient slept in short intervals and paced near the nurses station. Denied active SI during brief check but later asked staff to stay nearby if unsafe thoughts returned.",
        ),
        nursing_flowsheet_snapshot(
            state,
            start + timedelta(days=1, hours=23, minutes=30),
            spec,
            group_state,
            discharge_state,
        ),
        *rating_forms_for_spec(state, start + timedelta(days=2, hours=8), spec),
        group_note(
            state,
            start + timedelta(days=2, hours=10),
            "process",
            True,
            "Patient practiced one communication statement, then became quiet when discharge planning was discussed.",
            mood="Appropriate",
        ),
        nursing_note(
            state,
            start + timedelta(days=2, hours=12),
            "Day Shift",
            "Appropriate",
            "Patient appeared brighter and asked about step-down. Staff reviewed crisis plan; patient named one coping skill but could not consistently identify who would help with medication storage.",
            addendum="Anticipate discharge planning discussion if practitioner agrees; clinical readiness not yet determined.",
        ),
    ]
    if "copied_forward" in spec.documentation_challenge_tags:
        forms.append(nursing_note(
            state,
            start + timedelta(days=2, hours=13),
            "Day Shift Addendum",
            "Depressed",
            "COPY FORWARD from admission: high suicide risk with unsafe thoughts. Current addendum: patient denies plan this hour but remains guarded and needs cueing to use coping card.",
        ))
    forms.extend([
        psych_progress_for_spec(state, start + timedelta(days=2, hours=15), spec, discharge_state),
        medication_response(state, start + timedelta(days=2, hours=20), accepted=True, prn=False),
        nursing_note(
            state,
            start + timedelta(days=2, hours=23),
            "Evening Shift",
            "Appropriate" if ready_for_lloc else "Irritable",
            "Patient completed crisis plan review, denied SI/HI/AVH, and identified two supports for step-down." if ready_for_lloc else "Patient stated, 'I can say the plan here, but I do not know if it will work when I am by myself.' Staff stayed until calm.",
        ),
        treatment_plan_review(state, start + timedelta(days=3, hours=8)),
        social_work_ready_note(state, start + timedelta(days=3, hours=9))
        if ready_for_lloc
        else social_work_note(state, start + timedelta(days=3, hours=9), optimistic=spec.documentation_challenge == "contradiction"),
        group_note(
            state,
            start + timedelta(days=3, hours=11),
            "safety",
            True,
            "Patient completed the safety plan worksheet and named two supports and a follow-up appointment." if ready_for_lloc else "Patient completed part of a safety plan worksheet but left blank the overnight support section.",
            mood="Appropriate",
        ),
        discharge_screener(state, start + timedelta(days=3, hours=12), negative=negative_discharge_screen, refused=refused),
        final_summary(state, start + timedelta(days=3, hours=15)),
    ])
    return forms


def evidence_anchors(spec):
    anchors = [
        {
            "gold_field": "current_suicide_risk",
            "supporting_form": "Psychiatric Progress Note",
            "evidence_hint": "Risk discussion",
        },
        {
            "gold_field": "safe_for_lloc",
            "supporting_form": "Discharge Planning / Social Work Barrier Note",
            "evidence_hint": "Housing: unstable or not confirmed" if not spec.safe_for_lloc else "support",
        },
    ]
    for barrier in spec.barriers:
        anchors.append({
            "gold_field": "lower_level_of_care_barriers",
            "barrier": barrier,
            "supporting_form": "Discharge Planning / Social Work Barrier Note",
        })
    return anchors


def gold_for_spec(spec, state, forms):
    profile = state["profile"]
    return {
        "current_suicide_risk": spec.current_risk,
        "historical_suicide_risk": spec.historical_risk,
        "primary_diagnosis": profile["dx"],
        "psychosis_present": profile["has_psychosis"],
        "substance_use": profile["substance_list"],
        "safe_for_lloc": spec.safe_for_lloc,
        "expected_los_recommendation": spec.expected_los,
        "lower_level_of_care_barriers": spec.barriers,
        "documentation_challenge": spec.documentation_challenge,
        "documentation_challenge_tags": spec.documentation_challenge_tags,
        "key_evidence_forms": [
            "SAFE-T Columbia SSRS - Recent (Step 1)",
            "SAFE-T Columbia SSRS - Recent (Steps 2-5)",
            "Psychiatric Progress Note",
            "Discharge Planning / Social Work Barrier Note",
            "Interdisciplinary Treatment Plan Review",
        ],
        "evidence_anchors": evidence_anchors(spec),
        "do_not_claim": [
            "Do not confuse historical suicide risk with current risk.",
            "Do not conclude discharge readiness from one favorable note or a negative discharge screener alone.",
            "Do not invent missing or malformed rating scores.",
            "Do not ignore unresolved lower-level-of-care barriers.",
        ],
        "form_count": len(forms),
    }


def content_gold_checks(case):
    content = case["content"]
    gold = case["metadata"]["gold"]
    final_state = trajectory_state(case["metadata"]["mdp_trajectory"], "discharge_planning")
    checks = []
    checks.append({
        "check": "trajectory_lloc_matches_gold",
        "passed": (final_state["lloc_readiness"] == "ready") == gold["safe_for_lloc"],
    })
    for form_name in gold["key_evidence_forms"]:
        checks.append({
            "check": f"key_form_present:{form_name}",
            "passed": f"FORM: {form_name}" in content,
        })
    for barrier in gold["lower_level_of_care_barriers"]:
        tokens = [token for token in re.split(r"\W+", barrier.lower()) if len(token) >= 5]
        checks.append({
            "check": f"barrier_supported:{barrier}",
            "passed": any(token in content.lower() for token in tokens),
        })
    if "current_denial_vs_history" in gold["documentation_challenge_tags"]:
        checks.append({
            "check": "current_vs_historical_markers",
            "passed": "No current SI on this screen" in content and "historical interrupted attempt" in content,
        })
    if "recent_command_ah" in gold["documentation_challenge_tags"]:
        checks.append({
            "check": "recent_command_ah_markers",
            "passed": "recent command hallucinations" in content and "continued inpatient monitoring remains clinically indicated" in content,
        })
    if gold["safe_for_lloc"]:
        checks.append({
            "check": "safe_lloc_support_markers",
            "passed": "PHP intake confirmed" in content and "No current inpatient-level discharge barrier documented" in content,
        })
    if gold["documentation_challenge"] == "contradiction":
        checks.append({
            "check": "conflicting_readiness_markers",
            "passed": "Anticipate discharge planning discussion" in content and "not clinically ready" in content,
        })
    if gold["documentation_challenge"] == "missing_invalid_or_stale_evidence":
        checks.append({
            "check": "invalid_missing_stale_markers",
            "passed": (
                "Not documented" in content
                or "exceeds valid PHQ-9 range" in content
                or "COPY FORWARD" in content
                or "Patient Refused - Assume High Risk" in content
            ),
        })
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def build_case(spec):
    state = state_for_spec(spec)
    trajectory = build_mdp_trajectory(spec)
    forms = render_forms(spec, state, trajectory)
    case = {
        "id": state["id"],
        "title": state["title"],
        "metadata": {
            "benchmark": "ClinAuthBench",
            "version": "v1",
            "synthetic": True,
            "privacy_design": "structured_synthetic_state_no_real_patient_text",
            "level_of_care": "Inpatient Adult Psychiatric",
            "documentation_window_hours": 72,
            "diagnosis_category": state["profile"]["category"],
            "trajectory": state["trajectory"],
            "documentation_challenge": spec.documentation_challenge,
            "documentation_challenge_tags": spec.documentation_challenge_tags,
            "mdp_model": spec.mdp_model,
            "mdp_trajectory": trajectory,
            "gold": gold_for_spec(spec, state, forms),
        },
        "content": "\n\n".join(forms),
    }
    case["metadata"]["quality_checks"] = quality_checks(case)
    case["metadata"]["content_gold_checks"] = content_gold_checks(case)
    return case


def build_base_specs():
    base = datetime(2025, 4, 1)
    return [
        CaseSpec(1, "Current denial with high historical psychosis risk", "schizoaffective", "current_vs_historical_risk", "high_to_current_denial_with_history", base, True, "0 days", [], "low current SI by final screener with elevated historical risk requiring documentation review", "admission command hallucinations and prior interrupted attempt documented", ["current_denial_vs_history"]),
        CaseSpec(2, "Discharge screener negative but psychiatry not ready", "mdd_psychosis", "contradiction", "high_to_conflicting_readiness", base + timedelta(days=4), False, "2 days", ["psychiatry note says not ready", "safety plan incomplete", "collateral monitoring not confirmed"], "moderate current risk due to unresolved safety-plan reliability despite negative discharge screener", "prior medication nonadherence and recurrent crisis presentation documented", ["negative_discharge_screener", "readiness_conflict"]),
        CaseSpec(3, "Improving mood with unresolved LLOC barriers", "bpd", "lower_level_of_care_barrier_reasoning", "high_to_partial_improvement_barriers", base + timedelta(days=8), False, "3 days", ["overnight support not confirmed", "crisis plan incomplete", "follow-up appointment pending"], "improving but continued risk when alone because crisis plan and supports remain incomplete", "history of self-harm urges during interpersonal conflict documented", ["lloc_barriers"]),
        CaseSpec(4, "Refused C-SSRS and missing rating scores", "substance_induced", "missing_invalid_or_stale_evidence", "refusal_and_missing_scores", base + timedelta(days=12), False, "3 days", ["refused suicide intensity questions", "PHQ-9 refused", "recovery placement not confirmed"], "assumed high current suicide risk due to refused C-SSRS intensity items", "prior stimulant-associated paranoia and unsafe thoughts documented", ["refused_cssrs"], rating_mode="missing"),
        CaseSpec(5, "Copied-forward high risk with current moderate risk", "bipolar", "missing_invalid_or_stale_evidence", "stale_copy_forward_reconciliation", base + timedelta(days=16), False, "2 days", ["copied-forward risk text requires reconciliation", "medication response partial", "housing not confirmed"], "moderate current risk; copied-forward admission high-risk text is stale but current barriers remain", "history of impulsive unsafe behavior with medication nonadherence documented", ["copied_forward"]),
        CaseSpec(6, "Malformed PHQ-9 with persistent LLOC barriers", "mdd_psychosis", "missing_invalid_or_stale_evidence", "malformed_score_with_barriers", base + timedelta(days=20), False, "2 days", ["malformed PHQ-9 requires verification", "collateral monitoring not confirmed", "sleep not stabilized"], "moderate current risk based on narrative evidence; PHQ-9 value is invalid and should not be treated as valid", "prior ED presentation for suicidal ideation documented", ["malformed_score"], rating_mode="malformed"),
        CaseSpec(7, "Substance symptoms persist after intoxication clears", "substance_induced", "lower_level_of_care_barrier_reasoning", "intoxication_to_persistent_symptoms", base + timedelta(days=24), False, "4 days", ["persistent paranoia after observation", "dual diagnosis follow-up pending", "housing not confirmed"], "moderate-to-high current risk because paranoia and unsafe thoughts persist after intoxication clears", "cocaine-associated paranoia and missed treatment documented", ["substance_vs_primary_symptoms"]),
        CaseSpec(8, "Fragmented nursing notes reveal risk over shifts", "bipolar", "contradiction", "fragmented_shift_risk", base + timedelta(days=28), False, "3 days", ["night shift pacing", "day shift denial conflicts with later unsafe statement", "medication monitoring not arranged"], "ongoing moderate risk emerges only when fragmented shift notes are integrated", "history of impulsive behavior during mixed mood episodes documented", ["fragmented_shift_notes"]),
        CaseSpec(9, "Independent ADLs but safety plan unreliable", "bpd", "lower_level_of_care_barrier_reasoning", "adl_independent_safety_impaired", base + timedelta(days=32), False, "2 days", ["independent ADLs do not resolve suicide risk", "overnight coping plan incomplete", "support person not confirmed"], "moderate current risk despite independent ADLs because safety planning remains unreliable", "history of self-harm crisis after interpersonal conflict documented", ["adl_safety_mismatch"]),
        CaseSpec(10, "Psychosis improves but command hallucination risk remains", "schizoaffective", "current_vs_historical_risk", "partial_improvement_recent_command_ah", base + timedelta(days=36), False, "3 days", ["recent command hallucinations", "safety plan incomplete", "medication response partial"], "moderate-to-high current risk due to recent command hallucinations and incomplete safety plan", "prior interrupted attempt and treatment nonadherence documented", ["recent_command_ah"]),
    ]
