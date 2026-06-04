"""Shared synthetic form-rendering helpers for ClinAuthBench.

This module contains reusable generalized inpatient documentation templates used
by the MDP case builder. It does not call external models and does not copy real
sample text.
"""

import re
from collections import Counter
from datetime import datetime, timedelta

FORBIDDEN_SAMPLE_TERMS = {
    "synthetic_source_patient_name",
    "synthetic_source_facility_name",
    "synthetic_source_staff_name",
    "synthetic_source_identifier",
    "aspx?pdfid",
}

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "url": re.compile(r"https?://|www\.|aspx\?|pdfid=", re.I),
    "mrn_like": re.compile(r"\b(?:MRN|Account|Encounter|Claim)\s*[:#]\s*[A-Z0-9-]{4,}\b", re.I),
}


def ymd(dt):
    return dt.strftime("%Y%m%d")


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def form(name, dt, body):
    return f"FORM: {name} | CREATION_DATE: {iso(dt)}\n{body}"


def join_fields(fields):
    return " | ".join(fields)


def make_case_state(
    case_no,
    title,
    profile,
    start,
    trajectory,
    current_risk,
    historical_risk,
    barriers,
    expected_los,
    safe_for_lloc,
    traps,
):
    return {
        "case_no": case_no,
        "id": f"template_case_{case_no:03d}",
        "title": title,
        "patient_code": f"SYNTH-BH-{case_no:03d}",
        "start": start,
        "profile": profile,
        "trajectory": trajectory,
        "current_risk": current_risk,
        "historical_risk": historical_risk,
        "barriers": barriers,
        "expected_los": expected_los,
        "safe_for_lloc": safe_for_lloc,
        "traps": traps,
    }


def initial_treatment_plan(state, dt):
    profile = state["profile"]
    return form("Initial Nursing Treatment Plan", dt, join_fields([
        f"Admission: Time of Admission{dt.strftime('%H%M')}",
        f"Admission: Date of Admission{ymd(dt)}",
        "Admission Suicide Risk Screening has been reviewed and the Overall Risk Level Score is noted as:: High",
        f"Psychiatric Problem: {profile['problem']}",
        f"{profile['problem']}: AEB: {profile['aeb']}",
        f"{profile['problem']}: GOAL: Patient will not exhibit self harm behaviors while in hospital",
        f"{profile['problem']}: INTERVENTION: Query patient regarding thoughts of self-harm at least q waking shift",
        f"{profile['problem']}: INTERVENTION: Encourage patient to seek staff when urges, voices, or unsafe thoughts increase",
        "High Suicide Risk: GOAL: Patient will not attempt suicide while in hospital.",
        "High Suicide Risk: INTERVENTION: Assign room to support increased visual/auditory monitoring",
        "High Suicide Risk: INTERVENTION: Q 5 Minute Observation until practitioner modifies order",
        "High Suicide Risk: INTERVENTION: May not use sharps except by order and under supervision",
        "High Suicide Risk: INTERVENTION: Add patient to safety huddle and initiate Crisis Safety Plan",
        "High Suicide Risk: INTERVENTION: Patient will be assigned a roommate when clinically appropriate",
    ]))


def nursing_assessment(state, dt):
    profile = state["profile"]
    fields = [
        "1. Physical Signs: Mild or intermittent impairment",
        "2. Substance Abuse/Withdrawal: " + profile["withdrawal"],
        "3. Mood Related Signs: 5 - Severe impairment noted in affect regulation and hopelessness",
        "4. Anxiety: Moderate impairment; patient reports restlessness and difficulty settling",
        "5. Behavioral Disturbance: Mild to moderate impairment with intermittent pacing",
        "6. Psychosis: " + profile["psychosis_score"],
        "Admission Status: Voluntary per orders, Request Type: Routine continued stay review",
        "Appearance: Disheveled",
        "Behavior: Cooperative with intermittent guardedness",
        "Cognition/Thought Content/Thought Process: Distracted",
        "Cognition/Thought Content/Thought Process: Thought blocking observed during risk questions",
        "Considering your total clinical experience with this population, how mentally ill is the patient at this time?: 5 - Markedly ill",
        "Current Living Arrangements: unstable; patient cannot identify a reliable adult for overnight monitoring",
        "Emergency Contact: Patient unable to confirm phone number; collateral pending",
        f"Event(s) which led client to seek treatment at this time:: synthetic adult presents for {profile['reason']}.",
        "The individual reports unsafe thoughts were stronger before admission but gives inconsistent answers about current intent.",
        "Per record, individual has history of stopping medication after discharge and returning to crisis services.",
        "Learning Barriers: impaired concentration, poor recall of safety steps, and internal preoccupation",
        "Patient reports learning readiness level as:: Apprehensive/Tentative",
        "Patient's Anticipated Discharge Plan: step-down when risk, medication response, and safety plan are clinically stable",
        "Psychomotor: Ambulatory; pacing noted on unit",
        "Sleep/Rest Patterns: usual # of hours per night 6-7",
        "Sleep/Rest Patterns: 3-4 hours over prior night with frequent awakening",
        "Substance: " + profile["substances"],
        "Triggers that cause you to be upset or out of control: perceived rejection, nighttime voices, and conflict with family",
        "What happened in the last 72 hours which led to the precipitating crisis/chief complaint?: worsening mood, unsafe thoughts, and limited ability to use coping plan.",
    ]
    return form("Nursing Assessment", dt, join_fields(fields))


def safet_step1(state, dt, refused=False, current_denial=False):
    if refused:
        fields = [
            "1) Wish to be dead:: Patient Refused - Assume High Risk",
            "2) Current suicidal thoughts:: Patient Refused - Assume High Risk",
            "3) Suicidal thoughts w/ Method (w/no specific Plan or Intent or act): Patient Refused - Assume High Risk",
            "4) Suicidal Intent without Specific Plan:: Patient Refused - Assume High Risk",
            "5) Intent with Plan:: Patient Refused - Assume High Risk",
            "C-SSRS Suicidal Behavior Lifetime: Yes - prior preparatory behavior documented in synthetic record",
            "Is this patient a Moderate or High suicide risk?: Yes",
        ]
    elif current_denial:
        fields = [
            "1) Wish to be dead:: No",
            "2) Current suicidal thoughts:: No",
            "3) Suicidal thoughts w/ Method (w/no specific Plan or Intent or act): No",
            "4) Suicidal Intent without Specific Plan:: No",
            "5) Intent with Plan:: No",
            "C-SSRS Suicidal Behavior Lifetime: Yes - historical interrupted attempt; not a current endorsement",
            "Is this patient a Moderate or High suicide risk?: No current SI on this screen; history and collateral still require review",
        ]
    else:
        fields = [
            "1) Wish to be dead:: Yes - Low Risk",
            "2) Current suicidal thoughts:: Yes - Low Risk",
            "3) Suicidal thoughts w/ Method (w/no specific Plan or Intent or act): Yes - Moderate Risk",
            "4) Suicidal Intent without Specific Plan:: Yes - High Risk",
            "5) Intent with Plan:: patient did not provide details and became tearful",
            "C-SSRS Suicidal Behavior Lifetime: Yes - prior interrupted attempt documented in synthetic record",
            "Is this patient a Moderate or High suicide risk?: Yes",
        ]
    return form("SAFE-T Columbia SSRS - Recent (Step 1)", dt, join_fields(fields))


def safet_steps25(state, dt):
    profile = state["profile"]
    fields = [
        "Access to lethal methods: denies firearm access; access to medications requires collateral confirmation",
        "Activating Events: recent psychosocial stressor and unstable discharge supports",
        "Brief Evaluation Summary: Risk indicators present",
        "Brief Evaluation Summary: Warning signs present",
        "Clinical Status: Hopelessness",
        "Clinical Status: Agitation or severe anxiety",
        "Clinical Status: Highly impulsive behavior",
        "Clinical Status: Command hallucinations to hurt self" if profile["has_psychosis"] else "Clinical Status: affective instability with self-harm urges",
        "Clinical Status: Substance abuse or dependence" if profile["substances"] != "None reported" else "Clinical Status: no acute withdrawal observed",
        "Protective Factors: identifies one sibling as reason for living but cannot confirm contact",
        "Internal Factors - Other: limited coping reliability when distressed",
        "Is the patient willing to participate and willing to answer questions about Suicidal Ideation intensity?: Yes, with pauses and redirection",
        f".: Date{ymd(dt)}",
        f"Name of Psychiatric Practitioner Notified of Risk Level.: Time{dt.strftime('%H%M')}",
        "Relevant Mental Status Information: guarded, sleep-deprived, and requires close monitoring for changes in safety statements.",
        "Risk Level: High Suicide Risk",
        "Risk Stratification: High Suicide Risk: recent suicidal ideation and impaired ability to safety plan",
        "Select appropriate triage associated with above risk stratification:: High Suicide Risk: continued observation and practitioner review",
        "Treatment History: non-adherence after prior discharge and missed outpatient appointment",
    ]
    return form("SAFE-T Columbia SSRS - Recent (Steps 2-5)", dt, join_fields(fields))


def hp_exam(state, dt):
    profile = state["profile"]
    return form("History and Physical Examination", dt, join_fields([
        f"{dt.strftime('%m/%d/%Y %H:%M')} Glucose 91 mg/dL",
        f"{dt.strftime('%m/%d/%Y %H:%M')} BUN/Creatinine Ratio 18.4",
        f"{dt.strftime('%m/%d/%Y %H:%M')} Albumin 4.0 g/dL",
        f"{dt.strftime('%m/%d/%Y %H:%M')} Absolute Lymphocyte 2.14 k/uL",
        "Lungs: clear on auscultation bilaterally",
        "Lymph Nodes: none palpated",
        "Medical Diagnoses: hypertension; chronic low back pain",
        "Military Experience: patient denies military service",
        "Mobility:: Ambulatory",
        "Neuromuscular: WNL",
        "O2 Sat: 98",
        "Past physical/medical trauma: not discussed in this medical screen",
        f"Plan of Care Summary: admitted for {profile['dx']} and safety stabilization; medical team to follow blood pressure and sleep complaints.",
        "Preferred Pronouns: synthetic patient uses they/them in this generated record",
        "Pulse: 92",
        "Respiration: 16",
        "Skin: no rash or wound observed",
        "Social History:: outpatient follow-up inconsistent; no real identifiers included",
        "Temperature: 98.1",
        "Vitals: within acceptable range for unit monitoring",
        "Who is completing the form?: Advanced Practice Provider - SYNTHETIC_CLINICIAN",
    ]))


def lab_results_summary(state, dt):
    profile = state["profile"]
    prior = dt - timedelta(days=1, hours=2)
    current = dt
    if profile["substance_list"]:
        uds_components = [
            ("Dr Amp/Methamp", "Detected" if "COC-Cocaine" in profile["substance_list"] else "Not Detected", None),
            ("Ur Barbiturate", "Not Detected", None),
            ("Ur Benzodiazepine Screen", "Not Detected", None),
            ("Ur Cannabinoid", "Detected" if "THC-Marijuana" in profile["substance_list"] else "Not Detected", None),
            ("Ur Cocaine", "Detected" if "COC-Cocaine" in profile["substance_list"] else "Not Detected", None),
            ("Ur Fentanyl", "Not Detected", None),
            ("Ur Methadone", "Not Detected", None),
            ("Ur Opiates", "Not Detected", None),
            ("Ur PCP", "Not Detected", None),
            ("Ethanol", "<9", "mg/dL"),
        ]
    else:
        uds_components = [
            ("Dr Amp/Methamp", "Not Detected", None),
            ("Ur Barbiturate", "Not Detected", None),
            ("Ur Benzodiazepine Screen", "Not Detected", None),
            ("Ur Cannabinoid", "Not Detected", None),
            ("Ur Cocaine", "Not Detected", None),
            ("Ur Fentanyl", "Not Detected", None),
            ("Ur Methadone", "Not Detected", None),
            ("Ur Opiates", "Not Detected", None),
            ("Ur PCP", "Not Detected", None),
            ("Ethanol", "<9", "mg/dL"),
        ]
    fields = [
        f"CBC: WBC 6.18 x10e3/mcL Collection Date/Time {prior.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CBC: WBC 6.74 x10e3/mcL Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CBC: RBC 4.28 x10e6/mcL Collection Date/Time {prior.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CBC: Hgb 12.2 g/dL Collection Date/Time {prior.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CBC: Hct 38.1 % Collection Date/Time {prior.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CBC: Platelet 286 x10e3/mcL Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: Glucose Level 104 mg/dL Collection Date/Time {prior.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: Sodium 137 mmol/L Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: Potassium 3.5 mmol/L Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: Chloride 105 mmol/L Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: CO2 23 mmol/L Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: BUN 18 mg/dL Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: Creatinine 1.120 mg/dL Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: AST 28 U/L Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        f"CMP: ALT 42 U/L Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC",
        "Acetaminophen Level: <2.0 mcg/mL",
        "Laboratory Interpretation: no acute medical instability identified; continue medication monitoring as clinically indicated",
    ]
    for name, value, unit in uds_components:
        rendered_unit = f" {unit}" if unit else ""
        fields.append(f"Urine Drug Screen Component: {name} {value}{rendered_unit} Collection Date/Time {current.strftime('%m/%d/%Y %H:%M')} UTC")
    return form("Laboratory Results Summary", current, join_fields(fields))


def medication_consent(state, dt):
    profile = state["profile"]
    fields = [
        "Is the patient revoking consent of a medication?: No",
        "Medication Consent: Medication(s):" + profile["medication_consent"],
        "Medication Consent Discussion: risks, benefits, alternatives, sedation, metabolic effects, and need for follow-up labs reviewed in plain language.",
        "Patient response: signed after asking whether medication would make the voices stop tonight; education provided that response may be gradual.",
        f"Signatures: Date{ymd(dt)}",
        f"Signatures: Time{dt.strftime('%H%M')}",
        "Signatures: Signature of Patient or Legal Representative - SYNTHETIC_SIGNATURE_ON_FILE",
        "Staff Witness: SYNTHETIC_STAFF",
    ]
    return form("Medication Consent", dt, join_fields(fields))


def nursing_note(state, dt, shift, mood, body, addendum=None):
    fields = [
        "Active Medical Problems: stable with no acute complaint voiced" if "Day" in shift else "Active Medical Problems: no new medical complaint observed",
        "Activities of Daily Living: independent with prompting for hygiene" if mood != "Appropriate" else "Activities of Daily Living: independent",
        "Affect: " + ("Blunted" if mood in {"Depressed", "Irritable"} else "Appropriate"),
        "Aggression:: No aggression",
        "Ambulation: steady",
        "Any Falls for This Shift?: No",
        "Appearance: Disheveled",
        "Appetite: fair" if mood != "Appropriate" else "Appetite: good",
        "Behavior: Cooperative",
        "Behavior: guarded during safety questions" if mood != "Appropriate" else "Behavior: appropriate",
        "Cognition/Thought Content/Thought Process: Focused at times",
        "Cognition/Thought Content/Thought Process: Distracted when unit became loud",
        "Current Shift Assignment: " + shift,
        "Document findings:: compliant with medications as ordered" if "accepted" in body.lower() or "medication" in body.lower() else "Document findings:: participating in plan of care with staff prompting",
        "Eye Contact: Fair",
        "Fall Assessment: gait steady",
        "Last BM: Last BM" + ymd(dt),
        "Medication Compliance: Compliant with all medications" if "refused" not in body.lower() else "Medication Compliance: initially refused then accepted after education",
        "Medication Side Effects Observed: No",
        "Mood: " + mood,
        "Narrative Note Instructions:: Plan of Care Problem: suicide risk and impaired safety awareness. " + body,
        "Orientation: Person",
        "Orientation: Place",
        "Orientation: Situation",
        "Pain: denies acute pain" if "back" not in body.lower() else "Pain: reports chronic lower back discomfort; non-pharmacologic intervention offered",
        "Speech:: soft but coherent",
    ]
    if addendum:
        fields.insert(3, "Addendum: " + addendum)
    return form("Nursing Note", dt, join_fields(fields))


def group_note(state, dt, group_type, attended, response, mood="Appropriate"):
    title = {
        "activity": "Game Day",
        "process": "Passive, Aggressive, and Assertive Communication",
        "safety": "Coping Skills and Crisis Planning",
        "recovery": "Relapse Prevention and Medication Follow-up",
    }[group_type]
    focus = {
        "activity": "Cultivate communication skills, teamwork, positive socialization and creative problem solving through structured team activities.",
        "process": "Participants learn differences between passive, aggressive, and assertive communication styles and practice active listening.",
        "safety": "Participants identify early warning signs, coping steps, and who to contact before unsafe thoughts escalate.",
        "recovery": "Participants discuss medication adherence, relapse warning signs, and barriers to outpatient follow-up.",
    }[group_type]
    fields = [
        "Affect: " + ("Blunted" if mood != "Appropriate" else "Appropriate"),
        "Behavior: Cooperative" if attended else "Behavior: walked in hallway and declined invitation",
        "Cognition/Thought Content/Thought Process: Goal-Directed" if attended else "Cognition/Thought Content/Thought Process: Distracted",
        "Describe the patient response to intervention and progress: " + response,
        f"GROUP NOTE:: Date:{ymd(dt)}",
        "GROUP NOTE:: Group Focus:" + focus,
        f"GROUP NOTE:: START TIME:{dt.strftime('%H%M')}",
        f"GROUP NOTE:: END TIME:{(dt + timedelta(minutes=50)).strftime('%H%M')}",
        "GROUP NOTE:: Group Title:" + title,
        "GROUP TYPE:: Process" if group_type in {"process", "safety", "recovery"} else "GROUP TYPE:: Activity",
        "INTERVENTIONS:: Encouraged patient to attend session",
        "INTERVENTIONS:: Education",
        "Mood: " + mood,
        "Participation Level: " + ("Active" if attended and "shared" in response.lower() else "Minimal" if attended else "None/Did Not Attend Group"),
        "Reason for Not Attending Group:: refused" if not attended else "Patient response to interventions and summary of progress: Some Progress",
        "ALTERNATIVE ACTIVITY OFFERED DUE TO REFUSAL OF GROUP: Hand Out" if not attended else "Speech: Normal rate/volume",
        "Did patient accept the alternative activity offered?: Yes" if not attended else "Treatment Plan: Problem: safety planning and symptom stabilization",
    ]
    return form("Activity Therapy Group Progress Note" if group_type == "activity" else "Social Worker Group Progress Note", dt, join_fields(fields))


def medication_response(state, dt, accepted=True, prn=False):
    profile = state["profile"]
    fields = [
        "Scheduled medication: " + profile["scheduled_medication"],
        "Medication adherence: " + ("accepted with encouragement" if accepted else "initially refused scheduled dose; accepted after brief education and quiet room intervention"),
        "PRN medication: " + (profile["prn_medication"] + " administered for anxiety/agitation" if prn else "offered, patient declined"),
        "Medication education: patient verbalized partial understanding and asked for written medication list",
        "Side effects: no rigidity, rash, or acute dystonia observed",
        "Response: partial response; sleep and intensity of distress remain under monitoring",
        "Nursing follow-up: continue to monitor sedation, orthostasis, safety statements, and group tolerance",
    ]
    return form("Medication Administration / Treatment Response Note", dt, join_fields(fields))


def score_label_phq(total):
    if total >= 20:
        return "Severe Depression"
    if total >= 15:
        return "Moderately Severe Depression"
    if total >= 10:
        return "Moderate Depression"
    if total >= 5:
        return "Mild Depression"
    return "Minimal Depression"


def score_label_gad(total):
    if total >= 15:
        return "Severe Anxiety"
    if total >= 10:
        return "Moderate Anxiety"
    if total >= 5:
        return "Mild Anxiety"
    return "Minimal Anxiety"


def rating_score_profile(state):
    case_no = state.get("case_no", 0)
    category = state.get("profile", {}).get("category", "")
    tags = set(state.get("documentation_challenge_tags", []))
    safe_for_lloc = state.get("safe_for_lloc", False)

    if safe_for_lloc:
        phq_profiles = [
            [1, 1, 2, 2, 1, 1, 1, 0, 0],
            [2, 2, 2, 2, 1, 1, 2, 1, 0],
            [2, 2, 3, 2, 2, 2, 1, 1, 1],
            [1, 2, 2, 2, 1, 1, 1, 1, 0],
        ]
        gad_profiles = [
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1, 2],
            [2, 2, 2, 2, 1, 1, 1],
        ]
    else:
        phq_profiles = [
            [3, 3, 3, 3, 2, 3, 2, 2, 2],
            [3, 3, 2, 3, 2, 2, 2, 1, 2],
            [2, 3, 3, 3, 2, 3, 2, 2, 3],
            [2, 2, 3, 2, 2, 2, 2, 2, 1],
            [3, 3, 3, 2, 2, 2, 3, 2, 2],
        ]
        gad_profiles = [
            [3, 3, 2, 2, 2, 2, 2],
            [3, 2, 3, 2, 2, 2, 3],
            [2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 2, 3, 2, 3],
            [2, 3, 2, 3, 2, 2, 2],
        ]

    if "Trauma/anxiety" in category:
        gad_profiles = [
            [3, 3, 3, 2, 2, 2, 3],
            [3, 3, 2, 3, 2, 2, 3],
            [2, 3, 3, 2, 2, 2, 2],
        ] if not safe_for_lloc else [
            [2, 2, 2, 2, 1, 1, 2],
            [2, 2, 2, 1, 1, 1, 2],
        ]
    if "Major depressive" in category:
        phq_profiles = [
            [3, 3, 3, 3, 3, 3, 2, 2, 2],
            [3, 3, 3, 3, 2, 3, 3, 2, 2],
            [3, 3, 2, 3, 2, 3, 2, 2, 2],
        ] if not safe_for_lloc else [
            [2, 2, 3, 2, 2, 2, 2, 1, 1],
            [2, 2, 2, 2, 1, 2, 1, 1, 0],
        ]
    if "current_denial_vs_history" in tags or safe_for_lloc:
        phq_profiles = [scores[:-1] + [min(scores[-1], 1)] for scores in phq_profiles]

    return (
        phq_profiles[case_no % len(phq_profiles)],
        gad_profiles[case_no % len(gad_profiles)],
    )


def rating_forms(state, dt, missing=False, malformed=False):
    phq_items = [
        "1. Little interest or pleasure in doing things",
        "2. Feeling down, depressed, or hopeless",
        "3. Trouble falling or staying asleep, or sleeping too much",
        "4. Feeling tired or having little energy",
        "5. Poor appetite or overeating",
        "6. Feeling bad about yourself",
        "7. Trouble concentrating",
        "8. Moving or speaking slowly, or being fidgety/restless",
        "9. Thoughts that you would be better off dead or of hurting yourself",
    ]
    gad_items = [
        "1. Feeling nervous, anxious, or on edge",
        "2. Not being able to stop or control worrying",
        "3. Worrying too much about different things",
        "4. Trouble relaxing",
        "5. Being so restless that it is hard to sit still",
        "6. Becoming easily annoyed or irritable",
        "7. Feeling afraid as if something awful might happen",
    ]

    if missing:
        phq_lines = [
            "PHQ-9 item responses:",
            f"{phq_items[0]}: 2",
            f"{phq_items[1]}: 2",
            f"{phq_items[2]}: Not documented - patient declined before item completion",
            f"{phq_items[3]}: Not documented - patient declined before item completion",
            f"{phq_items[4]}: Not documented - patient declined before item completion",
            f"{phq_items[5]}: Not documented - patient declined before item completion",
            f"{phq_items[6]}: Not documented - patient declined before item completion",
            f"{phq_items[7]}: Not documented - patient declined before item completion",
            f"{phq_items[8]}: Not documented. Item 9: Not documented.",
            "PHQ-9 Total Score: Not documented.",
            "Reason: patient stated questions were making them more upset.",
        ]
        gad_lines = [
            "GAD-7 item responses:",
            *[f"{item}: Not documented - patient declined administration" for item in gad_items],
            "GAD-7 Total Score: Not documented.",
            "Reason: patient refusal and need for de-escalation. No numeric score documented.",
        ]
        return [
            form("PHQ-9 Assessment", dt, "\n".join(phq_lines)),
            form("GAD-7 Assessment", dt + timedelta(minutes=12), "\n".join(gad_lines)),
        ]
    if malformed:
        phq_scores = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        _, gad_scores = rating_score_profile(state)
        gad_total = sum(gad_scores)
        phq_lines = [
            "PHQ-9 item responses:",
            *[f"{item}: {score}" for item, score in zip(phq_items, phq_scores)],
            "PHQ-9 Total Score: 31 (entered value exceeds valid PHQ-9 range; requires verification).",
            "Calculated item sum from documented item responses: 27.",
            "Item 9: 3.",
        ]
        gad_lines = [
            "GAD-7 item responses:",
            *[f"{item}: {score}" for item, score in zip(gad_items, gad_scores)],
            f"GAD-7 Total Score: {gad_total} ({score_label_gad(gad_total)}).",
            f"Patient endorsed feeling afraid something awful might happen: {gad_scores[6]}.",
        ]
        return [
            form("PHQ-9 Assessment", dt, "\n".join(phq_lines)),
            form("GAD-7 Assessment", dt + timedelta(minutes=12), "\n".join(gad_lines)),
        ]
    phq_scores, gad_scores = rating_score_profile(state)
    phq_total = sum(phq_scores)
    gad_total = sum(gad_scores)
    phq_lines = [
        "PHQ-9 item responses:",
        *[f"{item}: {score}" for item, score in zip(phq_items, phq_scores)],
        f"PHQ-9 Total Score: {phq_total} ({score_label_phq(phq_total)}).",
        f"Item 9 (suicidal ideation): {phq_scores[8]}.",
        "Patient paused before answering and accepted staff support during scoring." if phq_scores[8] else "Patient denied item 9 on this administration.",
    ]
    gad_lines = [
        "GAD-7 item responses:",
        *[f"{item}: {score}" for item, score in zip(gad_items, gad_scores)],
        f"GAD-7 Total Score: {gad_total} ({score_label_gad(gad_total)}).",
        f"Patient endorsed feeling afraid something awful might happen: {gad_scores[6]}.",
    ]
    return [
        form("PHQ-9 Assessment", dt, "\n".join(phq_lines)),
        form("GAD-7 Assessment", dt + timedelta(minutes=12), "\n".join(gad_lines)),
    ]


def treatment_plan_review(state, dt):
    return form("Interdisciplinary Treatment Plan Review", dt, join_fields([
        "Problem: suicide risk and impaired safety awareness",
        "Problem status: active",
        "Goal progress: partial; no self-harm behavior on unit but safety plan not reliably usable without prompts",
        "Nursing summary: patient accepts checks but gives variable answers about ability to remain safe after discharge",
        "Therapy summary: group attendance inconsistent; patient benefits from direct prompts and quiet environment",
        "Medication summary: scheduled medication restarted; response partial and sleep remains below baseline",
        "Family/collateral summary: support person not yet confirmed for monitoring or medication storage",
        "Plan revision: continue observation, medication monitoring, safety plan rehearsal, and discharge coordination",
    ]))


def social_work_note(state, dt, optimistic=False):
    if optimistic:
        plan = "Patient asked whether discharge could happen tomorrow and identified a shelter option; social work explained placement and follow-up are not confirmed."
    else:
        plan = "Discharge barriers remain active: " + "; ".join(state["barriers"]) + ". Patient cannot yet describe what to do if unsafe thoughts return at night."
    fields = [
        "Discharge Planning Contact: synthetic collateral attempt placed; no real phone number stored",
        "Housing: unstable or not confirmed",
        "Transportation: not confirmed",
        "Outpatient follow-up: referral started; appointment pending",
        "Patient preference: wants step-down but becomes tearful when asked to review nighttime coping steps",
        "Social Work Narrative: " + plan,
        "Insurance/UR note: continued stay review requires synthesis of safety risk, medication response, and discharge supports",
    ]
    return form("Discharge Planning / Social Work Barrier Note", dt, join_fields(fields))


def psych_progress(state, dt, ready_conflict=False):
    profile = state["profile"]
    if ready_conflict:
        readiness = "Psychiatry assessment: not clinically ready for lower level today because safety plan is incomplete, sleep remains unstable, and collateral monitoring is not confirmed."
    elif state["safe_for_lloc"]:
        readiness = "Psychiatry assessment: may transition to PHP after final crisis plan review and confirmed follow-up."
    else:
        readiness = "Psychiatry assessment: continued inpatient monitoring remains clinically indicated due to variable safety statements and incomplete transition supports."
    fields = [
        f"Assessment: {profile['dx']}.",
        "Mental Status: disheveled, guarded, cooperative, speech soft, thought process intermittently blocked.",
        "Risk discussion: patient denies intent during interview but acknowledges unsafe thoughts can return quickly when alone.",
        "Medication plan: continue " + profile["scheduled_medication"] + "; monitor sedation, orthostasis, adherence, and symptom response.",
        "Sleep: 3.5 to 5 hours, fragmented; not yet at baseline.",
        "Treatment response: partial improvement in agitation but coping reliability remains limited.",
        readiness,
        "Plan: maintain current observation level pending practitioner review, repeat safety planning, and coordinate step-down supports.",
    ]
    return form("Psychiatric Progress Note", dt, join_fields(fields))


def discharge_screener(state, dt, negative=True, refused=False):
    if refused:
        answer = "Patient Refused - Assume High Risk"
        method = "Patient Refused - Assume High Risk"
        intent = "Patient Refused - Assume High Risk"
        plan = "Patient Refused - Assume High Risk"
        behavior = "Patient Refused - Assume High Risk"
        note = "Patient refused discharge screener items; form instruction requires high-risk assumption."
    elif negative:
        answer = "No"
        behavior = "No"
        note = "Negative screen documented during brief discharge review; compare with full 72-hour risk packet and practitioner note."
    else:
        answer = "Yes"
        method = "Yes - patient described nonspecific overdose thoughts without a worked-out plan"
        intent = "No - denied intention to act after leaving the hospital but remained unable to complete safety planning independently"
        plan = "No - no detailed plan documented on this screener"
        behavior = "No"
        note = "Positive item 2 screen documented; questions 3-5 were completed before question 6 and practitioner notification was required."
    fields = [
        f"1) While you were here in the hospital, have you wished you were dead or wished you could go to sleep and not wake up?: {answer}",
        f"2) While you were here in the hospital, have you actually had thoughts about killing yourself?: {answer}",
    ]
    if refused or not negative:
        fields.extend([
            f"3) Have you been thinking about how you might kill yourself?: {method}",
            f"4) Have you had these thoughts and had some intention of acting on them or do you have some intention of acting on them after you leave the hospital?: {intent}",
            f"5) Have you started to work out or worked out the details of how to kill yourself either while in the hospital or after discharge, and do you intend to carry out this plan?: {plan}",
        ])
    fields.extend([
        f"6) While you were here in the hospital, have you done anything, started to do anything, or prepared to do anything to end your life?: {behavior}",
        "Was the patient present on day of discharge?: Yes",
        note,
    ])
    return form("C-SSRS - Discharge Screener", dt, join_fields(fields))


def final_summary(state, dt):
    if state["safe_for_lloc"]:
        med_note = "Medical necessity note: step-down may be appropriate after final review because current risk screen is negative and supports are documented."
    else:
        med_note = "Medical necessity note: continued inpatient care is supported when risk monitoring, medication response, and transition barriers remain unresolved."
    return form("Interdisciplinary Treatment Summary", dt, join_fields([
        "Primary diagnosis: " + state["profile"]["dx"],
        "Problem: suicide risk, mood instability, and impaired safety awareness",
        "Goal progress: partial; patient uses coping steps with staff prompts but not independently",
        "Nursing summary: observation continued; patient remains cooperative but inconsistent when discussing safety outside the unit",
        "Therapy summary: attendance variable with some benefit when present",
        "Medication summary: partial response; no serious side effects observed",
        "Discharge summary: lower level of care depends on confirmed follow-up, safe housing, and reliable crisis plan use",
        med_note,
    ]))


def build_forms(state):
    start = state["start"]
    forms = [
        initial_treatment_plan(state, start.replace(hour=18, minute=29)),
        nursing_assessment(state, start.replace(hour=21, minute=10)),
        safet_step1(state, start.replace(hour=22, minute=0), refused="refusal" in state["traps"]),
        safet_steps25(state, start.replace(hour=22, minute=18)),
        hp_exam(state, start + timedelta(days=1, hours=6, minutes=20)),
        lab_results_summary(state, start + timedelta(days=1, hours=7, minutes=5)),
        medication_consent(state, start + timedelta(days=1, hours=9, minutes=35)),
        nursing_note(
            state,
            start + timedelta(days=1, hours=7),
            "Day Shift 12 hours (7am-7pm)",
            "Depressed",
            "Patient ate breakfast in room, accepted check-in, and stated voices were quieter but still present near bedtime. Patient denied current plan but asked staff not to leave them alone when hallway became loud.",
        ),
        group_note(
            state,
            start + timedelta(days=1, hours=14),
            "activity",
            False,
            "Patient was invited twice, walked briefly near group room, and declined. Alternative worksheet left with patient.",
            mood="Irritable",
        ),
        medication_response(state, start + timedelta(days=1, hours=16), accepted=False, prn=True),
        nursing_note(
            state,
            start + timedelta(days=1, hours=19),
            "Night Shift 12 hours (7pm-7am)",
            "Depressed",
            "Patient slept in short intervals, paced near nurses station, and reported coping card was hard to remember. Denied active SI during routine check but later asked whether staff would know if voices returned.",
        ),
        *rating_forms(
            state,
            start + timedelta(days=2, hours=8),
            missing="missing_scores" in state["traps"],
            malformed="malformed_score" in state["traps"],
        ),
        group_note(
            state,
            start + timedelta(days=2, hours=10),
            "process",
            True,
            "Patient participated with prompting and practiced one assertive communication statement, then became quiet when discharge was discussed.",
            mood="Appropriate",
        ),
        nursing_note(
            state,
            start + timedelta(days=2, hours=12),
            "Day Shift",
            "Appropriate",
            "Patient appeared brighter after lunch and stated they wanted to leave soon. Staff reviewed medication schedule and crisis plan; patient named one coping skill but could not identify who would monitor medications.",
            addendum="Anticipate discharge planning discussion if practitioner agrees; clinical readiness not yet determined.",
        ),
        psych_progress(state, start + timedelta(days=2, hours=15), ready_conflict="readiness_conflict" in state["traps"]),
        medication_response(state, start + timedelta(days=2, hours=20), accepted=True, prn=False),
        nursing_note(
            state,
            start + timedelta(days=2, hours=23),
            "Evening Shift",
            "Irritable",
            "Patient was cooperative but frustrated after phone call. Patient stated, 'I can say the plan here, but I do not know if it will work when I am by myself.' Staff remained with patient until calm.",
        ),
        treatment_plan_review(state, start + timedelta(days=3, hours=8)),
        social_work_note(state, start + timedelta(days=3, hours=9), optimistic="readiness_conflict" in state["traps"]),
        group_note(
            state,
            start + timedelta(days=3, hours=11),
            "safety",
            True,
            "Patient completed part of a safety plan worksheet but left blank the section for who to call overnight.",
            mood="Appropriate",
        ),
        discharge_screener(state, start + timedelta(days=3, hours=12), negative=True, refused="refusal" in state["traps"]),
        final_summary(state, start + timedelta(days=3, hours=15)),
    ]
    if "copied_forward" in state["traps"]:
        forms.insert(15, nursing_note(
            state,
            start + timedelta(days=2, hours=18),
            "Evening Shift Addendum",
            "Depressed",
            "COPY FORWARD from admission risk plan: high suicide risk with unsafe thoughts. Current addendum: patient denies plan this hour, remains guarded, and requires staff cueing to use coping card.",
        ))
    return forms


def gold_for_case(state, forms):
    return {
        "current_suicide_risk": state["current_risk"],
        "historical_suicide_risk": state["historical_risk"],
        "primary_diagnosis": state["profile"]["dx"],
        "psychosis_present": state["profile"]["has_psychosis"],
        "substance_use": state["profile"]["substance_list"],
        "safe_for_lloc": state["safe_for_lloc"],
        "expected_los_recommendation": state["expected_los"],
        "lower_level_of_care_barriers": state["barriers"],
        "intentional_traps": state["traps"],
        "key_evidence_forms": [
            "SAFE-T Columbia SSRS - Recent (Step 1)",
            "SAFE-T Columbia SSRS - Recent (Steps 2-5)",
            "Psychiatric Progress Note",
            "Discharge Planning / Social Work Barrier Note",
            "Interdisciplinary Treatment Plan Review",
        ],
        "do_not_claim": [
            "Do not conclude discharge readiness from one optimistic nursing or social work note.",
            "Do not treat a negative discharge screener as complete risk clearance without reconciling the full packet.",
            "Do not invent missing collateral, housing, medication-monitoring, PHQ-9, or GAD-7 details.",
        ],
        "form_count": len(forms),
    }


def safety_scan(text):
    findings = []
    lower = text.lower()
    for term in FORBIDDEN_SAMPLE_TERMS:
        if term in lower:
            findings.append({"type": "forbidden_sample_term", "value": term})
    for label, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"type": label, "value": match.group(0)[:80]})
    return findings


def quality_checks(case):
    content = case["content"]
    forms = re.findall(r"^FORM: (.*?) \| CREATION_DATE:", content, flags=re.M)
    repeated = [name for name, count in Counter(forms).items() if count > 1]
    rich_narrative_markers = [
        "Narrative Note Instructions::",
        "Social Work Narrative:",
        "Psychiatry assessment:",
        "Treatment response:",
        "Describe the patient response to intervention and progress:",
    ]
    rich_count = sum(content.count(marker) for marker in rich_narrative_markers)
    source_mentions_gold_answer = "Expected continued stay recommendation" in content
    findings = safety_scan(content)
    return {
        "passed": (
            not findings
            and 12 <= len(forms) <= 25
            and len(repeated) >= 2
            and rich_count >= 3
            and not source_mentions_gold_answer
        ),
        "form_count": len(forms),
        "repeated_form_types": repeated,
        "rich_narrative_marker_count": rich_count,
        "pii_phi_findings": findings,
        "source_mentions_gold_answer": source_mentions_gold_answer,
    }


def build_case(state):
    forms = build_forms(state)
    case = {
        "id": state["id"],
        "title": state["title"],
        "metadata": {
            "benchmark": "ClinAuthBench",
            "version": "template_helper_v0",
            "synthetic": True,
            "privacy_design": "style_only_synthetic_no_real_patient_text",
            "level_of_care": "Inpatient Adult Psychiatric",
            "documentation_window_hours": 72,
            "diagnosis_category": state["profile"]["category"],
            "trajectory": state["trajectory"],
            "gold": gold_for_case(state, forms),
        },
        "content": "\n\n".join(forms),
    }
    case["metadata"]["quality_checks"] = quality_checks(case)
    return case

