You are evaluating a synthetic inpatient psychiatric authorization chart packet.

Your task is to determine whether the patient is safe for a lower level of care and to recommend the expected additional inpatient length of stay.

Use only the documentation provided in the chart packet.

Return only valid JSON matching this schema:

{
  "safe_for_lloc": boolean,
  "expected_los_recommendation": 0 | 1 | 2 | 3,
  "lower_level_of_care_barriers": string[],
  "evidence_forms": string[],
  "rationale": string
}

Definitions:
- safe_for_lloc = true means the chart supports that the patient is ready or safe for a lower level of care.
- safe_for_lloc = false means the chart supports continued inpatient care.
- expected_los_recommendation is the expected additional inpatient length of stay in days: 0, 1, 2, or 3.
- lower_level_of_care_barriers should list concrete barriers from the chart.
- evidence_forms should list the form names or note types that support the answer.
- rationale should be concise and evidence-grounded.

Do not invent facts not present in the chart packet.
Do not use payer-specific criteria.
Do not claim this is a real patient record.

Chart packet:
{{CHART_PACKET}}