# ClinAuthBench Schema

This document summarizes the public v1 schema for the Hugging Face release package.

## Top-Level Record

```text
{
  "id": string,
  "title": string,
  "metadata": object,
  "content": string
}
```

## Top-Level Fields

- `id`: stable synthetic case identifier, such as `clin_auth_bench_v1_0001`.
- `title`: short case title.
- `content`: model-facing synthetic chart packet containing repeated 72-hour inpatient documentation forms.
- `metadata`: structured benchmark metadata, generation metadata, quality checks, and gold labels.

## Metadata Fields

- `benchmark`: benchmark name.
- `version`: dataset version.
- `synthetic`: boolean synthetic-data marker.
- `privacy_design`: short privacy-design description.
- `level_of_care`: v1 level-of-care scope.
- `documentation_window_hours`: documentation window size.
- `diagnosis_category`: synthetic diagnosis family.
- `trajectory`: synthetic trajectory descriptor.
- `documentation_challenge`: primary documentation challenge.
- `documentation_challenge_tags`: more specific challenge tags.
- `mdp_model`: `rule_based_v1` or `probabilistic_v1`.
- `mdp_trajectory`: synthetic transition trajectory metadata.
- `gold`: structured labels for evaluation.
- `quality_checks`: deterministic generation quality checks.
- `content_gold_checks`: deterministic content/gold consistency checks.

## Gold Fields

- `current_suicide_risk`: text label for current risk reasoning.
- `historical_suicide_risk`: text label for historical risk reasoning.
- `primary_diagnosis`: synthetic primary diagnosis.
- `psychosis_present`: boolean.
- `substance_use`: list of synthetic substance-use categories.
- `safe_for_lloc`: boolean lower-level-of-care readiness label.
- `expected_los_recommendation`: expected length-of-stay recommendation label.
- `lower_level_of_care_barriers`: list of unresolved barriers.
- `documentation_challenge`: primary challenge category.
- `documentation_challenge_tags`: challenge tags.
- `key_evidence_forms`: forms expected to contain important evidence.
- `evidence_anchors`: structured hints linking gold fields to supporting forms.
- `do_not_claim`: explicit unsupported claims the model should avoid.
- `form_count`: number of rendered forms in the chart packet.

## Documentation Challenge Values

- `current_vs_historical_risk`
- `contradiction`
- `lower_level_of_care_barrier_reasoning`
- `missing_invalid_or_stale_evidence`

## MDP Trajectory

Cases 1-120 use `rule_based_v1`. Cases 121-180 use `probabilistic_v1`.

Rule-based trajectory entries include the day, action, and resulting synthetic patient state.

Probabilistic trajectory entries include transition metadata such as transition options, conditioned probability mass, sample roll, selected outcome, selected probability, and resulting synthetic state.

The MDP trajectory is synthetic generation metadata. It is not an observed clinical trajectory and should not be treated as real patient-course data.
