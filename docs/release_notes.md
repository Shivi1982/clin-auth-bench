# ClinAuthBench V1 Release Notes

ClinAuthBench v1 is a synthetic inpatient health authorization benchmark. V1 focuses on adult inpatient psychiatric authorization over dense 72-hour chart packets.

## Included

- `README.md`: repository overview.
- `data/release/synthetic_bh_cases_v1_mdp_180.json`: 180-case public release file.
- `docs/schema.md`: public schema summary.
- `docs/release_notes.md`: this release note.
- `docs/scaling_plan.md`: scaling and future-version planning notes.
- `generators/`: reproducible v1 generation code.
- `qa/`: deterministic and optional hosted QA scripts.

## Not Included

The public GitHub repository intentionally excludes:

- review batch files
- archived pilot files
- earlier pilot files
- local audit result JSON files
- `__pycache__`
- `.DS_Store`
- generated QA output directories

## Dataset Composition

- Total cases: 180.
- Continued-stay cases: 108.
- Safe/LLOC-ready cases: 72.
- Contradiction/conflicting-documentation cases: 22.
- Rule-based MDP cases: 120.
- Probabilistic MDP cases: 60.
- Probabilistic MDP trace range: cases 121-180.

## QA Summary

- Deterministic QA passed: PASS=180, REVIEW=0, FAIL=0.
- Claude PHI audit sampled 30 full-content cases: PASS=30, FAIL=0.

## Scope

V1 is evaluation-scale and should be used for benchmark evaluation, error analysis, prompt testing, and small calibration studies. It is not training-from-scratch scale.

The dataset uses generalized synthetic authorization criteria and does not represent any payer, provider, facility, EHR, contract, or proprietary utilization-management rule set.

All chart dates are synthetic timeline artifacts and do not correspond to real patient encounters or real facility operations.

## License

ClinAuthBench v1 is released under the Creative Commons Attribution 4.0 International license (`CC BY 4.0`).
