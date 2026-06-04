# ClinAuthBench Scaling Plan

ClinAuthBench v1 is a synthetic inpatient health authorization benchmark. V1 focuses on adult inpatient psychiatric authorization over dense 72-hour chart packets.

The public v1 release is intentionally evaluation-scale: 180 cases with structured gold labels, evidence anchors, documentation-challenge metadata, `do_not_claim` constraints, and MDP trajectory metadata. It is not intended for training a model from scratch.

## V1 Design Goals

V1 was designed to evaluate whether language models can:

- synthesize evidence across dense multi-form chart packets
- distinguish current risk from historical risk
- identify contradictions across forms
- handle missing, malformed, refused, or stale structured evidence
- reason about lower-level-of-care readiness without overclaiming
- generate authorization-style summaries without fabricating evidence

The core benchmark pattern is broader than the v1 clinical scope: evidence-grounded authorization reasoning over dense documentation.

## V1 Composition

- Total cases: 180.
- Continued-stay cases: 108.
- Safe/LLOC-ready cases: 72.
- Contradiction/conflicting-documentation cases: 22.
- Rule-based MDP cases: 120.
- Probabilistic MDP cases: 60.
- Probabilistic MDP trace range: cases 121-180.

Diagnosis family counts:

- Schizoaffective disorder with command hallucinations: 26.
- Major depressive disorder with psychotic features: 26.
- Borderline personality disorder self-harm crisis: 26.
- Substance-induced mood or psychotic symptoms: 26.
- Bipolar mixed episode: 26.
- Dual diagnosis / OUD-related inpatient psychiatric cases: 25.
- Trauma/anxiety with suicidality: 25.

Documentation-challenge categories:

- `current_vs_historical_risk`
- `contradiction`
- `lower_level_of_care_barrier_reasoning`
- `missing_invalid_or_stale_evidence`

## MDP Use

ClinAuthBench uses Markov Decision Process-style synthetic state transitions to generate coherent 72-hour chart packets. It is not reinforcement learning in v1: no policy is trained, no reward function is optimized, and no agent learns from feedback.

The v1 generation path is:

```text
CaseSpec
-> hidden synthetic patient state
-> MDP-style transition sequence
-> rendered chart forms
-> metadata.gold labels
-> deterministic quality checks
-> deterministic content/gold consistency checks
```

Cases 1-120 use `rule_based_v1`. Cases 121-180 use `probabilistic_v1`, which records transition options, probabilities, compatible outcomes, RNG roll, selected outcome, and resulting synthetic state in `metadata.mdp_trajectory`.

## V1 QA Summary

- Deterministic QA passed: PASS=180, REVIEW=0, FAIL=0.
- Claude PHI audit sampled 30 full-content cases: PASS=30, FAIL=0.

The public repository reports QA summaries but does not include local audit output JSON files.

## V1 Limitations

- V1 contains 180 cases and is evaluation-scale, not training-from-scratch scale.
- V1 focuses on adult inpatient psychiatric authorization only.
- V1 does not include pediatric, geriatric, ED-only, PHP/IOP, outpatient, or residential-only cases.
- Synthetic notes are cleaner and more uniform than real EHR documentation.
- Each case uses a fixed 72-hour documentation window.
- Authorization criteria are generalized and are not tied to any payer, provider, facility, EHR, contract, or proprietary utilization-management policy.
- MDP trajectories are synthetic generation metadata, not observed clinical transitions.

## V1.5 Scaling Direction

V1.5 should preserve the frozen v1 release while adding a harder companion set. Good targets:

- sparse chart packets with fewer forms
- noisier copied-forward documentation
- longer windows beyond 72 hours
- partial packets with missing form types
- more varied score handling
- more contradiction subtypes
- cases where the final answer is intentionally less obvious

The v1.5 goal should be benchmark difficulty calibration, not just a larger case count.

## V2 Scaling Direction

V2 can extend the benchmark pattern beyond the v1 scope:

- other inpatient authorization domains
- ED-to-inpatient authorization packets
- PHP/IOP step-down reasoning
- pediatric or geriatric variants as separately scoped releases
- multi-window staged review tasks
- trajectory-aware model evaluation

V2 may also explore learning from trajectory structure and Bayesian optimization for generator calibration, benchmark composition, and systematic scaling.

## Research Directions

Useful public evaluation tasks include:

- disposition classification
- expected LOS recommendation
- source-grounded evidence citation
- contradiction detection
- unsupported-claim detection using `do_not_claim`
- lower-level-of-care barrier extraction
- current-vs-historical risk reconciliation
- trajectory-aware evaluation using `metadata.mdp_trajectory`

The strongest public contribution is not raw size. It is the benchmark structure: dense synthetic packets, explicit negative constraints, evidence anchors, documentation challenges, and reproducible trajectory metadata.
