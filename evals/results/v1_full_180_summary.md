# ClinAuthBench v1 full-180 evaluation summary

ClinAuthBench v1 is fully synthetic and is not real patient data.

Primary metrics use all expected cases as the denominator. Retrieval, JSON parsing, and schema-validation failures count as incorrect. Valid-only metrics are included only as diagnostics.

## Overall results

| Model | Source | Cases | Valid JSON | Schema valid | Safe-for-LLOC accuracy | LOS exact match | Safe valid-only | LOS valid-only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-OSS 120B | huggingface | 180 | 180/180 | 180/180 | 93.3% | 51.7% | 93.3% | 51.7% |
| GPT-5.2 | openai | 180 | 180/180 | 180/180 | 65.0% | 33.9% | 65.0% | 33.9% |
| Llama 3 8B Instruct | huggingface | 180 | 78/180 | 40/180 | 17.2% | 2.2% | 77.5% | 10.0% |

## Notes

- GPT-OSS 120B and GPT-5.2 produced schema-valid JSON for all 180 cases.
- Llama 3 8B Instruct is reported as a smaller open-weight diagnostic model because many outputs failed the strict JSON/schema contract.
- These results should not be interpreted as clinical performance, payer-policy performance, or real-world authorization safety.

## GPT-OSS 120B challenge breakdown

| Documentation challenge | Cases | Schema valid | Safe-for-LLOC accuracy | LOS exact match | Safe valid-only | LOS valid-only |
| --- | --- | --- | --- | --- | --- | --- |
| contradiction | 22 | 22/22 | 100.0% | 59.1% | 100.0% | 59.1% |
| current_vs_historical_risk | 45 | 45/45 | 77.8% | 46.7% | 77.8% | 46.7% |
| lower_level_of_care_barrier_reasoning | 81 | 81/81 | 97.5% | 44.4% | 97.5% | 44.4% |
| missing_invalid_or_stale_evidence | 32 | 32/32 | 100.0% | 71.9% | 100.0% | 71.9% |

## GPT-5.2 challenge breakdown

| Documentation challenge | Cases | Schema valid | Safe-for-LLOC accuracy | LOS exact match | Safe valid-only | LOS valid-only |
| --- | --- | --- | --- | --- | --- | --- |
| contradiction | 22 | 22/22 | 100.0% | 54.5% | 100.0% | 54.5% |
| current_vs_historical_risk | 45 | 45/45 | 13.3% | 13.3% | 13.3% | 13.3% |
| lower_level_of_care_barrier_reasoning | 81 | 81/81 | 74.1% | 40.7% | 74.1% | 40.7% |
| missing_invalid_or_stale_evidence | 32 | 32/32 | 90.6% | 31.2% | 90.6% | 31.2% |

## Llama 3 8B Instruct challenge breakdown

| Documentation challenge | Cases | Schema valid | Safe-for-LLOC accuracy | LOS exact match | Safe valid-only | LOS valid-only |
| --- | --- | --- | --- | --- | --- | --- |
| contradiction | 22 | 8/22 | 36.4% | 0.0% | 100.0% | 0.0% |
| current_vs_historical_risk | 45 | 6/45 | 2.2% | 0.0% | 16.7% | 0.0% |
| lower_level_of_care_barrier_reasoning | 81 | 13/81 | 12.3% | 4.9% | 76.9% | 30.8% |
| missing_invalid_or_stale_evidence | 32 | 13/32 | 37.5% | 0.0% | 92.3% | 0.0% |
