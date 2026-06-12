# Model Routing Notes

These notes document model access and routing behavior observed during ClinAuthBench v1 smoke-test setup.

The 12-case smoke set is a mechanical compatibility check. It is used to verify that a model route can retrieve outputs under the shared prompt/schema contract before running the full 180-case benchmark. Router and endpoint failures are not ClinAuthBench performance scores.

## Shared Retrieval Contract

All retrieval scripts use the same model-facing inputs:

- task instructions
- output JSON schema
- `case["content"]`

Retrieval scripts do not send `metadata.gold`, `evidence_anchors`, `do_not_claim`, `documentation_challenge`, or `documentation_challenge_tags`.

## Successful Smoke Routes

The following routes completed the 12-case smoke retrieval and can be scored locally:

| Source | Model ID | Notes |
| --- | --- | --- |
| Hugging Face router | `openai/gpt-oss-120b` | Completed retrieval through `/v1/chat/completions`. |
| Hugging Face router | `meta-llama/Meta-Llama-3-8B-Instruct` | Completed retrieval through `/v1/chat/completions`; output-contract failures are captured during scoring. |
| OpenAI Responses API | `gpt-5.2` | Completed retrieval through the local evaluation harness using the OpenAI API. |

## Unsupported Or Excluded Routes

The following model routes were attempted or considered but excluded from scored comparison because they could not be retrieved under the same endpoint contract in the tested environment.

| Model family | Attempted model ID | Observed issue | Interpretation |
| --- | --- | --- | --- |
| Google Gemma | `google/gemma-4-12B-it` | Hugging Face router reported the model is not a chat model for `/v1/chat/completions`. | Endpoint compatibility failure, not a benchmark score. |
| Google Gemma | `google/gemma-4-12B-it-assistant` | Hugging Face router reported the model is not a chat model for `/v1/chat/completions`. | Endpoint compatibility failure, not a benchmark score. |
| NVIDIA Nemotron | `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | Hugging Face router reported the model is not supported by any enabled provider. | Provider availability failure for the tested account/router, not a benchmark score. |
| Mistral Mixtral | `mistralai/Mixtral-8x7B-Instruct-v0.1` | Hugging Face router reported the model is not a chat model for `/v1/chat/completions`. | Endpoint compatibility failure, not a benchmark score. |

These failures should be reported as retrieval-route exclusions. A future version can add any of these model families if a compatible hosted chat-completions route or local inference path is available under the same prompt and output schema.

## Reporting Recommendation

For the technical report, keep the primary baseline table limited to models that completed retrieval. Report route failures separately in a short methods note:

> Additional open-weight model routes were attempted through the Hugging Face chat-completions router. Gemma, NVIDIA Nemotron, and Mixtral routes could not be executed under the shared endpoint contract in the tested environment and were excluded from scored comparison. These are provider/router compatibility failures, not model performance results.
