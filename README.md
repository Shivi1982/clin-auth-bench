# ClinAuthBench

ClinAuthBench is a synthetic inpatient health authorization benchmark. V1 focuses on adult inpatient psychiatric authorization over dense 72-hour synthetic case documentation.

The benchmark is designed to test whether language models can reason over multi-form inpatient documentation without inventing unsupported authorization claims.

Many clinical NLP datasets evaluate extraction, single-note summarization, or classification. ClinAuthBench targets a different failure mode: evidence discipline across dense multi-form case documentation where useful evidence is distributed across nursing notes, psychiatric progress notes, rating scales, medication-response notes, group notes, treatment-plan reviews, and discharge-planning documentation.

Part of ongoing research on clinical LLM evaluation — [shivi1982.github.io](https://shivi1982.github.io)

## Public Dataset

- Hugging Face: https://huggingface.co/datasets/Shivi1982/clin-auth-bench

## What Makes It Different

- **Explicit negative constraints:** every case includes `metadata.gold.do_not_claim`, a set of tempting but unsupported conclusions that models should avoid.
- **Evidence anchors:** gold labels include supporting form hints, so evaluation can check whether claims are grounded in the case documentation.
- **Documentation-challenge taxonomy:** v1 labels current-vs-historical risk, contradiction, lower-level-of-care barrier reasoning, and missing/invalid/stale structured evidence.
- **MDP-style generation:** cases are generated using Markov Decision Process-style synthetic state transitions. Cases 121-180 include probabilistic MDP trajectory metadata for trajectory-aware evaluation.
- **Dense case-documentation structure:** each record contains a 72-hour, multi-form synthetic case documentation rather than a single note.

V1 does not train a policy. The MDP trajectory is synthetic generation metadata. Future versions may explore learning from trajectory structure and Bayesian optimization for generator calibration, benchmark composition, and scaling.

## Dataset Summary

- 180 synthetic cases.
- 108 continued-stay cases.
- 72 safe or lower-level-of-care-ready cases.
- 22 contradiction/conflicting-documentation cases.
- 120 `rule_based_v1` cases.
- 60 `probabilistic_v1` cases.
- Cases 121-180 include probabilistic transition traces.

The active release file is:

```text
data/release/synthetic_bh_cases_v1_mdp_180.json
```

## Repository Layout

```text
.
+-- data/
|   +-- release/
|       +-- synthetic_bh_cases_v1_mdp_180.json
+-- docs/
|   +-- schema.md
|   +-- release_notes.md
|   +-- model_routing_notes.md
|   +-- scaling_plan.md
+-- generators/
|   +-- mdp_case_builder.py
|   +-- form_templates.py
|   +-- generate_v1_30.py
|   +-- generate_v1_70.py
|   +-- generate_v1_120.py
|   +-- generate_v1_170.py
|   +-- generate_v1_180.py
+-- qa/
|   +-- qa_gpt_oss.py
|   +-- Dataset_QA.py
+-- evals/
|   +-- retrieve_hf_outputs.py
|   +-- retrieve_openai_outputs.py
|   +-- retrieve_claude_outputs.py
|   +-- score_model_outputs.py
|   +-- prompts/
|   +-- schema/
|   +-- config/
|   +-- results/
|   +-- scripts/
|   +-- parsing.py
|   +-- metrics.py
+-- notebooks/
|   +-- baseline_clinauthbench_v1.ipynb
+-- README.md
+-- requirements.txt
+-- LICENSE
+-- DATA_LICENSE.md
+-- CITATION.cff
```

Local review batches, archive files, pilot files, audit result JSONs, generated QA outputs, raw/scored model-output JSONL files under `evals/model_outputs/`, caches, and `.DS_Store` files are intentionally excluded from the public repository. Consolidated evaluation summaries under `evals/results/` are included.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For live model-output retrieval, copy the example environment file and add the API keys you intend to use:

```bash
cp .env.example .env
```

The scoring and summarization scripts do not require API keys. API keys are only needed for live model-output retrieval through Hugging Face, OpenAI, or Anthropic.


## Quick Start

Load the JSON directly:

```python
import json

with open("data/release/synthetic_bh_cases_v1_mdp_180.json", encoding="utf-8") as f:
    cases = json.load(f)

print(len(cases))
print(cases[0]["id"])
print(cases[0]["metadata"]["gold"]["do_not_claim"])
```

With Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={"test": "data/release/synthetic_bh_cases_v1_mdp_180.json"},
    split="test",
)
print(dataset[0]["id"])
```

## Baseline Notebook

The first baseline notebook demonstrates how to load the dataset from Hugging Face, inspect one case, build a zero-shot prompt, run a few sample cases manually or through an API placeholder, and compute first-pass metrics:

```text
notebooks/baseline_clinauthbench_v1.ipynb
```

The notebook is intentionally simple. It is meant to prove usability and establish a starting evaluation protocol, not to claim benchmark-leading performance.

## Baseline Evaluation Harness

The `evals/` directory contains the v1 baseline evaluation harness. The goal is to run multiple model systems under the same retrieval contract, save raw outputs, parse and validate those outputs locally, and score them against gold labels without exposing hidden metadata to the model.

The 12-case smoke set is a mechanical compatibility check, not the publishable benchmark result. It is used to catch issues such as wrong model IDs, provider/router failures, malformed JSON, truncation, schema mismatches, or output-saving failures before running all 180 cases.

The Hugging Face retrieval script sends only:

- task instructions
- output JSON schema
- `case["content"]`

It does not send `metadata.gold`, `evidence_anchors`, `do_not_claim`, `documentation_challenge`, or `documentation_challenge_tags`.

Set local tokens in a `.env` file:

```bash
cp .env.example .env
# edit .env and set whichever tokens you need:
# HF_TOKEN=hf_...
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...  # optional
```

Run the 12-case smoke retrieval:

```bash
python evals/retrieve_hf_outputs.py --model-id openai/gpt-oss-120b
```

The current Hugging Face smoke retrieval script also supports:

```bash
python evals/retrieve_hf_outputs.py --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

Raw model outputs are written to:

```text
evals/model_outputs/raw/hf/<model_slug>/v1_smoke_outputs.jsonl
```

The retrieval script intentionally does not parse, validate, or score outputs. Parsing/schema validation and scoring are separate local steps so retrieval failures, parse failures, schema failures, and benchmark accuracy can be reported separately.

OpenAI-hosted retrieval is available through the local evaluation harness:

```bash
python evals/retrieve_openai_outputs.py --model-id gpt-5.2
```

### Supported Retrieval Routes

Three retrieval scripts share the same blind retrieval contract (task instructions, output schema, and `case["content"]` only). They differ only in the model provider and the API key they read from `.env`:

| Script | Provider | API key | Example model id |
| --- | --- | --- | --- |
| `evals/retrieve_hf_outputs.py` | Hugging Face router | `HF_TOKEN` | `openai/gpt-oss-120b`, `meta-llama/Meta-Llama-3-8B-Instruct` |
| `evals/retrieve_openai_outputs.py` | OpenAI | `OPENAI_API_KEY` | `gpt-5.2` |
| `evals/retrieve_claude_outputs.py` | Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |

All three write raw outputs to `evals/model_outputs/raw/<source>/<model_slug>/` and accept the same `--case-config`, `--case-set-name`, `--expected-case-count`, and `--output-path` arguments shown in the full-180 example below. New users should start with `retrieve_hf_outputs.py` on the 12-case smoke set before attempting the full-180 run.

Score any raw output JSONL locally:

```bash
python evals/score_model_outputs.py \
  --raw-output-path evals/model_outputs/raw/<source>/<model_slug>/v1_smoke_outputs.jsonl
```

For the full 180-case evaluation, use the full case config and run name:

```bash
python evals/retrieve_hf_outputs.py \
  --model-id openai/gpt-oss-120b \
  --case-config evals/config/v1_full_180_cases.json \
  --case-set-name v1_full_180_cases \
  --expected-case-count 180 \
  --output-path evals/model_outputs/raw/hf/openai__gpt-oss-120b/v1_full_180_outputs.jsonl

python evals/score_model_outputs.py \
  --raw-output-path evals/model_outputs/raw/hf/openai__gpt-oss-120b/v1_full_180_outputs.jsonl \
  --case-config evals/config/v1_full_180_cases.json \
  --case-set-name v1_full_180_cases \
  --run-name v1_full_180 \
  --expected-case-count 180
```

Use the same full-180 arguments with `retrieve_openai_outputs.py` or another supported retrieval script for additional model routes.

### Full-180 evaluation results

The full-180 baseline evaluation for completed model routes was run on all 180 synthetic cases using the same blind retrieval contract and local scoring pipeline.

Primary metrics use all expected cases as the denominator. Retrieval failures, JSON parsing failures, and schema-validation failures count as incorrect. Valid-only metrics are reported only as diagnostics.

| Model               |       Source | Cases | Valid JSON | Schema valid | Safe-for-LLOC accuracy | LOS exact match | Safe valid-only | LOS valid-only | Notes                                                                             |
| ------------------- | -----------: | ----: | ---------: | -----------: | ---------------------: | --------------: | --------------: | -------------: | --------------------------------------------------------------------------------- |
| GPT-OSS 120B        | Hugging Face |   180 |    180/180 |      180/180 |                  93.3% |           51.7% |           93.3% |          51.7% | Full schema-valid run                                                             |
| GPT-5.2             |       OpenAI |   180 |    180/180 |      180/180 |                  65.0% |           33.9% |           65.0% |          33.9% | Full schema-valid run                                                             |
| Llama 3 8B Instruct | Hugging Face |   180 |     78/180 |       40/180 |                  17.2% |            2.2% |           77.5% |          10.0% | Diagnostic smaller OSS model; many outputs failed the strict JSON/schema contract |

See the consolidated result artifact:

```text
evals/results/v1_full_180_summary.md
```

These results should not be interpreted as clinical performance, payer-policy performance, or real-world authorization safety.


### Hugging Face Model Routing Notes

Hugging Face router compatibility is model- and endpoint-dependent. During smoke testing, `openai/gpt-oss-120b` and `meta-llama/Meta-Llama-3-8B-Instruct` worked with the shared `/v1/chat/completions` retrieval path.

The Gemma, NVIDIA Nemotron, and Mixtral routes tested during setup could not be executed under the same Hugging Face chat-completions endpoint in the tested environment. These are provider/router compatibility exclusions, not ClinAuthBench performance results.

See `docs/model_routing_notes.md` for the standalone route log and recommended reporting language.

## Reproduce The V1 Release File

From the repository root:

```bash
python generators/generate_v1_180.py
```

This regenerates the 180-case release file. It may also create local review-batch output for QA; review-batch directories are ignored by git.

Do not regenerate the release file unless you intend to change the dataset version or verify reproducibility.

## Run Deterministic QA

```bash
python qa/qa_gpt_oss.py \
  --input data/release/synthetic_bh_cases_v1_mdp_180.json \
  --guardrail-profile none \
  --deterministic-only
```

Hosted GPT-OSS review requires `HF_TOKEN` and `huggingface_hub`:

```bash
export HF_TOKEN="hf_..."
python qa/qa_gpt_oss.py \
  --input data/release/synthetic_bh_cases_v1_mdp_180.json \
  --guardrail-profile none
```

Full narrative QA with Anthropic is available through `qa/Dataset_QA.py` and requires `ANTHROPIC_API_KEY`.

## Privacy And Affiliation Notice

ClinAuthBench is fully synthetic. It is generated from structured synthetic case specifications, Markov Decision Process-style synthetic trajectories, synthetic timelines, and controlled documentation templates.

It is not derived from, affiliated with, endorsed by, or representative of any healthcare provider, payer, employer, customer, EHR vendor, or real patient population.

The dataset does not contain real patient records, copied chart text, PHI, real facility names, real staff names, real patient names, MRNs, phone numbers, addresses, or source-system identifiers.

All chart dates are synthetic timeline artifacts and do not correspond to real patient encounters or real facility operations.

## Limitations

- V1 contains 180 cases and is evaluation-scale, not training-from-scratch scale.
- V1 focuses on adult inpatient psychiatric authorization only.
- V1 does not include pediatric, geriatric, ED-only, PHP/IOP, outpatient, or residential-only cases.
- Synthetic notes are cleaner and more uniform than real EHR documentation.
- Each case uses a fixed 72-hour documentation window.
- Authorization criteria are generalized and are not tied to any payer, provider, facility, EHR, contract, or proprietary utilization-management policy.
- MDP trajectories are synthetic generation metadata, not observed clinical transitions.

## License

- Code in this repository is released under the Apache License 2.0. See `LICENSE`.
- The dataset file under `data/release/` is released under Creative Commons Attribution 4.0 International (`CC BY 4.0`). See `DATA_LICENSE.md`.

## Citation

If you use ClinAuthBench, please cite the dataset version and repository. See `CITATION.cff`.
