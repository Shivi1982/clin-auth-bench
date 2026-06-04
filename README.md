# ClinAuthBench

ClinAuthBench is a synthetic inpatient health authorization benchmark. V1 focuses on adult inpatient psychiatric authorization over dense 72-hour chart packets.

The benchmark is designed to test whether language models can reason over multi-form inpatient documentation without inventing unsupported authorization claims.

Many clinical NLP datasets evaluate extraction, single-note summarization, or classification. ClinAuthBench targets a different failure mode: evidence discipline across a dense chart packet where useful evidence is distributed across nursing notes, psychiatric progress notes, rating scales, medication-response notes, group notes, treatment-plan reviews, and discharge-planning documentation.

## Public Dataset

- Hugging Face: https://huggingface.co/datasets/Shivi1982/clin-auth-bench

## What Makes It Different

- **Explicit negative constraints:** every case includes `metadata.gold.do_not_claim`, a set of tempting but unsupported conclusions that models should avoid.
- **Evidence anchors:** gold labels include supporting form hints, so evaluation can check whether claims are grounded in the packet.
- **Documentation-challenge taxonomy:** v1 labels current-vs-historical risk, contradiction, lower-level-of-care barrier reasoning, and missing/invalid/stale structured evidence.
- **MDP-style generation:** cases are generated using Markov Decision Process-style synthetic state transitions. Cases 121-180 include probabilistic MDP trajectory metadata for trajectory-aware evaluation.
- **Dense packet structure:** each record contains a 72-hour, multi-form chart packet rather than a single note.

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
+-- notebooks/
|   +-- baseline_clinauthbench_v1.ipynb
+-- README.md
+-- requirements.txt
+-- LICENSE
+-- DATA_LICENSE.md
+-- CITATION.cff
```

Local review batches, archive files, pilot files, audit result JSONs, generated QA outputs, caches, and `.DS_Store` files are intentionally excluded from the public repository.

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
