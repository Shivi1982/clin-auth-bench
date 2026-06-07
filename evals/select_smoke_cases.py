import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DATA_PATH = Path("data/release/synthetic_bh_cases_v1_mdp_180.json")
DEFAULT_OUTPUT_PATH = Path("evals/config/v1_smoke_cases.json")

EXPECTED_CHALLENGES = [
    "current_vs_historical_risk",
    "contradiction",
    "lower_level_of_care_barrier_reasoning",
    "missing_invalid_or_stale_evidence",
]


def load_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_case_id(record: dict):
    case_id = record.get("id") or record.get("case_id")

    if not case_id:
        raise ValueError("Case record is missing both 'id' and 'case_id'.")

    return case_id


def get_documentation_challenge(record: dict):
    metadata = record.get("metadata", {})
    challenge = metadata.get("documentation_challenge") or record.get("documentation_challenge")

    if not challenge:
        raise ValueError(f"Case {get_case_id(record)} is missing documentation_challenge.")

    return challenge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--per-challenge", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_cases(data_path)

    challenge_to_cases = defaultdict(list)

    for index, record in enumerate(cases):
        challenge = get_documentation_challenge(record)
        challenge_to_cases[challenge].append(
            {
                "case_id": get_case_id(record),
                "case_index": index,
                "documentation_challenge": challenge,
            }
        )

    challenge_counts = {
        challenge: len(items)
        for challenge, items in sorted(challenge_to_cases.items())
    }

    print("Dataset challenge distribution:")
    print(json.dumps(challenge_counts, indent=2))

    missing = [
        challenge
        for challenge in EXPECTED_CHALLENGES
        if challenge not in challenge_to_cases
    ]

    if missing:
        raise ValueError(f"Missing expected challenge categories: {missing}")

    selected_cases = []
    rng = random.Random(args.seed)

    for challenge in EXPECTED_CHALLENGES:
        available = challenge_to_cases[challenge]

        if len(available) < args.per_challenge:
            raise ValueError(
                f"Challenge '{challenge}' has only {len(available)} cases, "
                f"but requested {args.per_challenge}."
            )

        selected = rng.sample(available, args.per_challenge)
        selected = sorted(selected, key=lambda row: row["case_index"])
        selected_cases.extend(selected)

    selected_cases = sorted(selected_cases, key=lambda row: row["case_index"])

    smoke_manifest = {
        "name": "ClinAuthBench v1 smoke case set",
        "purpose": "Fixed stratified smoke-test subset used before full 180-case model evaluation.",
        "data_path": str(data_path),
        "selection_rule": "Random stratified selection by metadata.documentation_challenge.",
        "important_note": (
            "documentation_challenge is used only to select a balanced smoke-test subset. "
            "It must not be shown to models during prediction."
        ),
        "seed": args.seed,
        "per_challenge": args.per_challenge,
        "n_cases": len(selected_cases),
        "selected_challenge_counts": dict(
            Counter(row["documentation_challenge"] for row in selected_cases)
        ),
        "cases": selected_cases,
    }

    output_path.write_text(
        json.dumps(smoke_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSelected {len(selected_cases)} smoke cases.")
    print(f"Saved smoke manifest to: {output_path}")


if __name__ == "__main__":
    main()