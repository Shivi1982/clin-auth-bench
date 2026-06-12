import re
from collections import Counter, defaultdict


def normalize_los_days(value):
    """
    Converts values like:
    - 0
    - "0"
    - "0 days"
    - "0_days"
    - "3 days"
    - "4 days"
    into integer 0,1,2,3,4.
    """
    if value is None:
        return None

    if isinstance(value, int):
        return value if value in {0, 1, 2, 3, 4} else None

    text = str(value).strip().lower()
    match = re.search(r"\b([0-4])\b", text)
    if match:
        return int(match.group(1))

    return None


def normalize_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False

    return None


def get_gold_safe_for_lloc(record):
    return normalize_bool(record["metadata"]["gold"]["safe_for_lloc"])


def get_gold_los_days(record):
    return normalize_los_days(record["metadata"]["gold"]["expected_los_recommendation"])


def get_pred_safe_for_lloc(prediction):
    return normalize_bool(prediction.get("safe_for_lloc"))


def get_pred_los_days(prediction):
    return normalize_los_days(prediction.get("expected_los_recommendation"))


def accuracy(items):
    if not items:
        return None
    return sum(items) / len(items)


def evaluate_records(records, predictions):
    """
    records: list of dataset rows
    predictions: list of model predictions in same order
    """

    if len(records) != len(predictions):
        raise ValueError(
            f"Length mismatch: records={len(records)}, predictions={len(predictions)}"
        )

    safe_correct = []
    los_correct = []
    usable = 0

    confusion = Counter()

    for record, pred in zip(records, predictions):
        gold_safe = get_gold_safe_for_lloc(record)
        pred_safe = get_pred_safe_for_lloc(pred)

        gold_los = get_gold_los_days(record)
        pred_los = get_pred_los_days(pred)

        if pred_safe is not None and pred_los is not None:
            usable += 1

        if pred_safe is not None:
            safe_correct.append(gold_safe == pred_safe)
            confusion[(gold_safe, pred_safe)] += 1

        if pred_los is not None:
            los_correct.append(gold_los == pred_los)

    return {
        "n_cases": len(records),
        "n_predictions": len(predictions),
        "usable_predictions": usable,
        "safe_for_lloc_accuracy": accuracy(safe_correct),
        "expected_los_exact_match": accuracy(los_correct),
        "safe_for_lloc_confusion": {
            "gold_false_pred_false": confusion[(False, False)],
            "gold_false_pred_true": confusion[(False, True)],
            "gold_true_pred_false": confusion[(True, False)],
            "gold_true_pred_true": confusion[(True, True)],
        },
    }


def evaluate_by_challenge(records, predictions):
    groups = defaultdict(lambda: {"records": [], "predictions": []})

    for record, pred in zip(records, predictions):
        challenge = record["metadata"].get("documentation_challenge", "unknown")
        groups[challenge]["records"].append(record)
        groups[challenge]["predictions"].append(pred)

    results = {}

    for challenge, bundle in groups.items():
        results[challenge] = evaluate_records(
            bundle["records"],
            bundle["predictions"],
        )

    return results
