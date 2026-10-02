import argparse
import json
import math
from pathlib import Path

from utils.jsonl import read_jsonl
from utils.metrics import evaluate_caption_records
from .schema import NUMERIC_FIELDS, REQUIRED_FIELDS, items_text


def _item_names(label):
    if not isinstance(label, dict):
        return set()
    return {str(item.get("name", "")).strip().lower() for item in label.get("items", []) if isinstance(item, dict) and item.get("name")}


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def item_scores(gold_labels, predicted_labels):
    true_positive = false_positive = false_negative = 0
    classes = set()
    for gold, predicted in zip(gold_labels, predicted_labels):
        gold_names, predicted_names = _item_names(gold), _item_names(predicted)
        classes.update(gold_names | predicted_names)
        overlap = gold_names & predicted_names
        true_positive += len(overlap)
        false_positive += len(predicted_names - gold_names)
        false_negative += len(gold_names - predicted_names)
    micro = _f1(true_positive / max(1, true_positive + false_positive), true_positive / max(1, true_positive + false_negative))
    class_f1 = []
    for name in sorted(classes):
        tp = fp = fn = 0
        for gold, predicted in zip(gold_labels, predicted_labels):
            gold_has, predicted_has = name in _item_names(gold), name in _item_names(predicted)
            if gold_has and predicted_has:
                tp += 1
            elif predicted_has:
                fp += 1
            elif gold_has:
                fn += 1
        class_f1.append(_f1(tp / max(1, tp + fp), tp / max(1, tp + fn)))
    return micro, sum(class_f1) / len(class_f1) if class_f1 else 0.0


def structured_metrics(rows):
    gold, predicted, text_records = [], [], []
    parsed_count = valid_count = complete_count = 0
    consistency_total = consistency_valid = 0
    sums = {field: 0.0 for field in NUMERIC_FIELDS}
    counts = {field: 0 for field in NUMERIC_FIELDS}

    for row in rows:
        gold_label = row.get("label", {})
        result = row.get("parsed") or {}
        parsed_label = result.get("parsed") if isinstance(result, dict) else None
        if isinstance(parsed_label, dict):
            parsed_count += 1
            if all(field in parsed_label for field in REQUIRED_FIELDS):
                complete_count += 1
            if result.get("valid"):
                valid_count += 1
            if isinstance(parsed_label.get("items"), list) and isinstance(parsed_label.get("total_weight_g"), (int, float)) and not isinstance(parsed_label.get("total_weight_g"), bool) and math.isfinite(float(parsed_label["total_weight_g"])):
                weights = [item.get("weight_g") for item in parsed_label["items"] if isinstance(item, dict)]
                if len(weights) == len(parsed_label["items"]) and all(isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(float(weight)) for weight in weights):
                    consistency_total += 1
                    if "weight_total_inconsistent" not in result.get("errors", []):
                        consistency_valid += 1
        else:
            parsed_label = {"items": []}
        gold.append(gold_label)
        predicted.append(parsed_label)
        text_records.append({"references": [items_text(gold_label)], "prediction": items_text(parsed_label)})
        for field in NUMERIC_FIELDS:
            value = parsed_label.get(field) if isinstance(parsed_label, dict) else None
            target = gold_label.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and isinstance(target, (int, float)) and not isinstance(target, bool) and math.isfinite(float(target)):
                sums[field] += (float(value) - float(target)) ** 2
                counts[field] += 1

    micro, macro = item_scores(gold, predicted)
    result = evaluate_caption_records(text_records)
    result.update({
        "json_parse_rate": parsed_count / max(1, len(rows)),
        "schema_valid_rate": valid_count / max(1, len(rows)),
        "field_complete_rate": complete_count / max(1, len(rows)),
        "weight_consistency_rate": consistency_valid / consistency_total if consistency_total else None,
        "item_micro_f1": micro,
        "item_macro_f1": macro,
        "mse": {field: sums[field] / counts[field] if counts[field] else None for field in NUMERIC_FIELDS},
        "numeric_coverage": {field: counts[field] / max(1, len(rows)) for field in NUMERIC_FIELDS},
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="results/meal_json/evaluation.json")
    args = parser.parse_args()
    rows = list(read_jsonl(args.predictions))
    result = structured_metrics(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
