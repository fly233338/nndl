import argparse
import json
from collections import Counter
from pathlib import Path

from utils.jsonl import read_jsonl
from utils.metrics import evaluate_caption_records
from .schema import items_text


def _item_names(label):
    return {str(item.get("name", "")).strip().lower() for item in label.get("items", []) if item.get("name")}


def item_f1(gold_labels, predicted_labels):
    tp = fp = fn = 0
    per_sample = []
    for gold, pred in zip(gold_labels, predicted_labels):
        g, p = _item_names(gold), _item_names(pred)
        overlap = len(g & p)
        tp += overlap; fp += len(p - g); fn += len(g - p)
        precision = overlap / len(p) if p else 0.0
        recall = overlap / len(g) if g else 0.0
        per_sample.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    micro = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return micro, sum(per_sample) / max(1, len(per_sample))


def structured_metrics(rows):
    gold, predicted, text_records, invalid = [], [], [], 0
    numeric_fields = ("total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g")
    squared = Counter()
    for row in rows:
        gold_label = row["label"]
        parsed = row.get("parsed")
        if not parsed or not parsed.get("valid"):
            invalid += 1
            parsed_label = {"items": []}
            parsed_label.update({field: 0.0 for field in numeric_fields})
        else:
            parsed_label = parsed["parsed"]
        gold.append(gold_label); predicted.append(parsed_label)
        text_records.append({"references": [items_text(gold_label)], "prediction": items_text(parsed_label)})
        for field in numeric_fields:
            squared[field] += (float(parsed_label.get(field, 0)) - float(gold_label.get(field, 0))) ** 2
    micro, macro = item_f1(gold, predicted)
    result = evaluate_caption_records(text_records)
    result.update({"item_micro_f1": micro, "item_macro_f1": macro, "json_valid_rate": 1 - invalid / max(1, len(rows)), "field_complete_rate": sum(bool(row.get("parsed", {}).get("valid")) for row in rows) / max(1, len(rows))})
    result["mse"] = {field: value / max(1, len(rows)) for field, value in squared.items()}
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
