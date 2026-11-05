"""统计结构、数值误差和调用成本，无需文本指标依赖。"""

import argparse
import json
import math
from pathlib import Path

from experiments.task2_meal_json.evaluate import item_scores
from experiments.task2_meal_json.schema import NUMERIC_FIELDS, parse_and_validate
from utils.jsonl import read_jsonl


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    values = [value for value in values if _finite(value)]
    return sum(values) / len(values) if values else None


def evaluate_records(records, manifest=None):
    """按完整清单统计指标，缺失或失败预测计入覆盖率分母，不补零。"""
    rows = list(records)
    identifiers = [row["image_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate image IDs in predictions")

    if manifest is not None:
        gold_rows = list(read_jsonl(manifest))
        gold_ids = [row["image_id"] for row in gold_rows]
        if len(gold_ids) != len(set(gold_ids)):
            raise ValueError("Duplicate image IDs in evaluation manifest")

        if set(identifiers) - set(gold_ids):
            raise ValueError("Predictions contain images outside the evaluation manifest")
        modes = {row.get("mode") for row in rows}
        mode = next(iter(modes)) if len(modes) == 1 else None
        indexed = {row["image_id"]: row for row in rows}
        rows = [
            {
                **indexed.get(
                    row["image_id"],
                    {"image_id": row["image_id"], "mode": mode, "status": "missing_prediction"},
                ),
                "label": row["label"],
            }
            for row in gold_rows
        ]
    if not rows:
        raise ValueError("No predictions to evaluate")

    if any(not isinstance(row.get("label"), dict) for row in rows):
        raise ValueError("Evaluation needs labels in predictions or a --manifest with labels")
    sums = dict.fromkeys(NUMERIC_FIELDS, 0.0)
    absolutes = dict.fromkeys(NUMERIC_FIELDS, 0.0)
    counts = dict.fromkeys(NUMERIC_FIELDS, 0)
    valid = failures = missing = 0
    item_gold, item_predicted = [], []
    consistency_errors = []
    for row in rows:
        parsed = row.get("parsed") or parse_and_validate(row.get("raw_output", ""))
        prediction = parsed.get("parsed")
        is_valid = bool(parsed.get("valid")) and row.get("status") not in (
            "failed",
            "missing_prediction",
        )
        valid += is_valid
        failures += not is_valid
        missing += row.get("status") == "missing_prediction"
        item_gold.append(row["label"])
        item_predicted.append(prediction if is_valid else {"items": []})
        for field in NUMERIC_FIELDS:
            value = prediction.get(field) if isinstance(prediction, dict) else None
            target = row["label"].get(field)
            if is_valid and _finite(value) and _finite(target):
                error = value - target
                sums[field] += error**2
                absolutes[field] += abs(error)
                counts[field] += 1
        if is_valid and isinstance(prediction, dict):
            consistency_errors.append(
                abs(
                    sum(item["weight_g"] for item in prediction["items"])
                    - prediction["total_weight_g"],
                ),
            )
    micro_f1, macro_f1 = item_scores(item_gold, item_predicted)
    result = {
        "samples": len(rows),
        "item_micro_f1": micro_f1,
        "item_macro_f1": macro_f1,
        "schema_valid_rate": valid / len(rows),
        "failure_rate": failures / len(rows),
        "missing_prediction_count": missing,
        "mse": {
            field: sums[field] / counts[field] if counts[field] else None
            for field in NUMERIC_FIELDS
        },
        "mae": {
            field: absolutes[field] / counts[field] if counts[field] else None
            for field in NUMERIC_FIELDS
        },
        "numeric_coverage": {field: counts[field] / len(rows) for field in NUMERIC_FIELDS},
        "mean_item_weight_difference_g": _mean(consistency_errors),
        "mean_seconds": _mean([row.get("seconds") for row in rows]),
        "mean_model_requests": _mean([row.get("model_requests") for row in rows]),
        "mean_tool_calls": _mean([row.get("tool_call_count") for row in rows]),
    }
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(row.get("usage") or {}).get(field) for row in rows]
        result[f"mean_{field}"] = _mean(values)
        result[f"{field}_coverage"] = sum(_finite(value) for value in values) / len(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--manifest", help="Expected cohort; missing predictions count as failures")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = {path: evaluate_records(read_jsonl(path), args.manifest) for path in args.predictions}
    result = next(iter(runs.values())) if len(runs) == 1 else {"runs": runs}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
