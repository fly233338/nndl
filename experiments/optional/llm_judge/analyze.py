"""汇总 Judge 评分，并与人工评分或逐图指标分析相关性。"""

import argparse
import json
import math
from pathlib import Path

from utils.jsonl import read_jsonl

from .client import SCORE_FIELDS, parse_scores


def summary(records):
    valid = [r for r in records if r["status"] == "success"]
    return {
        "samples": len(records),
        "successes": len(valid),
        "failures": len(records) - len(valid),
        "coverage": len(valid) / len(records) if records else 0,
        "mean_scores": {
            key: sum(r["scores"][key] for r in valid) / len(valid) if valid else None
            for key in SCORE_FIELDS
        },
        "mean_overall": sum(r["overall"] for r in valid) / len(valid) if valid else None,
    }


def indexed(rows):
    result = {}
    for row in rows:
        if row["image_id"] in result:
            raise ValueError(f"Duplicate image ID: {row['image_id']}")
        result[row["image_id"]] = row
    return result


def correlation(pairs):
    """仅用有限数值配对计算相关性，样本不足或常量序列返回空指标。"""
    from scipy.stats import pearsonr, spearmanr

    pairs = [
        (a, b)
        for a, b in pairs
        if all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (a, b)
        )
    ]
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return {
            "n": len(pairs),
            "pearson": None,
            "spearman": None,
            "note": "Need at least 3 matched, nonconstant pairs",
        }
    left, right = zip(*pairs)
    return {
        "n": len(pairs),
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
    }


def analyze(records_path, human_ratings=None, automatic_scores=None):
    """按图片编号对齐有效评分，统计总体表现和可选的一致性指标。"""
    records = list(read_jsonl(records_path))
    indexed(records)
    valid = {}
    for record in records:
        if record["status"] == "success":
            scores = parse_scores(json.dumps({**record["scores"], "reason": record["reason"]}))
            record["overall"] = sum(scores[k] for k in SCORE_FIELDS) / len(SCORE_FIELDS)
            valid[record["image_id"]] = record
    result = summary(records)
    if human_ratings:
        human = indexed(read_jsonl(human_ratings))
        human_agreement = {}
        for field in SCORE_FIELDS:
            pairs = [
                (record["scores"][field], human[image_id].get(field))
                for image_id, record in valid.items()
                if image_id in human
            ]
            human_agreement[field] = correlation(pairs)
        result["human_agreement"] = human_agreement
    if automatic_scores:
        automatic = indexed(read_jsonl(automatic_scores))
        metrics = sorted({key for row in automatic.values() for key in row.get("metrics", {})})
        automatic_agreement = {}
        for metric in metrics:
            pairs = [
                (record["overall"], automatic[image_id].get("metrics", {}).get(metric))
                for image_id, record in valid.items()
                if image_id in automatic
            ]
            automatic_agreement[metric] = correlation(pairs)
        result["automatic_agreement"] = automatic_agreement
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--human-ratings")
    parser.add_argument("--automatic-scores")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.records, args.human_ratings, args.automatic_scores)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
