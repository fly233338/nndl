"""仅用符合条件的训练集单项餐，统计类别营养密度。"""

import argparse
import json
from pathlib import Path

from utils.config import load_yaml
from utils.files import sha256_file

from .prepare_cache import load_nutrition_cache, nutrition_signature
from .tools import NUTRIENT_FIELDS, NutritionDB, number


def calibrate(config):
    """仅统计名称对齐的训练单项餐，每组计一次，按总质量加权营养密度。"""
    rows = load_nutrition_cache(config)
    db = NutritionDB(config, calibrated=False)
    groups = {}
    eligible = 0
    seen_groups = set()
    for row in rows:
        if row["group"] in seen_groups:
            continue
        observed = row["observation"]["items"]
        gold_items = row["label"]["items"]
        # 混合餐的整餐标签无法唯一确定各食材的营养密度。
        if len(observed) != 1 or len(gold_items) != 1:
            continue
        item = observed[0]
        if db.normalize(gold_items[0]["name"]) not in {
            db.normalize(item["name"]),
            db.normalize(item["category"]),
        }:
            continue
        weight = number(row["label"]["total_weight_g"], "training weight", True)
        seen_groups.add(row["group"])
        key = "|".join(
            item[k].strip().lower() for k in ("category", "cooking_method", "food_state", "sauce")
        )
        stats = groups.setdefault(
            key,
            {"samples": 0, "weight_g": 0.0, "sums": dict.fromkeys(NUTRIENT_FIELDS, 0.0)},
        )
        stats["samples"] += 1
        stats["weight_g"] += weight
        eligible += 1
        for field in NUTRIENT_FIELDS:
            stats["sums"][field] += number(row["label"][field], field)
    categories = {
        key: {
            "samples": value["samples"],
            "per_100g": {f: 100 * value["sums"][f] / value["weight_g"] for f in NUTRIENT_FIELDS},
        }
        for key, value in groups.items()
        if value["samples"] >= config["nutrition"]["calibration_min_samples"]
    }
    result = {
        "source_split": "train",
        "training_manifest_sha256": sha256_file(Path(config["data_dir"]) / "train.jsonl"),
        "nutrition_signature": nutrition_signature(config),
        "resource_sha256": {
            key: sha256_file(config["resources"][key]) for key in ("table_path", "mapping_path")
        },
        "eligible_samples": eligible,
        "categories": categories,
        "method": (
            "mass-weighted class density, conditioned on cooking/state/sauce; "
            "only aligned single-item meals"
        ),
    }
    path = Path(config["resources"]["calibration_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"{eligible} eligible training samples; {len(categories)} calibrated classes; saved {path}",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/nutrition_agent.yaml")
    args = parser.parse_args()
    calibrate(load_yaml(args.config))
