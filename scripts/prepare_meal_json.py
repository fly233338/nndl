"""Normalize course-platform meal annotations into the FoodLens schema."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from utils.jsonl import write_jsonl
from experiments.meal_json.schema import validate_meal


def _read_source(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(row: dict[str, Any], fields: dict[str, str], image_root: Path) -> dict[str, Any]:
    image = row[fields["image"]]
    image_id = row.get(fields.get("image_id", "image_id"), Path(str(image)).stem)
    raw_items = row[fields["items"]]
    if isinstance(raw_items, str):
        raw_items = json.loads(raw_items)
    items = [
        {"name": str(item[fields["item_name"]]).strip(), "weight_g": float(item[fields["item_weight"]])}
        for item in raw_items
    ]
    label = {
        "items": items,
        "total_weight_g": float(row[fields["total_weight"]]),
        "total_calories_kcal": float(row[fields["calories"]]),
        "protein_g": float(row[fields["protein"]]),
        "fat_g": float(row[fields["fat"]]),
        "carbohydrate_g": float(row[fields["carbohydrate"]]),
    }
    image_path = str(image)
    if not (image_root / image_path).exists() and not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return {"image_id": str(image_id), "image_path": image_path, "label": label}


def prepare(config_path: str) -> None:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    section = config["meal_json"]
    fields = section["fields"]
    rows = _read_source(Path(section["source_path"]))
    image_root = Path(section["image_root"])
    samples = [normalize(row, fields, image_root) for row in rows]
    for sample in samples:
        valid, errors, _ = validate_meal(sample["label"])
        if not valid:
            raise ValueError(f"invalid label {sample['image_id']}: {errors}")
    groups = {}
    for sample in samples:
        groups.setdefault(sample["image_id"], []).append(sample)
    group_ids = list(groups)
    random.Random(section.get("split_seed", 2026)).shuffle(group_ids)
    val_ratio, test_ratio = section.get("val_ratio", 0.1), section.get("test_ratio", 0.1)
    test_count = int(len(group_ids) * test_ratio)
    val_count = int(len(group_ids) * val_ratio)
    test_ids = set(group_ids[:test_count])
    val_ids = set(group_ids[test_count:test_count + val_count])
    train_ids = set(group_ids[test_count + val_count:])
    test = [sample for image_id in test_ids for sample in groups[image_id]]
    val = [sample for image_id in val_ids for sample in groups[image_id]]
    train = [sample for image_id in train_ids for sample in groups[image_id]]
    output = Path(section["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, output / "train.jsonl")
    write_jsonl(val, output / "val.jsonl")
    write_jsonl(test, output / "test.jsonl")
    (output / "dataset_info.json").write_text(
        json.dumps({"num_train": len(train), "num_val": len(val), "num_test": len(test), "image_root": str(image_root)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data.yaml")
    args = parser.parse_args()
    prepare(args.config)
