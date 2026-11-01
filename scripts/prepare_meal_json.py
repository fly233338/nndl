"""Convert the provided Nutrition5k JSONL splits to project manifests."""

import argparse
import json
from pathlib import Path

import yaml

from utils.jsonl import read_jsonl, write_jsonl
from experiments.meal_json.schema import validate_meal


def prepare(config_path: str) -> None:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    section = config["meal_json"]
    source = Path(section["annotations_dir"])
    image_root = Path(section["image_root"])
    output = Path(section["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    info = {"image_root": str(image_root)}
    for split in ("train", "val", "test"):
        samples = []
        for row in read_jsonl(source / f"{split}.jsonl"):
            sample = dict(row)
            sample["image_id"] = sample.pop("dish_id")
            sample["image_path"] = sample.pop("image")
            sample["label"] = sample.pop("target")
            valid, errors, _ = validate_meal(sample["label"])
            if not valid:
                raise ValueError(f"invalid label {sample['image_id']}: {errors}")
            if not (image_root / sample["image_path"]).is_file():
                raise FileNotFoundError(f"Image not found: {image_root / sample['image_path']}")
            samples.append(sample)
        write_jsonl(samples, output / f"{split}.jsonl")
        info[f"num_{split}"] = len(samples)
        print(f"{split}: {len(samples)} rows")
    (output / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data.yaml")
    args = parser.parse_args()
    prepare(args.config)
