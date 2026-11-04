"""将固定版本 USDA 数据整理为每 100 克的营养表。"""

import argparse
import json
import math
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import write_jsonl

NUTRIENTS = {1003: "protein_g", 1004: "fat_g", 1005: "carbohydrate_g", 1008: "total_calories_kcal"}


def normalize_food(food, source):
    """统一营养单位和标准份量，缺失营养保留为空值。"""
    values = {field: None for field in NUTRIENTS.values()}
    energy = {}
    for entry in food.get("foodNutrients", []):
        nutrient = entry.get("nutrient", {})
        nutrient_id = nutrient.get("id")
        amount = entry.get("amount")
        if (
            not isinstance(amount, (int, float))
            or isinstance(amount, bool)
            or not math.isfinite(amount)
            or amount < 0
        ):
            continue
        unit = str(nutrient.get("unitName", "")).lower()
        if nutrient_id in (1008, 2047, 2048) and unit == "kcal":
            energy[nutrient_id] = float(amount)
        elif nutrient_id == 1062 and unit == "kj":
            energy[nutrient_id] = float(amount) / 4.184
        elif nutrient_id in NUTRIENTS and nutrient_id != 1008:
            if unit == "g":
                values[NUTRIENTS[nutrient_id]] = float(amount)
            elif unit == "mg":
                values[NUTRIENTS[nutrient_id]] = float(amount) / 1000
    for nutrient_id in (1008, 2048, 2047, 1062):
        if nutrient_id in energy:
            values["total_calories_kcal"] = energy[nutrient_id]
            break
    portions = []
    for portion in food.get("foodPortions", []):
        weight = portion.get("gramWeight")
        amount = portion.get("amount", 1)
        if (
            isinstance(weight, (int, float))
            and math.isfinite(weight)
            and weight > 0
            and isinstance(amount, (int, float))
            and amount > 0
        ):
            portions.append(
                {
                    "gram_weight": float(weight),
                    "amount": float(amount),
                    "description": portion.get("portionDescription")
                    or portion.get("modifier")
                    or portion.get("measureUnit", {}).get("name", ""),
                },
            )
    return {
        "food_id": str(food["fdcId"]),
        "name": food["description"],
        "description": food["description"],
        "data_type": food.get("dataType"),
        "basis_g": 100.0,
        "food_code": str(food.get("foodCode", food["fdcId"])),
        "portions": portions,
        "nutrients": values,
        "source": source["name"],
        "source_version": source["version"],
    }


def _read_foods(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".json")]
            if len(names) != 1:
                raise ValueError(f"Expected one USDA JSON file in {path}, found {len(names)}")
            with archive.open(names[0]) as stream:
                data = json.load(stream)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for foods in data.values():
        if isinstance(foods, list) and foods and isinstance(foods[0], dict) and "fdcId" in foods[0]:
            return foods
    raise ValueError(f"No food records in {path}")


def prepare(config_path, download=False):
    """读取固定来源的 USDA 数据，输出营养表和来源指纹。"""
    resources = load_yaml(config_path)["resources"]
    rows, sources = {}, []
    for source in resources["sources"]:
        path = Path(source["path"])
        if not path.is_file():
            if not download:
                raise FileNotFoundError(
                    f"Missing {path}. Run this command with --download "
                    "or place the USDA archive there.",
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            try:
                with (
                    urlopen(source["url"], timeout=resources["download_timeout"]) as response,
                    partial.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        foods = _read_foods(path)
        for food in foods:
            row = normalize_food(food, source)
            if row["food_id"] in rows:
                raise ValueError(f"Duplicate USDA food ID: {row['food_id']}")
            rows[row["food_id"]] = row
        sources.append({**source, "sha256": sha256_file(path), "food_count": len(foods)})
        print(f"{source['name']}: {len(foods)} foods", flush=True)
    table = Path(resources["table_path"])
    write_jsonl((rows[key] for key in sorted(rows)), table)
    metadata = {
        "schema_version": 2,
        "basis_g": 100,
        "food_count": len(rows),
        "sources": sources,
        "table_sha256": sha256_file(table),
    }
    destination = Path(resources["metadata_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} foods to {table}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/nutrition_agent.yaml")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing fixed-version USDA archives",
    )
    args = parser.parse_args()
    prepare(args.config, args.download)
