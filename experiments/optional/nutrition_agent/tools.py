"""提供营养查表、按克计算、重量归一和 RGB 局部裁剪工具。"""

import argparse
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from experiments.task2_meal_json.schema import NUMERIC_FIELDS
from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import read_jsonl

NUTRIENT_FIELDS = NUMERIC_FIELDS[1:]


def number(value, name, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be finite and nonnegative (positive={positive})")
    return float(value)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()))


class NutritionDB:
    """按名称和食品状态检索营养密度，并确定性地计算整餐营养。"""

    def __init__(self, config, calibrated=True):
        self.settings = config["nutrition"]
        self.rows = {r["food_id"]: r for r in read_jsonl(config["resources"]["table_path"])}
        if not self.rows or any("portions" not in row for row in self.rows.values()):
            raise ValueError(
                "Old or empty nutrition table; "
                "rerun prepare_resources to include standard portions",
            )
        self.mapping = load_yaml(config["resources"]["mapping_path"])
        self.calibration = {}
        if calibrated and self.settings["use_calibration"]:
            calibration = json.loads(
                Path(config["resources"]["calibration_path"]).read_text(encoding="utf-8"),
            )
            for key in ("table_path", "mapping_path"):
                if calibration["resource_sha256"][key] != sha256_file(config["resources"][key]):
                    raise ValueError(
                        "Nutrition priors do not match the frozen table/mapping; "
                        "rebuild calibration",
                    )
            self.calibration = calibration["categories"]

    def normalize(self, text):
        text = text.strip().lower()
        return self.mapping["aliases"].get(text, text)

    def lookup(
        self,
        food_name,
        ingredients,
        cooking_method,
        food_state="cooked",
        sauce="unknown",
        mixed_dish=False,
    ):
        """排除明确的状态冲突，返回排序候选和是否需要模型选择。"""
        name = self.normalize(food_name)
        method = self.mapping["methods"].get(cooking_method, cooking_method)
        state = self.mapping["states"].get(food_state, food_state)
        query = tokens(name)
        ingredient_tokens = tokens(" ".join(self.normalize(x) for x in ingredients))
        ranked = []
        for row in self.rows.values():
            if any(row["nutrients"][f] is None for f in NUTRIENT_FIELDS):
                continue
            description = row["description"].lower()
            if "as ingredient" in description:
                continue
            words = tokens(description)
            override = self.mapping.get("food_overrides", {}).get(row["food_id"], {})
            if words & set(self.mapping["raw_words"]):
                inferred_state = "raw"
            elif words & set(self.mapping["cooked_words"]):
                inferred_state = "cooked"
            elif words & set(self.mapping["dry_words"]):
                inferred_state = "raw"
            else:
                inferred_state = "unknown"
            row_state = override.get("state", inferred_state)
            if state != "unknown" and row_state != "unknown" and state != row_state:
                continue
            methods = set(override.get("methods", [])) or (
                words & set(self.mapping["cooked_words"])
            )
            requested = set(self.mapping["method_terms"].get(method, []))
            specific = methods - {"cooked", "prepared"}
            if requested and specific and not requested & specific:
                continue
            if method == "deep_fried" and words & {"stir", "sauteed"}:
                continue
            if method == "stir_fried" and "deep" in words:
                continue
            if "without sauce" in description or "no sauce" in description:
                inferred_sauce = "without"
            elif words & {"sauce", "gravy"}:
                inferred_sauce = "with"
            else:
                inferred_sauce = "unknown"
            has_sauce = override.get("sauce", inferred_sauce)
            if sauce == "without" and has_sauce == "with":
                continue
            if sauce == "with" and has_sauce == "without":
                continue
            overlap = len(query & words) / max(1, len(query))
            if not overlap:
                continue
            if mixed_dish and overlap < 1.0 and not override.get("finished_dish", False):
                continue
            score = 3 * overlap + SequenceMatcher(None, name, description).ratio()
            if state != "unknown" and row_state == state:
                score += 1.5
            elif row_state == "unknown":
                score -= 0.5
            score += len(query & tokens(description.split(",")[0])) / max(1, len(query))
            score += len(ingredient_tokens & words) / max(1, len(ingredient_tokens))
            score -= len(
                (words & set(self.mapping["ingredient_terms"])) - ingredient_tokens - query,
            )
            score += float(bool(requested & methods)) + float(
                mixed_dish and row["data_type"] == "Survey (FNDDS)",
            )
            score += float(sauce == "with" and has_sauce == "with")
            portions = row.get("portions", [])
            preferred = []
            for term in self.settings["portion_preference"]:
                for portion in portions:
                    portion_description = portion["description"].lower()
                    if term in portion_description and not portion_description.startswith(
                        "guideline",
                    ):
                        preferred.append(portion)

            if preferred:
                portion = preferred[0]
            else:
                fallback_portion = portions[0] if portions else None
                portion = next(
                    (portion for portion in portions if portion["amount"] == 1),
                    fallback_portion,
                )
            ranked.append(
                {
                    **row,
                    "standard_portion_g": portion["gram_weight"] if portion else None,
                    "match_score": round(score, 4),
                    "state_match": row_state,
                    "match_basis": {
                        "name_overlap": overlap,
                        "cooking_terms": sorted(methods),
                        "sauce": has_sauce,
                        "approximate": row_state == "unknown"
                        or (bool(requested) and not requested & methods),
                    },
                },
            )
        ranked.sort(key=lambda row: (-row["match_score"], row["food_id"]))
        candidates = ranked[: self.settings["top_k"]]
        if not candidates:
            raise ValueError(
                f"No state-compatible complete entry for {food_name}; extend the mapping/table",
            )
        density_spreads = []
        for field in NUTRIENT_FIELDS:
            densities = [candidate["nutrients"][field] for candidate in candidates]
            denominator = max(self.settings["density_floor"], candidates[0]["nutrients"][field])
            density_spreads.append((max(densities) - min(densities)) / denominator)
        spread = max(density_spreads)
        return {
            "candidates": candidates,
            "requires_choice": spread > self.settings["density_spread_threshold"],
            "relative_density_spread": spread,
            "default_food_id": candidates[0]["food_id"],
        }

    def density(self, food_id, category):
        """按配置混合原始数据库密度和可用的训练类别统计。"""
        values = dict(self.rows[food_id]["nutrients"])
        prior = self.calibration.get(category)
        if prior:
            alpha = self.settings["calibration_strength"]
            values = {
                f: (1 - alpha) * values[f] + alpha * prior["per_100g"][f] for f in NUTRIENT_FIELDS
            }
        return values

    def calculate(self, items_with_weight):
        if not items_with_weight:
            raise ValueError("At least one food item is required")
        output = {"items": [], **dict.fromkeys(NUMERIC_FIELDS, 0.0)}
        for item in items_with_weight:
            weight = number(item["weight_g"], "weight_g")
            density = self.density(item["food_id"], item.get("calibration_key", ""))
            output["items"].append({"name": item["name"], "weight_g": weight})
            output["total_weight_g"] += weight
            for field in NUTRIENT_FIELDS:
                output[field] += weight / 100 * number(density[field], field)
        return output


def normalize_weights(items, total_weight_g):
    """将正的相对份量归一到独立估计的总重，末项吸收浮点余差。"""
    total = number(total_weight_g, "total_weight_g", positive=True)
    shares = [number(item["relative_weight"], "relative_weight", positive=True) for item in items]
    denominator = sum(shares)
    if not denominator:
        raise ValueError("Empty item allocation")
    result = [
        {**item, "weight_g": total * share / denominator} for item, share in zip(items, shares)
    ]
    result[-1]["weight_g"] = total - sum(item["weight_g"] for item in result[:-1])
    return result


def crop_image(image_path, bbox, max_size):
    """按归一化坐标裁剪原图，仅供食品识别复核，不用于重新估重。"""
    if len(bbox) != 4 or not all(
        isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in bbox
    ):
        raise ValueError("bbox must be normalized [left, top, right, bottom]")
    left, top, right, bottom = bbox
    if left >= right or top >= bottom:
        raise ValueError("bbox must have positive area")
    with Image.open(image_path) as image:
        width, height = image.size
        crop = image.convert("RGB").crop(
            (
                int(left * width),
                int(top * height),
                max(int(right * width), int(left * width) + 1),
                max(int(bottom * height), int(top * height) + 1),
            ),
        )
    scale = max_size / max(crop.size)
    return crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/nutrition_agent.yaml")
    parser.add_argument("--food")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--state", default="cooked")
    parser.add_argument(
        "--items",
        help="JSON array: food_id, name, calibration_key (optional), weight_g",
    )
    parser.add_argument(
        "--uncalibrated",
        action="store_true",
        help="Use raw USDA densities before offline calibration",
    )
    args = parser.parse_args()
    db = NutritionDB(load_yaml(args.config), calibrated=not args.uncalibrated)
    result = (
        db.calculate(json.loads(Path(args.items).read_text(encoding="utf-8")))
        if args.items
        else db.lookup(args.food, [], args.method, args.state)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
