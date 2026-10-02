import json
import math
from typing import Any

NUMERIC_FIELDS = ("total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g")
REQUIRED_FIELDS = ("items",) + NUMERIC_FIELDS


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_meal(value: Any) -> tuple[bool, list[str], dict[str, Any] | None]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return False, ["root_not_object"], None
    missing = [field for field in REQUIRED_FIELDS if field not in value]
    errors.extend(f"missing:{field}" for field in missing)
    items = value.get("items")
    if not isinstance(items, list):
        errors.append("items_not_array")
    else:
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"item_{index}_not_object")
                continue
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                errors.append(f"item_{index}_name_invalid")
            if not _number(item.get("weight_g")) or item["weight_g"] < 0:
                errors.append(f"item_{index}_weight_invalid")
    for field in NUMERIC_FIELDS:
        if field in value and (not _number(value[field]) or value[field] < 0):
            errors.append(f"{field}_invalid")
    extras = set(value) - set(REQUIRED_FIELDS)
    errors.extend(f"extra:{field}" for field in sorted(extras))
    if isinstance(items, list) and _number(value.get("total_weight_g")):
        weights = [float(item["weight_g"]) for item in items if isinstance(item, dict) and _number(item.get("weight_g"))]
        if len(weights) == len(items):
            total = sum(weights)
            expected = float(value["total_weight_g"])
            if abs(total - expected) > max(2.0, 0.05 * max(1.0, total)):
                errors.append("weight_total_inconsistent")
    return not errors, errors, value if isinstance(value, dict) else None


def parse_and_validate(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        return {"valid": False, "parsed": None, "errors": ["invalid_json", str(exc)]}
    valid, errors, parsed = validate_meal(value)
    return {"valid": valid, "errors": errors, "parsed": parsed}


def canonical_json(label: dict[str, Any]) -> str:
    return json.dumps(label, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def items_text(label: dict[str, Any]) -> str:
    names = [str(item.get("name", "")).strip() for item in label.get("items", []) if isinstance(item, dict) and item.get("name")]
    if not names:
        return "A meal contains no identified food."
    return "A meal contains " + ", ".join(names) + "."
