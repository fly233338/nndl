import json
from typing import Any

NUMERIC_FIELDS = ("total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g")
REQUIRED_FIELDS = ("items",) + NUMERIC_FIELDS


def validate_meal(value: Any) -> tuple[bool, list[str], dict[str, Any] | None]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return False, ["root_not_object"], None
    missing = [field for field in REQUIRED_FIELDS if field not in value]
    errors.extend(f"missing:{field}" for field in missing)
    if not isinstance(value.get("items"), list):
        errors.append("items_not_array")
    else:
        for index, item in enumerate(value["items"]):
            if not isinstance(item, dict):
                errors.append(f"item_{index}_not_object")
                continue
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                errors.append(f"item_{index}_name_invalid")
            if not isinstance(item.get("weight_g"), (int, float)) or item.get("weight_g", -1) < 0:
                errors.append(f"item_{index}_weight_invalid")
    for field in NUMERIC_FIELDS:
        if field in value and (not isinstance(value[field], (int, float)) or value[field] < 0):
            errors.append(f"{field}_invalid")
    extras = set(value) - set(REQUIRED_FIELDS)
    errors.extend(f"extra:{field}" for field in sorted(extras))
    if isinstance(value.get("items"), list) and isinstance(value.get("total_weight_g"), (int, float)):
        total = sum(float(item.get("weight_g", 0)) for item in value["items"] if isinstance(item, dict))
        if abs(total - float(value["total_weight_g"])) > max(2.0, 0.05 * max(1.0, total)):
            errors.append("weight_total_inconsistent")
    return not errors, errors, value if not errors else None


def parse_and_validate(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"valid": False, "errors": ["invalid_json", str(exc)], "parsed": None}
    valid, errors, parsed = validate_meal(value)
    return {"valid": valid, "errors": errors, "parsed": parsed}


def canonical_json(label: dict[str, Any]) -> str:
    return json.dumps(label, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def items_text(label: dict[str, Any]) -> str:
    names = [str(item.get("name", "")).strip() for item in label.get("items", [])]
    names = [name for name in names if name]
    if not names:
        return "A meal contains no identified food."
    return "A meal contains " + ", ".join(names) + "."
