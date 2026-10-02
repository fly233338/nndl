from experiments.meal_json.schema import parse_and_validate


def validate_node(state: dict) -> dict:
    state["validation"] = parse_and_validate(state.get("raw_json", ""))
    return state


def nutrition_node(state: dict) -> dict:
    parsed = state.get("validation", {}).get("parsed") or {}
    state["nutrition"] = {field: parsed.get(field, 0.0) for field in ("total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g")}
    return state
