SCHEMA_TEXT = '{"items":[{"name":"food","weight_g":0.0}],"total_weight_g":0.0,"total_calories_kcal":0.0,"protein_g":0.0,"fat_g":0.0,"carbohydrate_g":0.0}'


def direct_prompt() -> str:
    return "Return valid JSON only. Use this schema: " + SCHEMA_TEXT


def items_prompt() -> str:
    return 'Return valid JSON only with this schema: {"items":[{"name":"food","weight_g":0.0}]}'


def nutrients_prompt(items: dict) -> str:
    import json
    return "Given these predicted items, return valid JSON containing only the five numeric nutrition fields: " + json.dumps(items, ensure_ascii=False)
