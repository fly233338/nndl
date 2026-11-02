import json


SCHEMA_TEXT = (
    '{"items":[{"name":"food","weight_g":0.0}],"total_weight_g":0.0,'
    '"total_calories_kcal":0.0,"protein_g":0.0,"fat_g":0.0,"carbohydrate_g":0.0}'
)


def direct_prompt() -> str:
    return "Return valid JSON only. Use this schema: " + SCHEMA_TEXT


def items_prompt() -> str:
    return 'Return valid JSON only with this schema: {"items":[{"name":"food","weight_g":0.0}]}'


def nutrients_prompt(items: dict) -> str:
    return (
        "Given these predicted items, return valid JSON containing only the five numeric "
        "nutrition fields: " + json.dumps(items, ensure_ascii=False)
    )


def review_prompt(context: dict) -> str:
    """结合图片和检查提示复核完整草稿，允许修正所有字段。"""
    return (
        "Review the provisional meal JSON draft against the image. The draft may be wrong. "
        "Check visible food names, omitted foods, unsupported additions, duplicate items, "
        "portion weights, and all five numeric fields. Correct only errors supported by "
        "the image or the checks; otherwise preserve the draft. Do not invent hidden "
        "ingredients. Item weights need not sum to total_weight_g; do not force equality. "
        "The energy relation 4*protein_g + 9*fat_g + 4*carbohydrate_g is approximate; "
        "do not force equality. You may revise both items and nutrition values. "
        "Return the COMPLETE JSON only, without explanations, using this schema: "
        + SCHEMA_TEXT
        + "\nChecks (warnings are not proof of an error): "
        + json.dumps(context["checks"], ensure_ascii=False)
        + "\nDraft to review:\n"
        + context["draft"]
    )
