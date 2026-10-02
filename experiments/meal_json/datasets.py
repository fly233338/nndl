from pathlib import Path

from utils.jsonl import read_jsonl


def load_rows(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def build_prompt(row: dict, mode: str = "direct") -> str:
    if mode == "items":
        return "Identify every food item and its weight in grams. Return JSON with only an items array."
    if mode == "nutrients":
        return "Using the image and the proposed food items, return only total_weight_g, total_calories_kcal, protein_g, fat_g, and carbohydrate_g as JSON."
    return "Analyze the food image and return only the required meal JSON schema. Do not add explanations."
