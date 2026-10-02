import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from .prompts import direct_prompt, items_prompt, nutrients_prompt
from .schema import parse_and_validate
from utils.jsonl import write_jsonl


def extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1]


def qwen_predict(model_dir: str, image_path: str, mode: str = "direct", items: dict | None = None) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, trust_remote_code=True).to(device).eval()
    prompt = direct_prompt() if mode == "direct" else items_prompt() if mode == "items" else nutrients_prompt(items or {})
    encoded = processor(text=prompt, images=Image.open(image_path).convert("RGB"), return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=256, do_sample=False)
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


def two_stage_qwen(model_dir: str, image_path: str) -> str:
    item_raw = extract_json(qwen_predict(model_dir, image_path, "items"))
    item_data = json.loads(item_raw)
    nutrient_raw = extract_json(qwen_predict(model_dir, image_path, "nutrients", item_data))
    nutrient_data = json.loads(nutrient_raw)
    return json.dumps({"items": item_data.get("items", []), **nutrient_data}, ensure_ascii=False)


def infer_manifest(model_dir: str, manifest: str, image_root: str, output: str, mode: str = "direct"):
    from utils.jsonl import read_jsonl
    rows = []
    for row in read_jsonl(manifest):
        image_path = str(Path(image_root) / row["image_path"])
        raw = two_stage_qwen(model_dir, image_path) if mode == "two_stage" else qwen_predict(model_dir, image_path, mode)
        parsed = parse_and_validate(extract_json(raw))
        rows.append({"image_id": row["image_id"], "label": row.get("label"), "raw_output": raw, "parsed": parsed})
    write_jsonl(rows, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="results/meal_json/predictions.jsonl")
    parser.add_argument("--mode", choices=["direct", "two_stage"], default="direct")
    args = parser.parse_args()
    infer_manifest(args.model_dir, args.manifest, args.image_root, args.output, args.mode)
