import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from utils.jsonl import read_jsonl, write_jsonl
from .prompts import direct_prompt, items_prompt, nutrients_prompt
from .schema import parse_and_validate


def extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else ""


def load_qwen_adapter(adapter_dir: str, base_model_name: str | None = None):
    adapter_path = Path(adapter_dir)
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    base_model_name = base_model_name or adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("base model name is missing from adapter_config.json")
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForImageTextToText.from_pretrained(base_model_name, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    return model, processor, device


def _prompt(mode, items=None):
    if mode == "items":
        return items_prompt()
    if mode == "nutrients":
        return nutrients_prompt(items or {})
    return direct_prompt()


@torch.no_grad()
def _predict_loaded(model, processor, device, image_path, mode="direct", items=None, max_new_tokens=256):
    image = Image.open(image_path).convert("RGB")
    prompt = _prompt(mode, items)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=text, images=image, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}
    input_length = encoded["input_ids"].shape[1]
    generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generated[:, input_length:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def qwen_predict(adapter_dir: str, image_path: str, mode: str = "direct", items: dict | None = None, base_model_name: str | None = None) -> str:
    model, processor, device = load_qwen_adapter(adapter_dir, base_model_name)
    return _predict_loaded(model, processor, device, image_path, mode, items)


def _two_stage(model, processor, device, image_path):
    stage1_raw = _predict_loaded(model, processor, device, image_path, "items")
    stage1_json = extract_json(stage1_raw)
    if not stage1_json:
        raise ValueError("stage1_invalid_json")
    stage1 = json.loads(stage1_json)
    stage2_raw = _predict_loaded(model, processor, device, image_path, "nutrients", stage1)
    stage2_json = extract_json(stage2_raw)
    if not stage2_json:
        raise ValueError("stage2_invalid_json")
    stage2 = json.loads(stage2_json)
    merged = {"items": stage1.get("items", []), **stage2}
    return json.dumps(merged, ensure_ascii=False), stage1_raw, stage2_raw


def infer_manifest(adapter_dir: str, manifest: str, image_root: str, output: str, mode: str = "direct", base_model_name: str | None = None):
    model, processor, device = load_qwen_adapter(adapter_dir, base_model_name)
    records = []
    for row in read_jsonl(manifest):
        image_path = str(Path(image_root) / row["image_path"])
        record = {"image_id": row["image_id"], "label": row.get("label"), "stage1_raw": None, "stage2_raw": None}
        try:
            if mode == "two_stage":
                raw, record["stage1_raw"], record["stage2_raw"] = _two_stage(model, processor, device, image_path)
            else:
                raw = _predict_loaded(model, processor, device, image_path, mode)
            record["raw_output"] = raw
            record["parsed"] = parse_and_validate(extract_json(raw))
        except Exception as exc:
            record["raw_output"] = record.get("stage2_raw") or record.get("stage1_raw") or ""
            record["parsed"] = {"valid": False, "errors": ["inference_error", str(exc)], "parsed": None}
        records.append(record)
    write_jsonl(records, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-name")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="results/meal_json/predictions.jsonl")
    parser.add_argument("--mode", choices=["direct", "two_stage"], default="direct")
    args = parser.parse_args()
    infer_manifest(args.adapter_dir, args.manifest, args.image_root, args.output, args.mode, args.base_model_name)
