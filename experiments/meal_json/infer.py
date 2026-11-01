import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
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
    processor.tokenizer.padding_side = "left"
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
def _predict_batch(model, processor, device, images, mode="direct", items=None, max_new_tokens=256):
    texts = []
    for index, image in enumerate(images):
        prompt = _prompt(mode, items[index] if items is not None else None)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}
    input_length = encoded["input_ids"].shape[1]
    generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generated[:, input_length:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


def _predict_loaded(model, processor, device, image_path, mode="direct", items=None, max_new_tokens=256):
    with Image.open(image_path) as image:
        images = [image.convert("RGB")]
    return _predict_batch(model, processor, device, images, mode, [items] if items is not None else None, max_new_tokens)[0]


def qwen_predict(adapter_dir: str, image_path: str, mode: str = "direct", items: dict | None = None, base_model_name: str | None = None) -> str:
    model, processor, device = load_qwen_adapter(adapter_dir, base_model_name)
    return _predict_loaded(model, processor, device, image_path, mode, items)


def _set_error(record, exc):
    record["raw_output"] = record["stage2_raw"] or record["stage1_raw"] or record["raw_output"]
    record["parsed"] = {"valid": False, "errors": ["inference_error", str(exc)], "parsed": None}


def infer_manifest(adapter_dir: str, manifest: str, image_root: str, output: str, mode: str = "direct", base_model_name: str | None = None, batch_size: int = 8):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model, processor, device = load_qwen_adapter(adapter_dir, base_model_name)
    rows = list(read_jsonl(manifest))
    records = []
    with tqdm(total=len(rows), desc=f"Qwen {mode}", unit="img") as progress:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            batch_records = []
            images, ready = [], []
            for row in batch:
                record = {"image_id": row["image_id"], "label": row.get("label"), "stage1_raw": None, "stage2_raw": None, "raw_output": "", "parsed": None}
                batch_records.append(record)
                try:
                    with Image.open(Path(image_root) / row["image_path"]) as image:
                        images.append(image.convert("RGB"))
                    ready.append(record)
                except Exception as exc:
                    _set_error(record, exc)
            try:
                if images:
                    outputs = _predict_batch(model, processor, device, images, "items" if mode == "two_stage" else mode)
                    next_images, next_records, next_items = [], [], []
                    for image, record, raw in zip(images, ready, outputs):
                        record["raw_output"] = raw
                        if mode != "two_stage":
                            record["parsed"] = parse_and_validate(extract_json(raw))
                            continue
                        record["stage1_raw"] = raw
                        try:
                            stage1_json = extract_json(raw)
                            if not stage1_json:
                                raise ValueError("stage1_invalid_json")
                            stage1 = json.loads(stage1_json)
                            next_images.append(image)
                            next_records.append(record)
                            next_items.append(stage1)
                        except Exception as exc:
                            _set_error(record, exc)
                    if next_images:
                        outputs = _predict_batch(model, processor, device, next_images, "nutrients", next_items)
                        for record, stage1, raw in zip(next_records, next_items, outputs):
                            record["stage2_raw"] = raw
                            try:
                                stage2_json = extract_json(raw)
                                if not stage2_json:
                                    raise ValueError("stage2_invalid_json")
                                merged = {"items": stage1.get("items", []), **json.loads(stage2_json)}
                                record["raw_output"] = json.dumps(merged, ensure_ascii=False)
                                record["parsed"] = parse_and_validate(record["raw_output"])
                            except Exception as exc:
                                _set_error(record, exc)
            except Exception as exc:
                for record in ready:
                    if record["parsed"] is None:
                        _set_error(record, exc)
            records.extend(batch_records)
            progress.update(len(batch))
    write_jsonl(records, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-name")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="results/meal_json/predictions.jsonl")
    parser.add_argument("--mode", choices=["direct", "two_stage"], default="direct")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    infer_manifest(args.adapter_dir, args.manifest, args.image_root, args.output, args.mode, args.base_model_name, args.batch_size)
