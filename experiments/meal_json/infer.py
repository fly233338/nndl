"""加载 Qwen LoRA Adapter，支持批量直接生成和两阶段生成。"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.config import load_yaml
from utils.jsonl import read_jsonl, write_jsonl
from .prompts import direct_prompt, items_prompt, nutrients_prompt
from .schema import parse_and_validate


def extract_json(text: str) -> str:
    """提取首个左花括号到最后一个右花括号之间的文本。"""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else ""


def load_qwen_adapter(adapter_dir: str, base_model_name: str | None = None):
    """加载基础模型、Adapter 和 processor，并切换为生成所需的左侧填充。"""
    adapter_path = Path(adapter_dir)
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    base_model_name = base_model_name or adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("base model name is missing from adapter_config.json")

    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    # 左侧填充保证每条输入的末尾都是有效 token。
    processor.tokenizer.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()

    return model, processor, device


def _prompt(mode, items=None):
    if mode == "items":
        return items_prompt()

    if mode == "nutrients":
        return nutrients_prompt(items or {})

    return direct_prompt()


@torch.no_grad()
def _predict_batch(
    model,
    processor,
    device,
    images,
    mode="direct",
    items=None,
    max_new_tokens=768,
    image_size=448,
):
    """批量生成文本；第二阶段的 items 与图片按相同顺序对应。"""
    # 与训练使用相同的图片尺寸，两阶段均使用这份输入。
    images = [image.resize((image_size, image_size)) for image in images]
    texts = []
    for index, image in enumerate(images):
        prompt = _prompt(mode, items[index] if items is not None else None)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            },
        ]
        texts.append(
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}

    input_length = encoded["input_ids"].shape[1]
    generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    # 生成结果包含补齐后的输入前缀，解码时只保留新生成部分。
    new_tokens = generated[:, input_length:]

    return processor.batch_decode(new_tokens, skip_special_tokens=True)


def _predict_loaded(
    model,
    processor,
    device,
    image_path,
    mode="direct",
    items=None,
    max_new_tokens=768,
    image_size=448,
):
    """复用批量生成逻辑处理单张图片。"""
    with Image.open(image_path) as image:
        images = [image.convert("RGB")]

    return _predict_batch(
        model,
        processor,
        device,
        images,
        mode,
        [items] if items is not None else None,
        max_new_tokens,
        image_size,
    )[0]


def qwen_predict(
    adapter_dir: str,
    image_path: str,
    mode: str = "direct",
    items: dict | None = None,
    base_model_name: str | None = None,
    config_path: str = "config/meal_json.yaml",
) -> str:
    """单图推理入口，返回模型生成的原始文本。"""
    model, processor, device = load_qwen_adapter(adapter_dir, base_model_name)
    qwen = load_yaml(config_path)["qwen"]
    return _predict_loaded(
        model,
        processor,
        device,
        image_path,
        mode,
        items,
        max_new_tokens=qwen.get("max_new_tokens", 768),
        image_size=qwen.get("image_size", 448),
    )


def _set_error(record, exc):
    """保留已有阶段输出，并记录当前图片的推理错误。"""
    record["raw_output"] = record["stage2_raw"] or record["stage1_raw"] or record["raw_output"]
    record["parsed"] = {"valid": False, "errors": ["inference_error", str(exc)], "parsed": None}


def infer_manifest(
    adapter_dir: str,
    manifest: str,
    image_root: str,
    output: str,
    mode: str = "direct",
    base_model_name: str | None = None,
    batch_size: int = 8,
    config_path: str = "config/meal_json.yaml",
):
    """批量执行直接或两阶段推理，保留清单顺序和失败记录。"""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    qwen = load_yaml(config_path)["qwen"]
    model, processor, device = load_qwen_adapter(adapter_dir, base_model_name)
    rows = list(read_jsonl(manifest))
    records = []

    with tqdm(total=len(rows), desc=f"Qwen {mode}", unit="img") as progress:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            batch_records = []
            images, ready = [], []

            for row in batch:
                record = {
                    "image_id": row["image_id"],
                    "label": row.get("label"),
                    "stage1_raw": None,
                    "stage2_raw": None,
                    "raw_output": "",
                    "parsed": None,
                }
                batch_records.append(record)
                try:
                    with Image.open(Path(image_root) / row["image_path"]) as image:
                        images.append(image.convert("RGB"))
                    ready.append(record)
                except Exception as exc:
                    _set_error(record, exc)

            try:
                if images:
                    outputs = _predict_batch(
                        model,
                        processor,
                        device,
                        images,
                        "items" if mode == "two_stage" else mode,
                        max_new_tokens=qwen.get("max_new_tokens", 768),
                        image_size=qwen.get("image_size", 448),
                    )
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

                    # 只保留第一阶段解析成功的图片，并同步保留对应食材。
                    if next_images:
                        outputs = _predict_batch(
                            model,
                            processor,
                            device,
                            next_images,
                            "nutrients",
                            next_items,
                            max_new_tokens=qwen.get("max_new_tokens", 768),
                            image_size=qwen.get("image_size", 448),
                        )
                        for record, stage1, raw in zip(next_records, next_items, outputs):
                            record["stage2_raw"] = raw
                            try:
                                stage2_json = extract_json(raw)
                                if not stage2_json:
                                    raise ValueError("stage2_invalid_json")

                                merged = {
                                    "items": stage1.get("items", []),
                                    **json.loads(stage2_json),
                                }
                                record["raw_output"] = json.dumps(merged, ensure_ascii=False)
                                record["parsed"] = parse_and_validate(record["raw_output"])
                            except Exception as exc:
                                _set_error(record, exc)
            except Exception as exc:
                for record in ready:
                    if record["parsed"] is None:
                        _set_error(record, exc)

            records.extend(batch_records)
            # 两阶段均结束后再计数，每张图片只更新一次进度。
            progress.update(len(batch))

    write_jsonl(records, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/meal_json.yaml")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-name")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="results/meal_json/predictions.jsonl")
    parser.add_argument("--mode", choices=["direct", "two_stage"], default="direct")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    infer_manifest(
        args.adapter_dir,
        args.manifest,
        args.image_root,
        args.output,
        args.mode,
        args.base_model_name,
        args.batch_size,
        args.config,
    )
