"""加载原始 Qwen 或 LoRA Adapter，复用相同的膳食 JSON 推理流程。"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.config import load_yaml
from utils.jsonl import read_jsonl, write_jsonl

from .prompts import direct_prompt, items_prompt, nutrients_prompt, review_prompt
from .schema import parse_and_validate


def extract_json(text: str) -> str:
    """提取首个左花括号到最后一个右花括号之间的文本。"""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else ""


def load_qwen_model(base_model_name: str | None = None, adapter_dir: str | None = None):
    """无 Adapter 时只加载原始模型；有 Adapter 时沿用已有的微调模型加载方式。"""
    if adapter_dir is not None:
        adapter_path = Path(adapter_dir)
        adapter_config = json.loads(
            (adapter_path / "adapter_config.json").read_text(encoding="utf-8"),
        )
        base_model_name = base_model_name or adapter_config.get("base_model_name_or_path")

    if not base_model_name:
        raise ValueError("Supply a base model name or an Adapter with base_model_name_or_path")

    if adapter_dir is None and (Path(base_model_name) / "adapter_config.json").is_file():
        raise ValueError("Zero-shot requires the original base model directory, not an Adapter")

    processor_source = adapter_dir if adapter_dir is not None else base_model_name
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
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
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model.to(device).eval()

    return model, processor, device


def load_qwen_adapter(adapter_dir: str, base_model_name: str | None = None):
    """保留已有 Adapter 加载接口。"""
    return load_qwen_model(base_model_name=base_model_name, adapter_dir=adapter_dir)


def _prompt(mode, items=None):
    if mode == "items":
        return items_prompt()

    if mode == "nutrients":
        return nutrients_prompt(items or {})

    if mode == "review":
        return review_prompt(items)

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
    """批量生成文本；第二阶段的条件与图片按相同顺序对应。"""
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
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
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
    adapter_dir: str | None,
    image_path: str,
    mode: str = "direct",
    items: dict | None = None,
    base_model_name: str | None = None,
    config_path: str = "config/task2_meal_json/meal_json.yaml",
) -> str:
    """单图推理返回原始文本；adapter_dir 为 None 时使用原始模型。"""
    qwen = load_yaml(config_path)["qwen"]
    if adapter_dir is None:
        base_model_name = base_model_name or qwen["model_name"]
    model, processor, device = load_qwen_model(base_model_name, adapter_dir)
    raw = _predict_loaded(
        model,
        processor,
        device,
        image_path,
        mode,
        items,
        max_new_tokens=qwen.get("max_new_tokens", 768),
        image_size=qwen.get("image_size", 448),
    )
    if mode != "draft_review":
        return raw

    draft = parse_and_validate(extract_json(raw))
    record = {"raw_output": raw, "parsed": draft, "draft_validation": draft}
    try:
        reviewed = _predict_loaded(
            model,
            processor,
            device,
            image_path,
            "review",
            {"draft": raw, "checks": _review_checks(draft)},
            max_new_tokens=qwen.get("max_new_tokens", 768),
            image_size=qwen.get("image_size", 448),
        )
    except Exception:
        if draft["valid"]:
            return raw
        raise

    _apply_review(record, reviewed)
    return record["raw_output"]


def _review_checks(draft):
    """只检查草稿本身，不读取真实标签；数值关系仅作为复核提示。"""
    checks = list(draft["errors"])
    if not draft["valid"]:
        return checks

    meal = draft["parsed"]
    names = [item["name"].strip().lower() for item in meal["items"]]
    if len(names) != len(set(names)):
        checks.append("Duplicate food names; check whether they describe the same portion.")
    if any(item["weight_g"] > meal["total_weight_g"] for item in meal["items"]):
        checks.append("An item weight exceeds total_weight_g; check the portion estimates.")
    if (
        sum(meal[field] for field in ("protein_g", "fat_g", "carbohydrate_g"))
        > meal["total_weight_g"]
    ):
        checks.append("Macronutrient mass exceeds total_weight_g; check units and estimates.")
    return checks


def _apply_review(record, raw):
    """保留复核原文；复核无效时仅回退到有效草稿，失败记录仍保留。"""
    reviewed = parse_and_validate(extract_json(raw))
    record["review_validation"] = reviewed
    record["review_error"] = None if reviewed["valid"] else "; ".join(reviewed["errors"])
    record["review_fallback"] = not reviewed["valid"] and record["draft_validation"]["valid"]
    if not record["review_fallback"]:
        record["raw_output"] = raw
        record["parsed"] = reviewed


def _set_error(record, exc):
    """保留已有阶段输出，并记录当前图片的推理错误。"""
    record["raw_output"] = record["stage2_raw"] or record["stage1_raw"] or record["raw_output"]
    record["parsed"] = {"valid": False, "errors": ["inference_error", str(exc)], "parsed": None}


def infer_manifest(
    adapter_dir: str | None,
    manifest: str,
    image_root: str,
    output: str,
    mode: str = "direct",
    base_model_name: str | None = None,
    batch_size: int = 8,
    config_path: str = "config/task2_meal_json/meal_json.yaml",
):
    """批量执行所选模式；无 Adapter 时执行 zero-shot，标签只用于后续评测。"""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    qwen = load_yaml(config_path)["qwen"]
    if adapter_dir is None:
        base_model_name = base_model_name or qwen["model_name"]
    model, processor, device = load_qwen_model(base_model_name, adapter_dir)
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
                        if mode == "draft_review":
                            draft = parse_and_validate(extract_json(raw))
                            record["stage1_raw"] = raw
                            record["parsed"] = draft
                            record["draft_validation"] = draft
                            record["review_checks"] = _review_checks(draft)
                            record["review_fallback"] = False
                            record["review_error"] = None
                            next_images.append(image)
                            next_records.append(record)
                            next_items.append({"draft": raw, "checks": record["review_checks"]})
                            continue
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

                    # 食材两阶段只处理可解析食材；草稿复核也允许修复无效草稿。
                    if next_images:
                        outputs = _predict_batch(
                            model,
                            processor,
                            device,
                            next_images,
                            "review" if mode == "draft_review" else "nutrients",
                            next_items,
                            max_new_tokens=qwen.get("max_new_tokens", 768),
                            image_size=qwen.get("image_size", 448),
                        )
                        for record, stage1, raw in zip(next_records, next_items, outputs):
                            record["stage2_raw"] = raw
                            if mode == "draft_review":
                                _apply_review(record, raw)
                                continue
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
                    if mode == "draft_review" and record["stage1_raw"] is not None:
                        record["review_error"] = str(exc)
                        record["review_fallback"] = record["draft_validation"]["valid"]
                    elif record["parsed"] is None:
                        _set_error(record, exc)

            records.extend(batch_records)
            # 两阶段均结束后再计数，每张图片只更新一次进度。
            progress.update(len(batch))

    write_jsonl(records, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/task2_meal_json/meal_json.yaml")
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--adapter-dir")
    model_source.add_argument(
        "--zero-shot",
        action="store_true",
        help="Use the original Qwen model without loading a LoRA Adapter",
    )
    parser.add_argument("--base-model-name")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=["direct", "two_stage", "draft_review"], default="direct")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.zero_shot:
        config = load_yaml(args.config)
        args.base_model_name = args.base_model_name or config["qwen"]["model_name"]
        if args.output is None:
            args.output = str(
                Path(config["zero_shot_output_dir"])
                / Path(args.base_model_name).name
                / args.mode
                / f"{Path(args.manifest).stem}_predictions.jsonl",
            )
    if args.output is None:
        if args.mode == "draft_review":
            args.output = str(
                Path(args.adapter_dir).parent
                / "draft_review"
                / f"{Path(args.manifest).stem}_predictions.jsonl",
            )
        else:
            args.output = "results/meal_json/predictions.jsonl"
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
