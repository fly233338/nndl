"""加载任务二自研模型，并按批次生成膳食 JSON。"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from utils.jsonl import read_jsonl, write_jsonl
from .schema import parse_and_validate
from .tokenizer import JSONTokenizer


def load_baseline(checkpoint, device="cpu"):
    """恢复 checkpoint 中的模型和 JSON tokenizer。"""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    tokenizer = JSONTokenizer(state["tokenizer"])
    from experiments.captioning.models import build_model

    model = build_model(state["model_name"], len(tokenizer.token_to_id), state["config"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, tokenizer, state


@torch.no_grad()
def infer_manifest(
    checkpoint: str,
    manifest: str,
    image_root: str,
    output: str,
    decode: str = "beam",
    beam_size: int = 3,
    batch_size: int = 16,
):
    """分批推理图片，保留清单顺序及每张图片的失败记录。"""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from experiments.captioning.datasets import image_transform

    model, tokenizer, state = load_baseline(
        checkpoint,
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    device = next(model.parameters()).device
    config = state.get("config", {})
    transform = image_transform(int(config.get("image_size", 224)))
    effective_beam = 1 if decode == "greedy" else beam_size

    rows = list(read_jsonl(manifest))
    records = []

    with tqdm(total=len(rows), desc="Baseline inference", unit="img") as progress:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            batch_records = []
            images, ready = [], []

            for row in batch:
                record = {
                    "image_id": row["image_id"],
                    "label": row.get("label"),
                    "raw_output": "",
                    "parsed": None,
                }
                batch_records.append(record)
                try:
                    with Image.open(Path(image_root) / row["image_path"]) as image:
                        images.append(transform(image.convert("RGB")))
                    ready.append(record)
                except Exception as exc:
                    record["parsed"] = {
                        "valid": False,
                        "errors": ["inference_error", str(exc)],
                        "parsed": None,
                    }

            if images:
                try:
                    generated = model.generate(
                        torch.stack(images).to(device),
                        tokenizer.token_to_id["<bos>"],
                        tokenizer.token_to_id["<eos>"],
                        int(config.get("max_length", 256)),
                        effective_beam,
                    )
                    for record, ids in zip(ready, generated):
                        record["raw_output"] = tokenizer.decode(ids.tolist())
                        record["parsed"] = parse_and_validate(record["raw_output"])
                except Exception as exc:
                    for record in ready:
                        if record["parsed"] is None:
                            record["parsed"] = {
                                "valid": False,
                                "errors": ["inference_error", str(exc)],
                                "parsed": None,
                            }

            records.extend(batch_records)
            progress.update(len(batch))

    write_jsonl(records, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="results/meal_json/baseline_predictions.jsonl")
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    infer_manifest(
        args.checkpoint,
        args.manifest,
        args.image_root,
        args.output,
        args.decode,
        args.beam_size,
        args.batch_size,
    )
