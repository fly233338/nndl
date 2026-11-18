"""加载图像描述模型，生成预测并计算统一文本指标。"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch

from utils.jsonl import write_jsonl
from utils.config import load_yaml
from utils.files import sha256_file
from utils.metrics import evaluate_caption_records, metric_metadata

from .datasets import CaptionDataset, collate_caption
from .infer import load_caption_model


@torch.no_grad()
def evaluate(
    checkpoint: str,
    manifest: str,
    image_root: str,
    output: str,
    max_length: int | None = None,
    beam_size: int | None = None,
    decode: str = "beam",
    image_size: int | None = None,
    batch_size: int | None = None,
    decode_config: str | None = None,
):
    model, vocab, state = load_caption_model(
        checkpoint,
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    config = state.get("config", {})
    max_length = int(max_length or config.get("max_length", 64))
    image_size = int(image_size or config.get("image_size", 224))
    beam_size = 1 if decode == "greedy" else int(beam_size or config.get("beam_size", 3))
    device = next(model.parameters()).device
    if decode_config:
        if not hasattr(model.decoder, "decoding"):
            raise ValueError("Decode settings are supported by Transformer decoders")
        model.decoder.decoding.update(load_yaml(decode_config).get("decoding", {}))
    dataset = CaptionDataset(manifest, image_root, vocab, max_length, False, image_size, config, True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size or config.get("eval_batch_size", config.get("batch_size", 16)),
        shuffle=False, num_workers=config.get("num_workers", 0),
        collate_fn=partial(collate_caption, pad_id=vocab.token_to_id["<pad>"]),
    )
    records = []
    started = time.perf_counter()
    for images, _, image_ids in loader:
        sequences = model.generate(
            images.to(device), vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"],
            max_length, beam_size,
        )
        for image_id, ids in zip(image_ids, sequences):
            records.append({
                "image_id": image_id, "references": dataset.references[image_id],
                "prediction": vocab.decode(ids.tolist()),
            })
    inference_seconds = time.perf_counter() - started

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = target.parent / f"predictions_{decode}.jsonl"
    write_jsonl(records, prediction_path)
    metrics = evaluate_caption_records(records)
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "decode": decode,
            "beam_size": beam_size,
            "image_size": image_size,
            "decoding": getattr(model.decoder, "decoding", {}),
            "preprocessing": config.get("preprocessing", "legacy_imagenet"),
            "inference_seconds": inference_seconds,
            "seconds_per_image": inference_seconds / max(1, len(records)),
            "manifest_sha256": sha256_file(manifest),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    try:
        metrics["metric_metadata"] = metric_metadata()
    except Exception:
        metrics["metric_metadata"] = {"tokenizer": "PTBTokenizer", "scale": "raw"}
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="results/captioning/metrics.json")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--decode-config")
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.manifest,
        args.image_root,
        args.output,
        args.max_length,
        args.beam_size,
        args.decode,
        args.image_size,
        args.batch_size,
        args.decode_config,
    )
