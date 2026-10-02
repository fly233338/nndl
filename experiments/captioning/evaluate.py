import argparse
import json
from pathlib import Path

import torch
from utils.jsonl import read_jsonl, write_jsonl
from utils.metrics import evaluate_caption_records, metric_metadata
from .datasets import image_transform
from .infer import load_caption_model


def evaluate(checkpoint: str, manifest: str, image_root: str, output: str, max_length: int | None = None, beam_size: int | None = None, decode: str = "beam", image_size: int | None = None):
    from PIL import Image
    model, vocab, state = load_caption_model(checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    config = state.get("config", {})
    max_length = int(max_length or config.get("max_length", 64))
    image_size = int(image_size or config.get("image_size", 224))
    beam_size = 1 if decode == "greedy" else int(beam_size or config.get("beam_size", 3))
    device = next(model.parameters()).device
    transform = image_transform(image_size)
    records = []
    for row in read_jsonl(manifest):
        image = transform(Image.open(Path(image_root) / row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
        ids = model.generate(image, vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"], max_length, beam_size)[0].tolist()
        records.append({"image_id": row["image_id"], "references": row["captions"], "prediction": vocab.decode(ids)})

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = target.parent / f"predictions_{decode}.jsonl"
    write_jsonl(records, prediction_path)
    metrics = evaluate_caption_records(records)
    metrics.update({"checkpoint": str(checkpoint), "decode": decode, "beam_size": beam_size, "image_size": image_size})
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
    args = parser.parse_args()
    evaluate(args.checkpoint, args.manifest, args.image_root, args.output, args.max_length, args.beam_size, args.decode, args.image_size)
