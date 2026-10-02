import argparse
import json
from pathlib import Path

import torch
from utils.jsonl import read_jsonl, write_jsonl
from utils.metrics import evaluate_caption_records
from .infer import load_caption_model
from .datasets import image_transform


def evaluate(checkpoint: str, manifest: str, image_root: str, output: str, max_length: int = 64, beam_size: int = 3):
    from PIL import Image
    model, vocab = load_caption_model(checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    transform = image_transform(224)
    records = []
    for row in read_jsonl(manifest):
        image = transform(Image.open(Path(image_root) / row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
        ids = model.generate(image, vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"], max_length, beam_size)[0].tolist()
        records.append({"image_id": row["image_id"], "references": row["captions"], "prediction": vocab.decode(ids)})
    metrics = evaluate_caption_records(records)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, target.with_name("predictions.jsonl"))
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="results/captioning/metrics.json")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.manifest, args.image_root, args.output)
