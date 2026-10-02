import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from utils.jsonl import read_jsonl, write_jsonl
from .schema import canonical_json, parse_and_validate
from .tokenizer import JSONTokenizer


def load_baseline(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    tokenizer = JSONTokenizer(state["tokenizer"])
    from experiments.captioning.models import build_model
    model = build_model(state["model_name"], len(tokenizer.token_to_id), state["config"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, tokenizer, state


def infer_manifest(checkpoint: str, manifest: str, image_root: str, output: str, decode: str = "beam", beam_size: int = 3):
    from experiments.captioning.datasets import image_transform
    model, tokenizer, state = load_baseline(checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    config = state.get("config", {})
    transform = image_transform(int(config.get("image_size", 224)))
    effective_beam = 1 if decode == "greedy" else beam_size
    records = []
    for row in read_jsonl(manifest):
        try:
            image = transform(Image.open(Path(image_root) / row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
            ids = model.generate(image, tokenizer.token_to_id["<bos>"], tokenizer.token_to_id["<eos>"], int(config.get("max_length", 256)), effective_beam)[0].tolist()
            raw = tokenizer.decode(ids)
            parsed = parse_and_validate(raw)
        except Exception as exc:
            raw = ""
            parsed = {"valid": False, "errors": ["inference_error", str(exc)], "parsed": None}
        records.append({"image_id": row["image_id"], "label": row.get("label"), "raw_output": raw, "parsed": parsed})
    write_jsonl(records, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="results/meal_json/baseline_predictions.jsonl")
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-size", type=int, default=3)
    args = parser.parse_args()
    infer_manifest(args.checkpoint, args.manifest, args.image_root, args.output, args.decode, args.beam_size)
