import argparse
import json
from pathlib import Path

import torch
from .models import build_model
from .tokenizer import Vocabulary


def load_caption_model(checkpoint: str, device: str = "cpu"):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    vocab = Vocabulary(state["vocab"])
    model = build_model(state["model_name"], len(vocab.token_to_id), state["config"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, vocab, state


def predict(checkpoint: str, image_path: str, image_size: int | None = None, beam_size: int | None = None, decode: str = "beam") -> str:
    from PIL import Image
    from .datasets import image_transform
    model, vocab, state = load_caption_model(checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    config = state.get("config", {})
    image_size = int(image_size or config.get("image_size", 224))
    max_length = int(config.get("max_length", 64))
    beam_size = 1 if decode == "greedy" else int(beam_size or config.get("beam_size", 3))
    device = next(model.parameters()).device
    image = image_transform(image_size)(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    ids = model.generate(image, vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"], max_length, beam_size)[0].tolist()
    return vocab.decode(ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    args = parser.parse_args()
    print(predict(args.checkpoint, args.image, args.image_size, args.beam_size, args.decode))
