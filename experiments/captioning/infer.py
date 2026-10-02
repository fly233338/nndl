import argparse
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
    return model, vocab


def predict(checkpoint: str, image_path: str, image_size: int = 224, beam_size: int = 3) -> str:
    from PIL import Image
    from .datasets import image_transform
    model, vocab = load_caption_model(checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    image = image_transform(image_size)(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    ids = model.generate(image, vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"], 64, beam_size)[0].tolist()
    return vocab.decode(ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(predict(args.checkpoint, args.image))
