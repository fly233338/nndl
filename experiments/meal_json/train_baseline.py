import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from utils.config import load_yaml
from utils.jsonl import read_jsonl
from .schema import canonical_json


def train(config_path: str):
    config = load_yaml(config_path)
    data_dir = Path(config["data_dir"])
    rows = list(read_jsonl(data_dir / "train.jsonl"))
    if not rows:
        raise ValueError("No meal training rows found. Run prepare_meal_json.py first.")
    from experiments.captioning.tokenizer import Vocabulary
    vocab = Vocabulary.build([canonical_json(row["label"]) for row in rows])
    from experiments.captioning.models import build_model
    from experiments.captioning.datasets import image_transform
    image_root = config.get("image_root", "")
    samples = [(row, canonical_json(row["label"])) for row in rows]
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, index):
            row, target = samples[index]
            image = Image.open(Path(image_root) / row["image_path"]).convert("RGB")
            return image_transform(224, True)(image), torch.tensor(vocab.encode(target, config.get("max_length", 256))), row["image_id"]
    def collate(batch):
        images, sequences, ids = zip(*batch)
        width = max(x.numel() for x in sequences)
        padded = torch.full((len(sequences), width), vocab.token_to_id["<pad>"], dtype=torch.long)
        for i, sequence in enumerate(sequences): padded[i, :sequence.numel()] = sequence
        return torch.stack(images), padded, ids
    loader = torch.utils.data.DataLoader(Dataset(), batch_size=1, shuffle=True, collate_fn=collate)
    model_name = "cnn_gru"
    model = build_model(model_name, len(vocab.token_to_id), {**config, "max_length": config.get("max_length", 256)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.token_to_id["<pad>"])
    output = Path(config.get("output_dir", "results/meal_json")) / "baseline"
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.get("epochs", 3)):
        model.train()
        for images, tokens, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device), tokens[:, :-1].to(device))
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1).to(device))
            loss.backward(); optimizer.step()
        print(f"baseline epoch={epoch + 1} loss={float(loss):.4f}")
    torch.save({"model": model.state_dict(), "model_name": model_name, "vocab": vocab.token_to_id, "config": config}, output / "best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/meal_json.yaml")
    train(parser.parse_args().config)
