import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from utils.config import load_yaml
from utils.jsonl import read_jsonl
from utils.seed import seed_everything
from .schema import canonical_json
from .tokenizer import JSONTokenizer


class MealDataset(torch.utils.data.Dataset):
    def __init__(self, rows, image_root, tokenizer, max_length, image_size, train):
        from experiments.captioning.datasets import image_transform
        self.rows = rows
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = image_transform(image_size, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.image_root / row["image_path"]).convert("RGB")
        text = canonical_json(row["label"])
        return self.transform(image), torch.tensor(self.tokenizer.encode(text, self.max_length), dtype=torch.long), row["image_id"]


def _collate(batch, pad_id):
    images, sequences, ids = zip(*batch)
    width = max(sequence.numel() for sequence in sequences)
    padded = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        padded[index, :sequence.numel()] = sequence
    return torch.stack(images), padded, ids


def _epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for images, tokens, _ in loader:
        images, tokens = images.to(device), tokens.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images, tokens[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
        if training:
            loss.backward()
            optimizer.step()
        total += float(loss.detach())
        count += 1
    return total / max(1, count)


def train(config_path: str, model_name: str | None = None):
    config = load_yaml(config_path)
    seed_everything(config.get("seed", 2026))
    data_dir = Path(config["data_dir"])
    train_rows = list(read_jsonl(data_dir / "train.jsonl"))
    val_rows = list(read_jsonl(data_dir / "val.jsonl"))
    if not train_rows:
        raise ValueError("No meal training rows found. Run prepare_meal_json.py first.")
    if model_name is None:
        model_name = config.get("caption_model", "vit_transformer")
    caption_checkpoint = config.get("caption_model_checkpoint")
    if caption_checkpoint and not Path(caption_checkpoint).exists():
        raise FileNotFoundError(f"Caption model checkpoint not found: {caption_checkpoint}")
    texts = [canonical_json(row["label"]) for row in train_rows]
    tokenizer = JSONTokenizer.build(texts)
    image_root = config.get("image_root", "") or json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8")).get("image_root", "")
    image_size = int(config.get("image_size", 224))
    max_length = int(config.get("max_length", 256))
    train_set = MealDataset(train_rows, image_root, tokenizer, max_length, image_size, True)
    val_set = MealDataset(val_rows, image_root, tokenizer, max_length, image_size, False)
    collate = lambda batch: _collate(batch, tokenizer.token_to_id["<pad>"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.get("batch_size", 4), shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.get("batch_size", 4), shuffle=False, collate_fn=collate)
    from experiments.captioning.models import build_model
    model = build_model(model_name, len(tokenizer.token_to_id), {**config, "max_length": max_length})
    if caption_checkpoint:
        caption_state = torch.load(caption_checkpoint, map_location="cpu", weights_only=False)
        if caption_state.get("model_name") != model_name:
            raise ValueError(f"Checkpoint model {caption_state.get('model_name')} does not match {model_name}")
        current_state = model.state_dict()
        transferred = {
            key: value
            for key, value in caption_state["model"].items()
            if key in current_state
            and current_state[key].shape == value.shape
            and not key.startswith("decoder.embedding.")
            and not key.startswith("decoder.output.")
        }
        model.load_state_dict(transferred, strict=False)
        print(f"loaded {len(transferred)} compatible parameters from {caption_checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.get("learning_rate", 1e-4))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])
    output = Path(config.get("output_dir", "results/meal_json")) / f"baseline_{model_name}"
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(int(config.get("epochs", 3))):
        started = time.perf_counter()
        train_loss = _epoch(model, train_loader, device, criterion, optimizer)
        val_loss = _epoch(model, val_loader, device, criterion)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "seconds": time.perf_counter() - started}
        history.append(record)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "model_name": model_name, "tokenizer": tokenizer.token_to_id, "config": config, "epoch": epoch + 1, "val_loss": val_loss}
        torch.save(checkpoint, output / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, output / "best.pt")
        print(f"baseline={model_name} epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    (output / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer.save(output / "tokenizer.json")
    return output / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/meal_json.yaml")
    parser.add_argument("--model", choices=["cnn_gru", "attention_rnn", "transformer", "graph_transformer", "vit_transformer"])
    args = parser.parse_args()
    train(args.config, args.model)
