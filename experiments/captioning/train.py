import argparse
import json
import time
from pathlib import Path

import torch
from utils.config import load_yaml
from utils.seed import seed_everything
from .datasets import CaptionDataset, collate_caption
from .models import build_model
from .tokenizer import Vocabulary


def _captions(manifest: Path):
    from utils.jsonl import read_jsonl
    return [caption for row in read_jsonl(manifest) for caption in row.get("captions", [])]


def train_model(config_path: str, model_name: str) -> Path:
    config = load_yaml(config_path)
    seed_everything(config.get("seed", 2026))
    data_dir = Path(config["data_dir"])
    image_root = config.get("image_root") or json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8")).get("image_root", "")
    vocab_path = data_dir / "vocab.json"
    vocab = Vocabulary.build(_captions(data_dir / "train.jsonl")) if not vocab_path.exists() else Vocabulary.load(vocab_path)
    vocab.save(vocab_path)
    train_set = CaptionDataset(data_dir / "train.jsonl", image_root, vocab, config.get("max_length", 64), True)
    loader = torch.utils.data.DataLoader(train_set, batch_size=config.get("batch_size", 16), shuffle=True, num_workers=config.get("num_workers", 0), collate_fn=lambda batch: collate_caption(batch, vocab.token_to_id["<pad>"]))
    model = build_model(model_name, len(vocab.token_to_id), config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.get("learning_rate", 1e-4))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.token_to_id["<pad>"])
    output = Path(config.get("output_dir", "results/captioning")) / model_name
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(config.get("epochs", 10)):
        model.train()
        total, batches = 0.0, 0
        started = time.perf_counter()
        for images, tokens, _ in loader:
            images, tokens = images.to(device), tokens.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, tokens[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        average = total / max(1, batches)
        print(f"{model_name} epoch={epoch + 1} loss={average:.4f} seconds={time.perf_counter() - started:.1f}")
        if average < best_loss:
            best_loss = average
            torch.save({"model": model.state_dict(), "model_name": model_name, "vocab": vocab.token_to_id, "config": config}, output / "best.pt")
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/captioning.yaml")
    parser.add_argument("--model", choices=["cnn_gru", "attention_rnn", "transformer", "graph_transformer", "vit_transformer"])
    args = parser.parse_args()
    config = load_yaml(args.config)
    models = [args.model] if args.model else config.get("models", [])
    for model in models:
        train_model(args.config, model)


if __name__ == "__main__":
    main()
