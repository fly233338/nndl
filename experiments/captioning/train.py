import argparse
import json
import time
from functools import partial
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


def _run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    for images, tokens, _ in loader:
        images, tokens = images.to(device), tokens.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images, tokens[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(1, batches)


def train_model(config_path: str, model_name: str) -> Path:
    config = load_yaml(config_path)
    seed_everything(config.get("seed", 2026))
    data_dir = Path(config["data_dir"])
    image_root = config.get("image_root") or json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8")).get("image_root", "")
    vocab_path = data_dir / "vocab.json"
    vocab = Vocabulary.build(_captions(data_dir / "train.jsonl")) if not vocab_path.exists() else Vocabulary.load(vocab_path)
    vocab.save(vocab_path)
    train_set = CaptionDataset(data_dir / "train.jsonl", image_root, vocab, config.get("max_length", 64), True)
    val_set = CaptionDataset(data_dir / "val.jsonl", image_root, vocab, config.get("max_length", 64), False)
    collate = partial(collate_caption, pad_id=vocab.token_to_id["<pad>"])
    loader = torch.utils.data.DataLoader(train_set, batch_size=config.get("batch_size", 16), shuffle=True, num_workers=config.get("num_workers", 0), collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=config.get("num_workers", 0), collate_fn=collate)
    model = build_model(model_name, len(vocab.token_to_id), config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.get("learning_rate", 1e-4))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.token_to_id["<pad>"])
    output = Path(config.get("output_dir", "results/captioning")) / model_name
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    patience = config.get("patience", 3)
    min_delta = config.get("min_delta", 0.0)
    stale_epochs = 0
    history = []
    for epoch in range(config.get("epochs", 10)):
        started = time.perf_counter()
        train_loss = _run_epoch(model, loader, device, criterion, optimizer)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, device, criterion)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "seconds": time.perf_counter() - started}
        history.append(record)
        print(f"{model_name} epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} seconds={record['seconds']:.1f}")
        checkpoint = {"model": model.state_dict(), "model_name": model_name, "vocab": vocab.token_to_id, "config": config, "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        torch.save(checkpoint, output / "last.pt")
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            stale_epochs = 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"{model_name} early_stop epoch={epoch + 1} patience={patience}")
                break
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
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
