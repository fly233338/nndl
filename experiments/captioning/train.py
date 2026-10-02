import argparse
import json
import subprocess
import time
from functools import partial
from pathlib import Path

import torch
from utils.config import load_yaml
from utils.metrics import evaluate_caption_records
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


@torch.no_grad()
def _validation_cider(model, loader, device, vocab, max_length, beam_size):
    model.eval()
    records = []
    seen = set()
    for images, _, ids in loader:
        images = images.to(device)
        generated = model.generate(images, vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"], max_length, beam_size)
        for image_id, sequence in zip(ids, generated):
            if image_id in seen:
                continue
            seen.add(image_id)
            records.append({"image_id": image_id, "references": loader.dataset.references[image_id], "prediction": vocab.decode(sequence.tolist())})
    return evaluate_caption_records(records, metrics=("CIDEr-D",)).get("CIDEr-D", 0.0)


def _git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train_model(config_path: str, model_name: str) -> Path:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    data_dir = Path(config["data_dir"])
    info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8")) if (data_dir / "dataset_info.json").exists() else {}
    image_root = config.get("image_root") or info.get("image_root", "")
    image_size = int(config.get("image_size", 224))
    max_length = int(config.get("max_length", 64))
    vocab_path = data_dir / "vocab.json"
    vocab = Vocabulary.load(vocab_path) if vocab_path.exists() else Vocabulary.build(_captions(data_dir / "train.jsonl"))
    vocab.save(vocab_path)

    train_set = CaptionDataset(data_dir / "train.jsonl", image_root, vocab, max_length, True, image_size)
    val_set = CaptionDataset(data_dir / "val.jsonl", image_root, vocab, max_length, False, image_size)
    val_set.references = {row["image_id"]: row.get("captions", []) for row in val_set.rows}
    collate = partial(collate_caption, pad_id=vocab.token_to_id["<pad>"])
    loader = torch.utils.data.DataLoader(train_set, batch_size=config.get("batch_size", 16), shuffle=True, num_workers=config.get("num_workers", 0), collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=config.get("num_workers", 0), collate_fn=collate)

    model_config = {**config, "max_length": max_length}
    model = build_model(model_name, len(vocab.token_to_id), model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.get("learning_rate", 1e-4))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.token_to_id["<pad>"])

    output = Path(config.get("output_dir", "results/captioning")) / model_name
    output.mkdir(parents=True, exist_ok=True)
    resume = config.get("resume")
    start_epoch = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))

    best_cider = float("-inf")
    patience = int(config.get("patience", 4))
    stale_epochs = 0
    history = []
    started_total = time.perf_counter()
    decode_beam = 1 if config.get("validation_decode", "beam") == "greedy" else int(config.get("beam_size", 3))
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    for epoch in range(start_epoch, int(config.get("epochs", 10))):
        started = time.perf_counter()
        train_loss = _run_epoch(model, loader, device, criterion, optimizer)
        val_loss = _run_epoch(model, val_loader, device, criterion)
        val_cider = _validation_cider(model, val_loader, device, vocab, max_length, decode_beam)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_CIDEr-D": val_cider, "seconds": time.perf_counter() - started}
        history.append(record)
        print(f"{model_name} epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_CIDEr-D={val_cider:.4f} seconds={record['seconds']:.1f}")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_name": model_name,
            "vocab": vocab.token_to_id,
            "config": config,
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_CIDEr-D": val_cider,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "elapsed_seconds": time.perf_counter() - started_total,
            "git_revision": _git_revision(),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        }
        torch.save(checkpoint, output / "last.pt")
        if val_cider > best_cider:
            best_cider = val_cider
            stale_epochs = 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"{model_name} early_stop epoch={epoch + 1} patience={patience}")
                break
    (output / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
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
