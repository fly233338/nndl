"""训练描述模型，支持分阶段微调并按验证集 CIDEr-D 选择检查点。"""

import argparse
import json
import math
import subprocess
import time
from functools import partial
from pathlib import Path

import torch

from utils.config import load_yaml, _merge
from utils.files import sha256_file
from utils.jsonl import read_jsonl
from utils.metrics import evaluate_caption_records
from utils.seed import seed_everything

from .datasets import CaptionDataset, collate_caption, resolve_preprocessing
from .models import build_model
from .models.checkpoint import align_vit_state_dict
from .tokenizer import Vocabulary


def _run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None, config=None):
    config = config or {}
    training = optimizer is not None
    model.train(training)
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    amp = device.type == "cuda" and config.get("mixed_precision", "none") != "none"
    dtype = torch.bfloat16 if config.get("mixed_precision") == "bf16" else torch.float16
    total_loss = total_tokens = group_tokens = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, tokens, _) in enumerate(loader):
        images, tokens = images.to(device, non_blocking=True), tokens.to(device, non_blocking=True)
        valid_tokens = int(tokens[:, 1:].ne(criterion.ignore_index).sum())
        with torch.set_grad_enabled(training), torch.autocast(device.type, dtype=dtype, enabled=amp):
            logits = model(images, tokens[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
        total_loss += float(loss.detach())
        total_tokens += valid_tokens
        if training:
            scaler.scale(loss).backward()
            group_tokens += valid_tokens
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                # 按实际有效词元数归一化，处理最后一个不足累积步数的批次。
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_tokens)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 5.0))
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def _validation_cider(model, loader, device, vocab, config):
    model.eval()
    records = []
    beam_size = 1 if config.get("validation_decode", "beam") == "greedy" else config.get("beam_size", 3)
    for images, _, ids in loader:
        generated = model.generate(
            images.to(device), vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"],
            config.get("max_length", 64), beam_size,
        )
        for image_id, sequence in zip(ids, generated):
            records.append({
                "image_id": image_id, "references": loader.dataset.references[image_id],
                "prediction": vocab.decode(sequence.tolist()),
            })
    return evaluate_caption_records(records, metrics=("CIDEr-D",))["CIDEr-D"]


def _optimizer(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        encoder = name.startswith("encoder.")
        decay = parameter.ndim > 1 and ".embedding." not in name and ".position." not in name
        key = encoder, decay
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": config.get("encoder_learning_rate", config["learning_rate"] / 10) if encoder else config["learning_rate"],
                "weight_decay": config.get("weight_decay", 0.01) if decay else 0.0,
            }
        # 保留尚未解冻的参数，解冻后无需重建优化器或丢失动量。
        groups[key]["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()))


def _scheduler(optimizer, config, updates_per_epoch):
    total = updates_per_epoch * int(config["epochs"])
    warmup = int(total * config.get("warmup_ratio", 0))

    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if config.get("scheduler", "constant") == "cosine":
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            floor = config.get("min_lr_ratio", 0.1)
            return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train_model(config_path: str, model_name: str, seed=None, run_dir=None) -> Path:
    config = load_yaml(config_path)
    config = _merge(config, config.get("experiments", {}).get(model_name, {}))
    if seed is not None:
        config["seed"] = seed
    config["preprocessing"] = resolve_preprocessing(config, model_name)
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    data_dir = Path(config["data_dir"])
    info_path = data_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    image_root = config.get("image_root") or info.get("image_root", "")
    image_size, max_length = int(config.get("image_size", 224)), int(config.get("max_length", 64))
    vocab_path = data_dir / "vocab.json"
    vocab = Vocabulary.load(vocab_path) if vocab_path.exists() else Vocabulary.build(
        [caption for row in read_jsonl(data_dir / "train.jsonl") for caption in row["captions"]],
    )
    vocab.save(vocab_path)
    datasets = [
        CaptionDataset(data_dir / f"{split}.jsonl", image_root, vocab, max_length, training, image_size, config, unique)
        for split, training, unique in (("train", True, False), ("val", False, False), ("val", False, True))
    ]
    if set(datasets[0].references).intersection(datasets[1].references):
        raise ValueError("Training and validation image IDs overlap")
    device_name = config.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type == "cuda" and config.get("mixed_precision") == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; select fp16 or none")
    workers = int(config.get("num_workers", 0))
    loaders = [
        torch.utils.data.DataLoader(
            dataset,
            batch_size=config.get("batch_size", 16) if index == 0 else config.get("eval_batch_size", config.get("batch_size", 16)),
            shuffle=index == 0,
            num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
            collate_fn=partial(collate_caption, pad_id=vocab.token_to_id["<pad>"]),
        )
        for index, dataset in enumerate(datasets)
    ]
    train_loader, val_loader, decode_loader = loaders
    model = build_model(model_name, len(vocab.token_to_id), config).to(device)
    partial_tuning = "encoder_trainable_blocks" in config
    if partial_tuning and model_name != "vit_transformer":
        raise ValueError("Partial block fine-tuning is configured for vit_transformer only")
    if partial_tuning:
        model.encoder.set_trainable_blocks(0)
    output = Path(run_dir) if run_dir else Path(config.get("output_dir", "results/captioning")) / model_name
    resume, initialize = config.get("resume"), config.get("init_checkpoint")
    if resume and initialize:
        raise ValueError("Choose resume or init_checkpoint, not both")
    if not resume and (output / "best.pt").exists():
        raise FileExistsError(f"Choose a new output_dir to preserve the existing run: {output}")
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config, math.ceil(len(train_loader) / config.get("gradient_accumulation_steps", 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.get("mixed_precision") == "fp16")
    best_cider, stale_epochs, start_epoch, history = -math.inf, 0, 0, []
    if resume or initialize:
        state = torch.load(resume or initialize, map_location="cpu", weights_only=False)
        if state["vocab"] != vocab.token_to_id or state["model_name"] != model_name:
            raise ValueError("Checkpoint model and training vocabulary must match")
        if bool(state["config"].get("tie_embeddings", False)) != bool(config.get("tie_embeddings", False)):
            raise ValueError("Initialization requires the same embedding weight tying setting")
        weights, _ = align_vit_state_dict(state["model"], model.state_dict())
        if initialize and config.get("global_fusion") == "gated":
            missing, unexpected = model.load_state_dict(weights, strict=False)
            allowed = ("decoder.global_projection.", "decoder.global_gate.")
            if unexpected or any(not name.startswith(allowed) for name in missing):
                raise ValueError(f"Unexpected initialization mismatch: {missing}, {unexpected}")
        else:
            model.load_state_dict(weights, strict=True)
        if resume:
            if "scheduler" not in state:
                raise ValueError("Use init_checkpoint for legacy checkpoints; resume needs optimizer/scheduler state")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, best_cider = int(state["epoch"]), state["best_val_CIDEr-D"]
            stale_epochs, history = state["stale_epochs"], state["history"]
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=vocab.token_to_id["<pad>"], label_smoothing=config.get("label_smoothing", 0), reduction="sum",
    )
    provenance = {f"{split}_manifest_sha256": sha256_file(data_dir / f"{split}.jsonl") for split in ("train", "val")}
    if resume and any(state.get(key) != value for key, value in provenance.items()):
        raise ValueError("Training/validation manifests changed since the resumed checkpoint")
    if initialize:
        provenance["init_checkpoint_sha256"] = sha256_file(initialize)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    started_total = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"])):
        if partial_tuning:
            blocks = int(config["encoder_trainable_blocks"]) if epoch >= config.get("encoder_unfreeze_epoch", 0) else 0
            model.encoder.set_trainable_blocks(blocks)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, device, criterion, optimizer, scheduler, scaler, config)
        val_loss = _run_epoch(model, val_loader, device, criterion, config=config)
        val_cider = _validation_cider(model, decode_loader, device, vocab, config)
        if not math.isfinite(val_cider):
            raise ValueError("Non-finite validation CIDEr-D")
        improved = val_cider > best_cider + config.get("min_delta", 0)
        if improved:
            best_cider, stale_epochs = val_cider, 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            "val_CIDEr-D": val_cider, "seconds": time.perf_counter() - started,
            "learning_rates": scheduler.get_last_lr(),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_trainable_blocks": model.encoder._trainable_blocks if partial_tuning else None,
        }
        history.append(record)
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "model_name": model_name, "vocab": vocab.token_to_id, "config": config, "seed": seed,
            **record, "best_val_CIDEr-D": best_cider, "stale_epochs": stale_epochs, "history": history,
            "total_params": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": time.perf_counter() - started_total, "git_revision": _git_revision(),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            **provenance,
        }
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if stale_epochs >= config.get("patience", 4):
            break
    (output / "completed.json").write_text(json.dumps({"best_checkpoint": "best.pt", "best_val_CIDEr-D": best_cider}), encoding="utf-8")
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/task1_captioning/baseline.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--model", choices=["cnn_gru", "attention_rnn", "transformer", "graph_transformer", "vit_transformer"])
    args = parser.parse_args()
    config = load_yaml(args.config)
    names = [args.model] if args.model else config.get("models", list(config.get("experiments", {})))
    if args.run_dir and len(names) != 1:
        parser.error("--run-dir requires one --model")
    for name in names:
        train_model(args.config, name, args.seed, args.run_dir)


if __name__ == "__main__":
    main()
