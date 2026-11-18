"""从任务一 CE 检查点开始，以 CIDEr-D 奖励进行 SCST 微调。"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from utils.config import load_yaml
from utils.files import sha256_file
from utils.metrics import evaluate_caption_records
from utils.seed import seed_everything

from .data import disjoint, load_model, make_loader, manifest, resolve_device
from .evaluate import caption_records
from .policy import policy_loss, rollout
from .reward import CiderDReward


def run_epoch(model, vocab, loader, reward, device, max_length, optimizer, grad_clip):
    # 关闭 dropout；采样不求导，随后重算采样序列概率以更新解码器。
    """每批重新采样，以采样和贪心奖励之差执行策略梯度更新。"""
    model.eval()
    totals = dict.fromkeys(
        ("loss", "sampled_reward", "greedy_reward", "advantage", "positive_advantage_rate"),
        0.0,
    )
    samples = 0
    bos_id, eos_id = vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"]
    for images, ids in tqdm(loader, desc="SCST"):
        images = images.to(device)
        with torch.no_grad():
            features = model.encoder(images)
            memory = model.decoder._encode(features)
            greedy = rollout(model.decoder, memory, bos_id, eos_id, max_length, sample=False)
            sampled = rollout(model.decoder, memory, bos_id, eos_id, max_length, sample=True)
        sampled_text = [vocab.decode(sequence.tolist()) for sequence in sampled]
        greedy_text = [vocab.decode(sequence.tolist()) for sequence in greedy]
        sampled_reward, greedy_reward = reward.rewards(ids, sampled_text, greedy_text)
        advantage = torch.as_tensor(sampled_reward - greedy_reward, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = policy_loss(model.decoder, features, sampled, advantage, eos_id)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite SCST loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.decoder.parameters(),
            grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        count = len(ids)
        totals["loss"] += float(loss.detach()) * count
        totals["sampled_reward"] += float(sampled_reward.sum())
        totals["greedy_reward"] += float(greedy_reward.sum())
        totals["advantage"] += float(advantage.sum())
        totals["positive_advantage_rate"] += int((advantage > 0).sum())
        samples += count
    return {key: value / samples for key, value in totals.items()}


def train(config_path):
    """从 CE 权重微调，按验证集 CIDEr-D 选择轮次并保存训练来源。"""
    config = load_yaml(config_path)
    seed_everything(config["seed"])
    if config["epochs"] < 1 or config["patience"] < 1:
        raise ValueError("epochs and patience must be positive")
    output = Path(config["output_dir"])
    source = Path(config["checkpoint"])
    if output.resolve() == source.parent.resolve() or source.resolve() in {
        (output / name).resolve() for name in ("best.pt", "last.pt")
    }:
        raise ValueError("SCST outputs must not overwrite the CE checkpoint")

    if any((output / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError("SCST run already exists; choose a new output_dir for a new run")
    train_rows, val_rows = manifest(config, "train"), manifest(config, "val")
    disjoint(train_rows, val_rows)
    reward = CiderDReward(config)
    device = resolve_device(config)
    model, vocab, initial, model_config = load_model(source, config, device)
    if initial.get("training_method") == "scst_cider_d":
        raise ValueError("Initialize this experiment from the original CE checkpoint")
    image_size = model_config.get("image_size", 224)
    max_length = model_config.get("max_length", 64)
    if not 2 <= max_length <= model.decoder.position.num_embeddings:
        raise ValueError("Invalid generation length in checkpoint")
    loader = make_loader(train_rows, config, image_size, shuffle=True, model_config=model_config)
    val_loader = make_loader(val_rows, config, image_size, model_config=model_config)
    if config["validation_decode"] not in ("greedy", "beam"):
        raise ValueError("validation_decode must be greedy or beam")
    beam_size = 1 if config["validation_decode"] == "greedy" else config["beam_size"]
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "ce_checkpoint_sha256": sha256_file(source),
        "train_manifest_sha256": sha256_file(Path(config["data_dir"]) / "train.jsonl"),
        "val_manifest_sha256": sha256_file(Path(config["data_dir"]) / "val.jsonl"),
        "reward_cache_sha256": sha256_file(config["reward_cache"]),
    }
    ce_records = caption_records(model, vocab, val_loader, val_rows, device, max_length, beam_size)
    ce_val = evaluate_caption_records(ce_records, metrics=("CIDEr-D",))["CIDEr-D"]
    (output / "ce_reference.json").write_text(
        json.dumps(
            {
                "val_CIDEr-D": ce_val,
                "validation_decode": config["validation_decode"],
                "beam_size": beam_size,
                **provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history, best, stale = [], -math.inf, 0
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        metrics = run_epoch(
            model,
            vocab,
            loader,
            reward,
            device,
            max_length,
            optimizer,
            config["grad_clip"],
        )
        records = caption_records(model, vocab, val_loader, val_rows, device, max_length, beam_size)
        val_cider = evaluate_caption_records(records, metrics=("CIDEr-D",))["CIDEr-D"]
        if not math.isfinite(val_cider):
            raise ValueError("Non-finite validation CIDEr-D")
        history.append(
            {
                "epoch": epoch,
                "train": metrics,
                "val_CIDEr-D": val_cider,
                "delta_vs_ce_val": val_cider - ce_val,
                "seconds": time.perf_counter() - started,
            },
        )
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_name": initial["model_name"],
            "vocab": vocab.token_to_id,
            "config": model_config,
            "scst_config": config,
            "seed": config["seed"],
            "epoch": epoch,
            "val_CIDEr-D": val_cider,
            "training_method": "scst_cider_d",
            **provenance,
        }
        torch.save(state, output / "last.pt")
        if val_cider > best:
            best, stale = val_cider, 0
            torch.save(state, output / "best.pt")
        else:
            stale += 1
        (output / "training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(history[-1]), flush=True)
        if stale >= config["patience"]:
            break
    print(f"SCST best val CIDEr-D={best:.6f}; CE val={ce_val:.6f}; delta={best - ce_val:+.6f}")
    return output / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/caption_rl.yaml")
    args = parser.parse_args()
    train(args.config)
