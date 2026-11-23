"""独立膳食实验：Qwen 辅助监督和 SigLIP 迁移，按生成 micro-F1 选型。"""

import argparse
import copy
import json
import math
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.config import load_yaml, _merge
from utils.files import sha256_file
from utils.jsonl import read_jsonl, write_jsonl
from utils.seed import seed_everything
from .evaluate import structured_metrics, _item_names
from .prompts import direct_prompt
from .schema import NUMERIC_FIELDS, canonical_json, parse_and_validate
from .tokenizer import JSONTokenizer


DEFAULT_CONFIG = "config/task2_meal_json/ablation.yaml"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def experiment_config(path, name):
    config = load_yaml(path)
    name = config.pop("aliases", {}).get(name, name)
    presets = config.pop("experiments")
    if name not in presets:
        raise ValueError(f"Unknown experiment {name}; choose from {list(presets)}")
    config = _merge(config, presets[name])
    config["experiment"] = name
    return config


def augment(image, settings):
    if not settings["enabled"]:
        return image
    if random.random() < settings["horizontal_flip"]:
        image = ImageOps.mirror(image)
    if random.random() < settings["rotation_probability"]:
        angle = random.uniform(-settings["rotation_degrees"], settings["rotation_degrees"])
        image = image.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=tuple(settings["fill"]))
    if random.random() < settings["jitter_probability"]:
        for enhancer, key in ((ImageEnhance.Brightness, "brightness"), (ImageEnhance.Contrast, "contrast")):
            image = enhancer(image).enhance(random.uniform(1 - settings[key], 1 + settings[key]))
    return image


class MealData(Dataset):
    def __init__(self, rows, config, training=False):
        self.rows, self.config, self.training = rows, config, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(Path(self.config["data"]["image_root"]) / row["image_path"]) as source:
            image = source.convert("RGB")
        if self.training:
            image = augment(image, self.config["augmentation"])
        return image, row


def raw_collate(batch):
    return list(zip(*batch))


def training_statistics(rows):
    names = sorted(set().union(*(_item_names(row["label"]) for row in rows)))
    values = np.asarray([[row["label"][key] for key in NUMERIC_FIELDS] for row in rows], dtype=np.float64)
    return {"item_names": names, "numeric_fields": list(NUMERIC_FIELDS),
            "numeric_scale": np.maximum(values.std(axis=0), 1.0).tolist()}


def auxiliary_head(dim, hidden, outputs):
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class QwenMeal(nn.Module):
    def __init__(self, config, stats, device, processor_path=None):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, TaskType, get_peft_model

        self.config, self.stats, self.device = config, stats, device
        settings = config["qwen"]
        source = settings["model_path"] or settings["model_id"]
        options = {"revision": settings["revision"], "local_files_only": settings["local_files_only"]}
        self.processor = AutoProcessor.from_pretrained(processor_path or source, **options)
        # Qwen processor 的 size 上下界表示像素面积。
        self.processor.image_processor.size = {
            "shortest_edge": settings["min_pixels"], "longest_edge": settings["max_pixels"],
        }
        self.base = AutoModelForImageTextToText.from_pretrained(
            source, **options, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        revision = getattr(self.base.config, "_commit_hash", None)
        if revision:
            settings["revision"] = revision
        targets = [name for name, module in self.base.named_modules()
                   if isinstance(module, nn.Linear) and "language_model.layers." in name
                   and name.rsplit(".", 1)[-1] in settings["target_modules"]]
        if not targets:
            raise ValueError("No language-only LoRA targets found; inspect model module names")
        self.lora_targets = targets
        self.base = get_peft_model(self.base, LoraConfig(
            r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
            target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM,
        ))
        if any(parameter.requires_grad for name, parameter in self.base.named_parameters() if ".visual." in name):
            raise ValueError("Visual encoder and merger must remain frozen")
        if settings["gradient_checkpointing"]:
            self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.base.enable_input_require_grads()
        self.base.config.use_cache = False
        dim = self.base.config.text_config.hidden_size
        aux = config["auxiliary"]
        self.item_head = auxiliary_head(dim, aux["hidden_dim"], len(stats["item_names"])) if aux["items"] else None
        self.numeric_head = auxiliary_head(dim, aux["hidden_dim"], 5) if aux["numeric"] else None
        self.register_buffer("numeric_scale", torch.tensor(stats["numeric_scale"], dtype=torch.float32))
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        # 视觉主干与 merger 始终冻结且保持 eval。
        for name, module in self.base.named_modules():
            if name.endswith(".visual"):
                module.eval()
        return self

    def messages(self, answer=None):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": direct_prompt()}]}]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def encode(self, images, rows=None):
        training = rows is not None
        self.processor.tokenizer.padding_side = "right" if training else "left"
        prompt = self.processor.apply_chat_template(self.messages(), tokenize=False, add_generation_prompt=True)
        prompts = self.processor(text=[prompt] * len(images), images=list(images), padding=True,
                                 return_tensors="pt", truncation=False)
        if training:
            texts = [self.processor.apply_chat_template(self.messages(canonical_json(row["label"])),
                     tokenize=False, add_generation_prompt=False) for row in rows]
            inputs = self.processor(text=texts, images=list(images), padding=True,
                                    return_tensors="pt", truncation=False)
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100
            lengths = prompts["attention_mask"].sum(1)
            for i, length in enumerate(lengths.tolist()):
                if not torch.equal(inputs["input_ids"][i, :length], prompts["input_ids"][i, :length]):
                    raise ValueError("Chat template answer boundary is not a matching prompt prefix")
                if int(inputs["attention_mask"][i].sum()) <= length:
                    raise ValueError("No supervised answer tokens")
                labels[i, :length] = -100
            positions = lengths - 1
        else:
            inputs, labels, positions = prompts, None, None
        if inputs["input_ids"].shape[1] > self.config["data"]["max_length"]:
            raise ValueError("Sequence exceeds data.max_length; increase it or reduce qwen.max_pixels. No truncation performed.")
        inputs = {key: value.to(self.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        return inputs, labels.to(self.device) if training else None, positions.to(self.device) if training else None

    def batch_loss(self, images, rows):
        inputs, labels, positions = self.encode(images, rows)
        auxiliary = self.item_head is not None or self.numeric_head is not None
        output = self.base(**inputs, use_cache=False, output_hidden_states=auxiliary, return_dict=True)
        target = labels[:, 1:]
        token_loss = F.cross_entropy(output.logits[:, :-1].float().transpose(1, 2), target,
                                     ignore_index=-100, reduction="none")
        ce = (token_loss.sum(1) / target.ne(-100).sum(1).clamp_min(1)).mean()
        loss, logged = ce, {"generation_loss": float(ce.detach())}
        if auxiliary:
            hidden = output.hidden_states[-1][torch.arange(len(rows), device=self.device), positions]
            if self.item_head is not None:
                label_names = [_item_names(row["label"]) for row in rows]
                targets = torch.tensor([[name in names for name in self.stats["item_names"]] for names in label_names],
                                       dtype=torch.float32, device=self.device)
                logits = self.item_head(hidden.float())
                item_loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
                loss = loss + self.config["auxiliary"]["item_weight"] * item_loss
                proposed = logits.detach().sigmoid() >= 0.5
                actual = targets.bool()
                tp = (proposed & actual).sum().item()
                logged.update(item_loss=float(item_loss.detach()), aux_item_tp=tp,
                              aux_item_fp=(proposed & ~actual).sum().item(),
                              aux_item_fn=(~proposed & actual).sum().item())
            if self.numeric_head is not None:
                predictions = F.softplus(self.numeric_head(hidden.float()).float())
                targets = torch.tensor([[row["label"][field] for field in NUMERIC_FIELDS] for row in rows],
                                       dtype=torch.float32, device=self.device)
                numeric_loss = ((predictions - targets) / self.numeric_scale).square().mean()
                loss = loss + self.config["auxiliary"]["numeric_weight"] * numeric_loss
                logged["numeric_loss"] = float(numeric_loss.detach())
                for i, field in enumerate(NUMERIC_FIELDS):
                    logged[f"aux_mse_{field}"] = float((predictions[:, i] - targets[:, i]).square().mean().detach())
        return loss, logged

    @torch.no_grad()
    def generate_text(self, images):
        inputs, _, _ = self.encode(images)
        sequences = self.base.generate(**inputs, do_sample=False, num_beams=self.config["decoding"]["beam_size"],
                                       max_new_tokens=self.config["decoding"]["max_new_tokens"], use_cache=True)
        return self.processor.batch_decode(sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class SiglipMeal(nn.Module):
    def __init__(self, config, stats, device, transfer=None):
        super().__init__()
        from experiments.task1_captioning.strong_captioner import StrongCaptioner, ImageTransform

        self.config, self.device = config, device
        self.tokenizer = JSONTokenizer(stats["tokenizer"])
        architecture = copy.deepcopy(config["caption_architecture"])
        architecture["encoder"]["model_path"] = config["siglip"]["model_path"]
        architecture["data"]["max_length"] = config["data"]["max_length"]
        architecture["training"] = config["training"]
        architecture["model"]["gradient_checkpointing"] = config["siglip"]["gradient_checkpointing"]
        architecture["decoding"] = {"beam_size": config["decoding"]["beam_size"], "length_penalty": 0.0}
        architecture["augmentation"]["enabled"] = False
        architecture["lora"]["enabled"] = False
        self.captioner = StrongCaptioner(architecture, self.tokenizer, device)
        config["caption_architecture"]["encoder"] = copy.deepcopy(architecture["encoder"])
        self.transform = ImageTransform(architecture)
        if transfer is not None:
            source = transfer["decoder"]
            weights = {k: v for k, v in source.items() if k not in ("embedding.weight", "output.weight", "position.weight")}
            missing, unexpected = self.captioner.decoder.load_state_dict(weights, strict=False)
            if set(missing) != {"embedding.weight", "output.weight", "position.weight"} or unexpected:
                raise ValueError("Unexpected caption decoder migration keys")
            with torch.no_grad():
                count = min(source["position.weight"].shape[0], config["data"]["max_length"])
                self.captioner.decoder.position.weight[:count].copy_(source["position.weight"][:count])

    def pixels(self, images):
        return torch.stack([self.transform(image) for image in images]).to(self.device)

    def batch_loss(self, images, rows):
        sequences = [torch.tensor(self.tokenizer.encode(canonical_json(row["label"]), self.config["data"]["max_length"]),
                                  dtype=torch.long) for row in rows]
        pad = self.tokenizer.token_to_id["<pad>"]
        tokens = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad).to(self.device)
        logits = self.captioner(self.pixels(images), tokens[:, :-1])
        targets = tokens[:, 1:]
        losses = F.cross_entropy(logits.float().transpose(1, 2), targets, ignore_index=pad, reduction="none")
        loss = (losses.sum(1) / targets.ne(pad).sum(1).clamp_min(1)).mean()
        return loss, {"generation_loss": float(loss.detach())}

    @torch.no_grad()
    def generate_text(self, images):
        sequences = self.captioner.generate(self.pixels(images), self.tokenizer,
                                             max_new_tokens=self.config["decoding"]["max_new_tokens"])
        return [self.tokenizer.decode(sequence.tolist()) for sequence in sequences]


def device_for(config, override=None):
    device = torch.device(override or config["training"]["device"])
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("This experiment requires a BF16-capable CUDA GPU; use --device cpu only for CPU execution")
    return device


def precision(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def make_model(config, stats, device, transfer=None, processor_path=None):
    return QwenMeal(config, stats, device, processor_path) if config["backend"] == "qwen" else SiglipMeal(config, stats, device, transfer)


def loader_for(rows, config, training=False, generator=None):
    return DataLoader(MealData(rows, config, training), shuffle=training, generator=generator,
                      batch_size=config["training"]["batch_size" if training else "eval_batch_size"],
                      num_workers=config["data"]["num_workers"], collate_fn=raw_collate)


def optimizer_for(model, config):
    groups = {}
    settings = config["training"]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        fresh = config["backend"] == "siglip" and any(key in name for key in ("embedding.weight", "position.weight"))
        lr = settings["new_parameters_lr"] if fresh else settings["learning_rate"]
        decay = settings["weight_decay"] if parameter.ndim > 1 and not fresh else 0.0
        groups.setdefault((lr, decay), {"params": [], "lr": lr, "weight_decay": decay})["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), betas=tuple(settings["betas"]), eps=settings["eps"])


def trainable_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_parameters(model, saved):
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if expected != set(saved):
        raise ValueError("Checkpoint trainable parameter names differ")
    with torch.no_grad():
        for name, value in saved.items():
            parameters[name].copy_(value)


def run_epoch(model, loader, config, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total, count, group_count, updates = {}, 0, 0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    accumulation = config["training"]["gradient_accumulation_steps"]
    for index, (images, rows) in enumerate(tqdm(loader, desc="train" if training else "val loss", leave=False)):
        with torch.set_grad_enabled(training), precision(device):
            loss, logged = model.batch_loss(images, rows)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite meal loss")
        batch = len(rows)
        logged["loss"] = float(loss.detach())
        for key, value in logged.items():
            total[key] = total.get(key, 0.0) + value * (1 if key.startswith("aux_item_") else batch)
        count += batch
        if training:
            (loss * batch).backward()
            group_count += batch
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_count)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["training"]["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                group_count = 0
    result = {key: value / count for key, value in total.items() if not key.startswith("aux_item_")}
    if "aux_item_tp" in total:
        tp, fp, fn = (total[f"aux_item_{key}"] for key in ("tp", "fp", "fn"))
        result["aux_item_micro_f1_at_0.5"] = 2 * tp / max(1, 2 * tp + fp + fn)
    return result, updates


@torch.no_grad()
def generate_records(model, loader, device):
    model.eval()
    records = []
    started = time.perf_counter()
    for images, rows in tqdm(loader, desc="generate JSON", leave=False):
        with precision(device):
            texts = model.generate_text(images)
        for row, text in zip(rows, texts):
            start, end = text.find("{"), text.rfind("}")
            extracted = text[start:end + 1] if start >= 0 and end > start else text
            records.append({"image_id": row["image_id"], "label": row.get("label"), "raw_output": text,
                            "parsed": parse_and_validate(extracted)})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return records, time.perf_counter() - started


def check_json_lengths(rows, config):
    if config["backend"] == "siglip":
        longest = max(len(canonical_json(row["label"])) + 2 for row in rows)
        if longest > config["data"]["max_length"]:
            raise ValueError(f"JSON needs {longest} characters; increase data.max_length (no truncation allowed)")


def train(args):
    source = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    config = copy.deepcopy(source["config"]) if source else experiment_config(args.config, args.experiment)
    if source and source.get("format") != "strong_meal_v1":
        raise ValueError("Expected a strong_meal checkpoint")
    if args.seed is not None:
        if source and args.seed != config["seed"]:
            raise ValueError("Resume cannot change seed")
        config["seed"] = args.seed
    if args.model_path:
        config["qwen" if config["backend"] == "qwen" else "siglip"]["model_path"] = args.model_path
    if args.caption_checkpoint:
        if source:
            raise ValueError("Resume already contains migrated decoder weights")
        config["siglip"]["caption_checkpoint"] = args.caption_checkpoint
    if config["backend"] not in ("qwen", "siglip"):
        raise ValueError("backend must be qwen or siglip")
    for argument, key in ((args.batch_size, "batch_size"), (args.accumulation, "gradient_accumulation_steps")):
        if argument is not None:
            if source and argument != config["training"][key]:
                raise ValueError(f"Resume requires unchanged {key}")
            config["training"][key] = argument
    if min(config["training"][key] for key in ("batch_size", "eval_batch_size", "gradient_accumulation_steps", "epochs", "patience")) < 1:
        raise ValueError("Batch sizes, accumulation, epochs and patience must be positive")
    if config["training"]["mixed_precision"] != "bf16":
        raise ValueError("strong_meal currently supports BF16 training only")
    output = Path(args.resume).resolve().parent if source else Path(args.output_dir or config["output_dir"]) / config["experiment"] / f"seed_{config['seed']}"
    if not source and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run already exists: {output}. Use --resume or a new output directory")
    if source and args.output_dir:
        raise ValueError("Resume always writes to its original run directory")
    seed_everything(config["seed"])
    device = device_for(config, args.device)
    generator = torch.Generator().manual_seed(config["seed"])
    data_dir = Path(config["data"]["data_dir"])
    rows = {split: list(read_jsonl(data_dir / f"{split}.jsonl")) for split in ("train", "val")}
    if not all(rows.values()):
        raise ValueError("Training and validation manifests must be nonempty")
    if {r["image_id"] for r in rows["train"]} & {r["image_id"] for r in rows["val"]}:
        raise ValueError("Training and validation image IDs overlap")
    provenance = {split: sha256_file(data_dir / f"{split}.jsonl") for split in rows}
    if source and provenance != source["provenance"]:
        raise ValueError("Resume manifests differ from the original run")
    stats = source["statistics"] if source else training_statistics(rows["train"])
    transfer = None
    if config["backend"] == "siglip" and not source:
        checkpoint = config["siglip"]["caption_checkpoint"]
        transfer = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if transfer.get("format") != "siglip_captioner_v1" or transfer["config"]["encoder"].get("type", "siglip") != "siglip":
            raise ValueError("SigLIP transfer requires a strong_captioner SigLIP checkpoint")
        if transfer.get("adapter"):
            raise ValueError("Transfer currently requires the frozen SigLIP checkpoint without vision LoRA")
        config["caption_architecture"] = copy.deepcopy(transfer["config"])
        config["caption_checkpoint_sha256"] = sha256_file(checkpoint)
        stats["tokenizer"] = JSONTokenizer.build([canonical_json(row["label"]) for row in rows["train"]]).token_to_id
    for split in rows:
        check_json_lengths(rows[split], config)
    model = make_model(config, stats, device, transfer, str(output / "processor") if source and config["backend"] == "qwen" else None)
    del transfer
    optimizer = optimizer_for(model, config)
    training_loader = loader_for(rows["train"], config, True, generator)
    validation_loader = loader_for(rows["val"], config)
    from experiments.task1_captioning.strong_captioner import make_scheduler
    steps_per_epoch = math.ceil(len(training_loader) / config["training"]["gradient_accumulation_steps"])
    scheduler = make_scheduler(optimizer, config, steps_per_epoch)
    history, best, stale, start, global_step = [], -math.inf, 0, 0, 0
    if source:
        restore_parameters(model, source["parameters"])
        optimizer.load_state_dict(source["optimizer"])
        scheduler.load_state_dict(source["scheduler"])
        history, best, stale, start, global_step = (source[key] for key in ("history", "best_micro_f1", "stale", "epoch", "global_step"))
        random.setstate(source["rng"]["python"])
        np.random.set_state(source["rng"]["numpy"])
        torch.set_rng_state(source["rng"]["torch"])
        generator.set_state(source["rng"]["loader"])
        if device.type == "cuda" and source["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(source["rng"]["cuda"])
        del source
    output.mkdir(parents=True, exist_ok=True)
    if config["backend"] == "qwen":
        model.processor.save_pretrained(output / "processor")
    known = set(stats["item_names"])
    val_names = [name for row in rows["val"] for name in _item_names(row["label"])]
    metadata = {"total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "provenance": provenance, "lora_targets": getattr(model, "lora_targets", []),
                "unseen_val_item_occurrence_rate": sum(name not in known for name in val_names) / max(1, len(val_names)),
                "unseen_val_names": sorted(set(val_names) - known),
                "effective_batch_size": config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"],
                "torch_version": str(torch.__version__), "train_images": len(rows["train"]), "val_images": len(rows["val"])}
    metadata["versions"] = {name: version(name) for name in ("transformers", "peft", "numpy")}
    if config["backend"] == "qwen":
        metadata["base_model_config"] = model.base.config.to_dict()
        metadata["preprocessing"] = model.processor.image_processor.to_dict()
    else:
        metadata["base_model_config"] = model.captioner.vision.config.to_dict()
        metadata["preprocessing"] = config["caption_architecture"]["preprocessing"]
    write_json(output / "run_config.json", config)
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "statistics.json", stats)
    for epoch in range(start, config["training"]["epochs"]):
        if stale >= config["training"]["patience"]:
            break
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        training_metrics, updates = run_epoch(model, training_loader, config, device, optimizer, scheduler)
        validation_loss, _ = run_epoch(model, validation_loader, config, device)
        records, inference_seconds = generate_records(model, validation_loader, device)
        metrics = structured_metrics(records, include_text=False)
        score = metrics["item_micro_f1"]
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation micro-F1")
        improved = score > best
        best, stale = max(best, score), 0 if improved else stale + 1
        global_step += updates
        entry = {"epoch": epoch + 1, "global_step": global_step, "train": training_metrics,
                 "val_loss": validation_loss, "val_metrics": metrics, "val_micro_f1": score,
                 "seconds": time.perf_counter() - started, "inference_seconds": inference_seconds,
                 "learning_rates": scheduler.get_last_lr(),
                 "peak_memory_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                 "peak_memory_reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        history.append(entry)
        state = {"format": "strong_meal_v1", "config": config, "statistics": stats, "metadata": metadata,
                 "parameters": trainable_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch + 1, "global_step": global_step, "best_micro_f1": best, "stale": stale,
                 "history": history, "provenance": provenance,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "loader": generator.get_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}}
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
            write_jsonl(records, output / "best_val_predictions.jsonl")
            write_json(output / "best_val_metrics.json", metrics)
        del state
        write_json(output / "training_history.json", history)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    write_json(output / "completed.json", {"best_checkpoint": "best.pt", "best_micro_f1": best})


def load_run(args):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "strong_meal_v1":
        raise ValueError("Use a strong_meal checkpoint")
    config = copy.deepcopy(state["config"])
    if args.model_path:
        config["qwen" if config["backend"] == "qwen" else "siglip"]["model_path"] = args.model_path
    if getattr(args, "image_root", None):
        config["data"]["image_root"] = args.image_root
    device = device_for(config, args.device)
    processor = Path(args.checkpoint).parent / "processor"
    model = make_model(config, state["statistics"], device, processor_path=str(processor) if config["backend"] == "qwen" else None)
    restore_parameters(model, state["parameters"])
    return model.eval(), config, device


def evaluate(args):
    model, config, device = load_run(args)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("Evaluation batch size must be positive")
        config["training"]["eval_batch_size"] = args.batch_size
    manifest = Path(args.manifest or Path(config["data"]["data_dir"]) / f"{args.split}.jsonl")
    rows = list(read_jsonl(manifest))
    if not rows:
        raise ValueError("Empty evaluation manifest")
    records, seconds = generate_records(model, loader_for(rows, config), device)
    output = Path(args.output or Path(args.checkpoint).parent / "evaluation" / f"{args.split}.json")
    predictions = output.with_name(output.stem + "_predictions.jsonl")
    write_jsonl(records, predictions)
    metrics = structured_metrics(records)
    metrics.update(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
                   manifest=str(manifest), manifest_sha256=sha256_file(manifest), decoding=config["decoding"],
                   backend=config["backend"], model_config=config["qwen"] if config["backend"] == "qwen" else config["caption_architecture"]["encoder"],
                   preprocessing=model.processor.image_processor.to_dict() if config["backend"] == "qwen" else config["caption_architecture"]["preprocessing"],
                   inference_seconds=seconds, seconds_per_image=seconds / len(rows), prediction_path=str(predictions))
    write_json(output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


@torch.no_grad()
def predict(args):
    model, _, device = load_run(args)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    with precision(device):
        text = model.generate_text([image])[0]
    result = {"image": args.image, "raw_output": text}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", default=DEFAULT_CONFIG)
    training.add_argument("--experiment", default="qwen_lora")
    training.add_argument("--resume")
    training.add_argument("--seed", type=int)
    training.add_argument("--output-dir")
    training.add_argument("--caption-checkpoint")
    training.add_argument("--batch-size", type=int)
    training.add_argument("--accumulation", type=int)
    training.set_defaults(callback=train)
    for name, callback in (("evaluate", evaluate), ("predict", predict)):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output")
        if name == "evaluate":
            command.add_argument("--batch-size", type=int)
            command.add_argument("--split", choices=["val", "test"], default="val")
            command.add_argument("--manifest")
            command.add_argument("--image-root")
        else:
            command.add_argument("--image", required=True)
        command.set_defaults(callback=callback)
    for command in (training, commands.choices["evaluate"], commands.choices["predict"]):
        command.add_argument("--device")
        command.add_argument("--model-path")
    listing = commands.add_parser("list-experiments")
    listing.add_argument("--config", default=DEFAULT_CONFIG)
    listing.set_defaults(callback=lambda args: print("\n".join(load_yaml(args.config)["experiments"])))
    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
