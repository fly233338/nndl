"""ViT / SigLIP + 自建 Transformer 的独立架构对照。"""

import argparse
import copy
import json
import math
import random
import re
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_layer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import read_jsonl, write_jsonl
from utils.metrics import evaluate_caption_records, metric_metadata
from utils.seed import seed_everything

from .tokenizer import Vocabulary


def merge_config(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge_config(result[key], value) if isinstance(value, dict) else copy.deepcopy(value)
    return result


def experiment_config(path, name):
    config = load_yaml(path)
    name = config.pop("aliases", {}).get(name, name)
    experiments = config.pop("experiments")
    if name not in experiments:
        raise ValueError(f"Unknown experiment {name}; choose from {list(experiments)}")
    config = merge_config(config, experiments[name])
    config["experiment"] = name
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class ImageTransform:
    def __init__(self, config, training=False):
        self.preprocess = config["preprocessing"]
        self.augment = config["augmentation"]
        self.training = training and self.augment["enabled"]
        self.fill = tuple(round(x * 255) for x in self.preprocess["mean"])
        self.jitter = transforms.ColorJitter(
            brightness=self.augment["brightness"], contrast=self.augment["contrast"],
        )
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.preprocess["mean"], self.preprocess["std"]),
        ])

    def __call__(self, image, captions=""):
        image = image.convert("RGB")
        if self.training:
            words = set(re.findall(r"[a-z]+", captions.lower()))
            directional = bool(words.intersection(self.augment["direction_words"]))
            colored = bool(words.intersection(self.augment["color_words"]))
            if not directional and random.random() < self.augment["horizontal_flip"]:
                image = ImageOps.mirror(image)
            if not directional and random.random() < self.augment["rotation_probability"]:
                angle = random.uniform(-self.augment["rotation_degrees"], self.augment["rotation_degrees"])
                image = image.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=self.fill)
            if not colored and random.random() < self.augment["jitter_probability"]:
                image = self.jitter(image)
        size = self.preprocess["image_size"]
        resample = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}[
            self.preprocess["interpolation"]
        ]
        image = image.resize((size, size), resample)
        return self.to_tensor(image)


class CaptionData(Dataset):
    def __init__(self, manifest, config, vocab, training=False, unique=False):
        self.rows = list(read_jsonl(manifest))
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        self.references = {row["image_id"]: row["captions"] for row in self.rows}
        if len(self.references) != len(self.rows):
            raise ValueError("Use the prepared manifest with one row per image")
        self.items = [(i, caption) for i, row in enumerate(self.rows)
                      for caption in (row["captions"][:1] if unique else row["captions"])]
        self.root = Path(config["data"]["image_root"])
        self.max_length = config["data"]["max_length"]
        self.vocab = vocab
        self.transform = ImageTransform(config, training)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row_index, caption = self.items[index]
        row = self.rows[row_index]
        with Image.open(self.root / row["image_path"]) as image:
            pixels = self.transform(image, " ".join(row["captions"]))
        tokens = torch.tensor(self.vocab.encode(caption, self.max_length), dtype=torch.long)
        return pixels, tokens, row["image_id"]


def collate(batch, pad_id):
    images, sequences, ids = zip(*batch)
    tokens = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_id)
    return torch.stack(images), tokens, ids


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def split_heads(self, x):
        return x.reshape(x.size(0), x.size(1), self.heads, self.head_dim).transpose(1, 2)

    def key_value(self, x):
        return self.split_heads(self.k(x)), self.split_heads(self.v(x))

    def attend(self, query, kv, mask=None):
        output = F.scaled_dot_product_attention(
            self.split_heads(self.q(query)), *kv, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).contiguous().flatten(2))

    def forward(self, query, memory=None, mask=None):
        return self.attend(query, self.key_value(query if memory is None else memory), mask)

    def step(self, query, past=None):
        kv = self.key_value(query)
        if past is not None:
            kv = tuple(torch.cat([old, new], dim=2) for old, new in zip(past, kv))
        # 单个新 query 可以看到全部历史 key，不使用非方形的 causal mask。
        return self.attend(query, kv), kv


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, activation, dropout):
        super().__init__()
        self.activation = activation
        self.up = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden) if activation == "swiglu" else None
        self.out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.silu(self.gate(x)) * self.up(x) if self.gate is not None else F.gelu(self.up(x))
        return self.out(self.dropout(hidden))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, heads, dropout = config["dim"], config["heads"], config["dropout"]
        self.self_norm, self.cross_norm, self.ffn_norm = [nn.LayerNorm(dim) for _ in range(3)]
        self.self_attention = Attention(dim, heads, dropout)
        self.cross_attention = Attention(dim, heads, dropout)
        self.ffn = FeedForward(dim, config["ffn_dim"], config["activation"], dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, mask):
        x = x + self.dropout(self.self_attention(self.self_norm(x), mask=mask))
        x = x + self.dropout(self.cross_attention(self.cross_norm(x), memory))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))

    def step(self, x, visual_kv, past):
        context, past = self.self_attention.step(self.self_norm(x), past)
        x = x + self.dropout(context)
        x = x + self.dropout(self.cross_attention.attend(self.cross_norm(x), visual_kv))
        return x + self.dropout(self.ffn(self.ffn_norm(x))), past


class CaptionDecoder(nn.Module):
    def __init__(self, config, feature_dim, vocab_size, pad_id):
        super().__init__()
        settings = config["model"]
        dim = settings["dim"]
        if dim % settings["heads"] or settings["activation"] not in ("swiglu", "gelu"):
            raise ValueError("Decoder dimensions/activation are invalid")
        self.pad_id = pad_id
        self.gradient_checkpointing = settings["gradient_checkpointing"]
        self.project = nn.Sequential(nn.Linear(feature_dim, dim), nn.LayerNorm(dim))
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.position = nn.Embedding(config["data"]["max_length"], dim)
        self.dropout = nn.Dropout(settings["dropout"])
        self.blocks = nn.ModuleList([DecoderBlock(settings) for _ in range(settings["layers"])])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self.initialize)
        residual_std = 0.02 / math.sqrt(3 * settings["layers"])
        for block in self.blocks:
            for layer in (block.self_attention.out, block.cross_attention.out, block.ffn.out):
                nn.init.normal_(layer.weight, std=residual_std)
        if settings["tie_embeddings"]:
            self.output.weight = self.embedding.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, memory, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.dropout(self.embedding(tokens) + self.position(positions))
        causal = torch.ones(tokens.size(1), tokens.size(1), dtype=torch.bool, device=tokens.device).tril()
        # SDPA 的布尔 True 表示允许关注，同时屏蔽 padding key。
        mask = causal[None, None] & tokens.ne(self.pad_id)[:, None, None, :]
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint_layer(block, x, memory, mask, use_reentrant=False)
            else:
                x = block(x, memory, mask)
        return self.output(self.norm(x))

    def visual_cache(self, memory):
        return [block.cross_attention.key_value(memory) for block in self.blocks]

    def step(self, token, position, visual_cache, past):
        positions = torch.full_like(token, position)
        x = self.embedding(token) + self.position(positions)
        next_cache = []
        for block, visual_kv, cached in zip(self.blocks, visual_cache, past):
            x, cached = block.step(x, visual_kv, cached)
            next_cache.append(cached)
        return self.output(self.norm(x))[:, -1], next_cache


class StrongCaptioner(nn.Module):
    def __init__(self, config, vocab, device):
        super().__init__()
        from transformers import SiglipVisionModel, ViTModel

        self.config = config
        encoder = config["encoder"]
        precision = config["training"]["mixed_precision"]
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": torch.float32}[precision]
        self.encoder_type = encoder.get("type", "siglip")
        if self.encoder_type not in ("siglip", "vit"):
            raise ValueError("encoder.type must be siglip or vit")
        visual_class = ViTModel if self.encoder_type == "vit" else SiglipVisionModel
        self.vision = visual_class.from_pretrained(
            encoder["model_path"] or encoder["model_id"], revision=encoder["revision"],
            local_files_only=encoder["local_files_only"],
            torch_dtype=dtype if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        resolved_revision = getattr(self.vision.config, "_commit_hash", None)
        if resolved_revision:
            encoder["revision"] = resolved_revision
        self.vision.requires_grad_(False)
        self.vision.eval()
        if config["preprocessing"]["image_size"] % self.vision.config.patch_size:
            raise ValueError("image_size must be divisible by the visual patch size")
        self.lora_enabled = config["lora"]["enabled"]
        self.lora_targets = []
        if self.lora_enabled:
            if self.encoder_type != "siglip":
                raise ValueError("Legacy vision LoRA is only supported for SigLIP")
            self.add_lora(config["lora"])
        self.decoder = CaptionDecoder(
            config, self.vision.config.hidden_size, len(vocab.token_to_id), vocab.token_to_id["<pad>"],
        )
        self.to(device)

    def add_lora(self, settings):
        from peft import LoraConfig, inject_adapter_in_model

        layers = self.vision.config.num_hidden_layers
        count = settings["last_layers"]
        if not 1 <= count <= layers:
            raise ValueError("LoRA last_layers is outside the encoder")
        # 同时适配不同 Transformers 版本的 encoder.layers / layers 命名。
        for name, module in self.vision.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)\..*\.(q_proj|v_proj)$", name)
            if match and int(match[1]) >= layers - count and isinstance(module, nn.Linear):
                self.lora_targets.append(name)
        if len(self.lora_targets) != count * 2:
            raise ValueError("Could not locate Q/V projections for all selected SigLIP layers")
        self.vision = inject_adapter_in_model(LoraConfig(
            r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
            target_modules=self.lora_targets, bias="none",
        ), self.vision)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        # 基础主干保持 eval，仅 LoRA dropout 随训练切换。
        if self.lora_enabled:
            for name, module in self.vision.named_modules():
                if "lora_dropout" in name:
                    module.train(mode)
        return self

    def memory(self, images):
        context = nullcontext() if self.lora_enabled and self.training else torch.no_grad()
        with context:
            patches = self.vision(pixel_values=images, interpolate_pos_encoding=True).last_hidden_state
            if self.encoder_type == "vit":
                patches = patches[:, 1:]
        return self.decoder.project(patches)

    def forward(self, images, tokens):
        return self.decoder(self.memory(images), tokens)

    def adapter_state(self):
        return {name: value for name, value in self.vision.state_dict().items() if "lora_" in name}

    def load_weights(self, state, initialize=False):
        self.decoder.load_state_dict(state["decoder"])
        adapters = state["adapter"]
        expected = set(self.adapter_state())
        if initialize and not adapters and self.lora_enabled:
            return
        if set(adapters) != expected:
            raise ValueError("Checkpoint LoRA settings differ; do not discard trained adapters")
        if adapters:
            self.vision.load_state_dict(adapters, strict=False)

    @torch.no_grad()
    def generate(self, images, vocab, beam_size=None, max_new_tokens=None):
        decode = self.config["decoding"]
        beam = beam_size or decode["beam_size"]
        if not 1 <= beam <= len(vocab.token_to_id):
            raise ValueError("beam_size must be between 1 and the vocabulary size")
        max_length = self.config["data"]["max_length"]
        if max_new_tokens is not None:
            max_length = min(max_length, max_new_tokens + 1)
        bos, eos = vocab.token_to_id["<bos>"], vocab.token_to_id["<eos>"]
        memory = self.memory(images)
        batch, device = images.size(0), images.device
        visual = [tuple(t.repeat_interleave(beam, dim=0) for t in kv)
                  for kv in self.decoder.visual_cache(memory)]
        past = [None] * len(self.decoder.blocks)
        tokens = torch.full((batch, beam, 1), bos, dtype=torch.long, device=device)
        scores = torch.full((batch, beam), -torch.inf, device=device)
        scores[:, 0] = 0
        finished = torch.zeros((batch, beam), dtype=torch.bool, device=device)
        lengths = torch.zeros((batch, beam), dtype=torch.long, device=device)
        vocab_size = len(vocab.token_to_id)
        for position in range(max_length - 1):
            logits, past = self.decoder.step(tokens[:, :, -1:].reshape(batch * beam, 1), position, visual, past)
            logits = logits.float()
            logits[:, [self.decoder.pad_id, bos]] = -torch.inf
            log_probs = logits.log_softmax(-1).reshape(batch, beam, vocab_size)
            log_probs = log_probs.masked_fill(finished[:, :, None], -torch.inf)
            log_probs[:, :, eos] = torch.where(finished, 0.0, log_probs[:, :, eos])
            candidates = scores[:, :, None] + log_probs
            candidate_lengths = lengths + (~finished).long()
            penalty = ((5 + candidate_lengths.float()) / 6).pow(decode["length_penalty"])
            ranking = candidates / penalty[:, :, None]
            indices = ranking.flatten(1).topk(beam, dim=1).indices
            parents, selected = indices // vocab_size, indices % vocab_size
            scores = candidates.flatten(1).gather(1, indices)
            lengths = candidate_lengths.gather(1, parents)
            tokens = torch.cat([tokens.gather(1, parents[:, :, None].expand(-1, -1, tokens.size(-1))),
                                selected[:, :, None]], dim=-1)
            finished = finished.gather(1, parents) | selected.eq(eos)
            flat_parents = (torch.arange(batch, device=device)[:, None] * beam + parents).flatten()
            past = [tuple(t.index_select(0, flat_parents) for t in kv) for kv in past]
            # 同一图片的视觉 K/V 相同，不必随其 beam 父节点重排。
            if finished.all():
                break
        ranking = scores / ((5 + lengths.float()) / 6).pow(decode["length_penalty"])
        best = ranking.argmax(dim=1)
        return tokens[torch.arange(batch, device=device), best]


def get_device(config, override=None):
    device = torch.device(override or config["training"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run on the training machine or select --device cpu")
        if config["training"]["mixed_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16")
        torch.set_float32_matmul_precision("high")
    return device


def autocast(config, device):
    precision = config["training"]["mixed_precision"]
    return torch.autocast(device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                          enabled=device.type == "cuda" and precision != "none")


def make_loader(dataset, config, device, training=False, generator=None, batch_size=None):
    return DataLoader(
        dataset, batch_size=batch_size or config["training"]["batch_size" if training else "eval_batch_size"],
        shuffle=training, num_workers=config["data"]["num_workers"], pin_memory=device.type == "cuda",
        collate_fn=partial(collate, pad_id=dataset.vocab.token_to_id["<pad>"]), generator=generator,
    )


def make_optimizer(model, config):
    settings = config["training"]
    embedding_ids = {id(module.weight) for module in model.decoder.modules() if isinstance(module, nn.Embedding)}
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        adapter = name.startswith("vision.")
        decay = parameter.ndim > 1 and id(parameter) not in embedding_ids
        key = adapter, decay
        if key not in groups:
            groups[key] = {
                "params": [], "lr": config["lora"]["learning_rate"] if adapter else settings["learning_rate"],
                "weight_decay": settings["weight_decay"] if decay else 0.0,
            }
        groups[key]["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), betas=tuple(settings["betas"]), eps=settings["eps"])


def make_scheduler(optimizer, config, updates_per_epoch):
    settings = config["training"]
    total = updates_per_epoch * settings["epochs"]
    warmup = max(1, round(total * settings["warmup_ratio"]))

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return settings["min_lr_ratio"] + (1 - settings["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def loss_epoch(model, loader, config, device, criterion, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    settings = config["training"]
    accumulation = settings["gradient_accumulation_steps"]
    total_loss = total_tokens = group_tokens = updates = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc="train" if training else "val loss", leave=False)
    for index, (images, tokens, _) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        target = tokens[:, 1:]
        token_count = int(target.ne(criterion.ignore_index).sum())
        with torch.set_grad_enabled(training), autocast(config, device):
            logits = model(images, tokens[:, :-1])
            loss = criterion(logits.flatten(0, 1), target.reshape(-1))
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError("Non-finite caption loss")
        total_loss += loss_value
        total_tokens += token_count
        if training:
            scaler.scale(loss).backward()
            group_tokens += token_count
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_tokens)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], settings["grad_clip"])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    updates += 1
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
        progress.set_postfix(loss=round(total_loss / total_tokens, 4))
    return total_loss / total_tokens, updates


@torch.no_grad()
def generate_records(model, loader, config, device, vocab, beam_size=None):
    model.eval()
    records = []
    for images, _, image_ids in tqdm(loader, desc="generate", leave=False):
        with autocast(config, device):
            sequences = model.generate(images.to(device, non_blocking=True), vocab, beam_size)
        for image_id, sequence in zip(image_ids, sequences):
            records.append({"image_id": image_id, "references": loader.dataset.references[image_id],
                            "prediction": vocab.decode(sequence.tolist())})
    return records


def check_checkpoint(state, config, vocab, provenance, resume):
    if state["format"] != "siglip_captioner_v1" or state["vocab"] != vocab.token_to_id:
        raise ValueError("Initialization/resume requires this experiment format and the same vocabulary")
    previous = state["config"]
    architecture = lambda c: {key: value for key, value in c["model"].items() if key != "gradient_checkpointing"}
    if architecture(previous) != architecture(config) or previous["data"]["max_length"] != config["data"]["max_length"]:
        raise ValueError("Decoder architecture differs; train a new size from scratch")
    for key in ("model_id", "revision"):
        if previous["encoder"][key] != config["encoder"][key]:
            raise ValueError(f"Visual encoder {key} changed")
    if previous["lora"]["enabled"]:
        for key in ("enabled", "last_layers", "rank", "alpha", "dropout"):
            if previous["lora"][key] != config["lora"][key]:
                raise ValueError(f"Keep the trained LoRA {key} when loading its weights")
    if state["provenance"] != provenance:
        raise ValueError("Training/validation manifests changed")
    if resume:
        for key in ("preprocessing", "augmentation", "lora", "seed"):
            if previous[key] != config[key]:
                raise ValueError(f"Resume requires the same {key}; use --init-checkpoint for a new run")
        if previous["training"] != config["training"]:
            raise ValueError("Resume requires the original training schedule; use --init-checkpoint to continue")


def train(args):
    config = experiment_config(args.config, args.experiment)
    source = args.resume or args.init_checkpoint
    source_state = torch.load(source, map_location="cpu", weights_only=False) if source else None
    if source_state:
        if source_state.get("format") != "siglip_captioner_v1":
            raise ValueError("Use a checkpoint produced by strong_captioner.py")
        previous = source_state["config"]
        if args.resume:
            config = copy.deepcopy(previous)
        if config.get("continuation", False):
            # 续训实验继承选中模型的结构，不把 S/L 检查点变回默认 M。
            for key in ("model", "augmentation"):
                config[key] = copy.deepcopy(previous[key])
            config["data"]["max_length"] = previous["data"]["max_length"]
            for key in ("mean", "std", "interpolation"):
                config["preprocessing"][key] = previous["preprocessing"][key]
        if config["encoder"]["revision"] == "main":
            config["encoder"]["revision"] = previous["encoder"]["revision"]
    for argument, key in ((args.batch_size, "batch_size"), (args.accumulation, "gradient_accumulation_steps"),
                          (args.device, "device")):
        if argument is not None:
            config["training"][key] = argument
    if args.seed is not None:
        config["seed"] = args.seed
    if args.model_path:
        config["encoder"]["model_path"] = args.model_path
    if not args.resume:
        from transformers import AutoImageProcessor
        enc = config["encoder"]
        processor = AutoImageProcessor.from_pretrained(
            enc["model_path"] or enc["model_id"], revision=enc["revision"],
            local_files_only=enc["local_files_only"],
        )
        config["preprocessing"].update(mean=list(processor.image_mean), std=list(processor.image_std),
            interpolation={2: "bilinear", 3: "bicubic"}[int(processor.resample)])
    settings = config["training"]
    if min(settings["batch_size"], settings["gradient_accumulation_steps"], settings["epochs"]) < 1:
        raise ValueError("Batch size, accumulation and epochs must be positive")
    if config.get("continuation", False) and not (args.init_checkpoint or args.resume):
        raise ValueError("Continuation experiments require --init-checkpoint (or --resume)")
    output = Path(args.output_dir or config["output_dir"]) / config["experiment"] / f"seed_{config['seed']}"
    if args.resume:
        original_output = Path(args.resume).resolve().parent
        if args.output_dir and output.resolve() != original_output:
            raise ValueError("Resume writes to the original run; use --init-checkpoint for a new directory")
        output = original_output
    if not args.resume and any((output / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError(f"Choose a new output directory to preserve {output}")
    seed_everything(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    device = get_device(config)
    data = config["data"]
    vocab_path = Path(data["vocab_path"])
    vocab = Vocabulary.load(vocab_path) if vocab_path.exists() else Vocabulary.build(
        [caption for row in read_jsonl(data["train_manifest"]) for caption in row["captions"]],
    )
    datasets = [CaptionData(data[f"{split}_manifest"], config, vocab, training, unique)
                for split, training, unique in (("train", True, False), ("val", False, False), ("val", False, True))]
    if set(datasets[0].references) & set(datasets[1].references):
        raise ValueError("Training and validation images overlap")
    loaders = [make_loader(dataset, config, device, i == 0, generator if i == 0 else None)
               for i, dataset in enumerate(datasets)]
    provenance = {f"{split}_sha256": sha256_file(data[f"{split}_manifest"]) for split in ("train", "val")}
    model = StrongCaptioner(config, vocab, device)
    optimizer = make_optimizer(model, config)
    updates_per_epoch = math.ceil(len(loaders[0]) / settings["gradient_accumulation_steps"])
    scheduler = make_scheduler(optimizer, config, updates_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and settings["mixed_precision"] == "fp16")
    history, best, early_best, stale, start, global_step = [], -math.inf, -math.inf, 0, 0, 0
    init_sha256 = sha256_file(args.init_checkpoint) if args.init_checkpoint else None
    if source_state:
        check_checkpoint(source_state, config, vocab, provenance, bool(args.resume))
        model.load_weights(source_state, initialize=bool(args.init_checkpoint))
        if args.resume:
            optimizer.load_state_dict(source_state["optimizer"])
            scheduler.load_state_dict(source_state["scheduler"])
            scaler.load_state_dict(source_state["scaler"])
            history, best, stale = source_state["history"], source_state["best_val_CIDEr-D"], source_state["stale_epochs"]
            early_best = source_state["early_stop_best"]
            init_sha256 = source_state["metadata"]["init_checkpoint_sha256"]
            start, global_step = source_state["epoch"], source_state["global_step"]
            random.setstate(source_state["rng"]["python"])
            torch.set_rng_state(source_state["rng"]["torch"])
            generator.set_state(source_state["rng"]["loader"])
            if device.type == "cuda" and source_state["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(source_state["rng"]["cuda"])
        del source_state
    output.mkdir(parents=True, exist_ok=True)
    vocab.save(output / "vocab.json")
    write_json(output / "run_config.json", config)
    counts = {"total_params": sum(p.numel() for p in model.parameters()),
              "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "visual_trainable_params": sum(p.numel() for p in model.vision.parameters() if p.requires_grad),
              "decoder_params": sum(p.numel() for p in model.decoder.parameters()),
              "visual_tokens": (config["preprocessing"]["image_size"] // model.vision.config.patch_size) ** 2}
    metadata = {**counts, "provenance": provenance, "lora_targets": model.lora_targets,
                "effective_batch_size": settings["batch_size"] * settings["gradient_accumulation_steps"],
                "updates_per_epoch": updates_per_epoch, "torch_version": str(torch.__version__),
                "visual_config": model.vision.config.to_dict(),
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                "init_checkpoint_sha256": init_sha256,
                "vocab_size": len(vocab.token_to_id),
                "train_images": len(datasets[0].rows), "train_captions": len(datasets[0]),
                "val_images": len(datasets[1].rows),
                "train_truncated_captions": sum(len(vocab.tokenize(c)) > data["max_length"] - 2
                                                for row in datasets[0].rows for c in row["captions"]),
                "val_truncated_captions": sum(len(vocab.tokenize(c)) > data["max_length"] - 2
                                              for row in datasets[1].rows for c in row["captions"])}
    write_json(output / "run_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token_to_id["<pad>"],
                                    label_smoothing=settings["label_smoothing"], reduction="sum")
    for epoch in range(start, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss, updates = loss_epoch(model, loaders[0], config, device, criterion, optimizer, scheduler, scaler)
        val_loss, _ = loss_epoch(model, loaders[1], config, device, criterion)
        inference_started = time.perf_counter()
        records = generate_records(model, loaders[2], config, device, vocab)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        cider = evaluate_caption_records(records, metrics=("CIDEr-D",))["CIDEr-D"]
        if not math.isfinite(cider):
            raise FloatingPointError("Non-finite validation CIDEr-D")
        improved = cider > best
        best = max(best, cider)
        if cider > early_best + settings["min_delta"]:
            early_best, stale = cider, 0
        else:
            stale += 1
        global_step += updates
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {"epoch": epoch + 1, "global_step": global_step, "train_loss": train_loss,
                  "val_loss": val_loss, "val_CIDEr-D": cider, "seconds": time.perf_counter() - started,
                  "inference_seconds": inference_seconds,
                  "learning_rates": scheduler.get_last_lr(),
                  "peak_memory_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "peak_memory_reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        history.append(record)
        state = {"format": "siglip_captioner_v1", "decoder": model.decoder.state_dict(),
                 "adapter": model.adapter_state(), "vocab": vocab.token_to_id, "config": config,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch + 1, "global_step": global_step, "history": history,
                 "best_val_CIDEr-D": best, "early_stop_best": early_best, "stale_epochs": stale,
                 "provenance": provenance, "metadata": metadata,
                 "rng": {"python": random.getstate(), "torch": torch.get_rng_state(), "loader": generator.get_state(),
                         "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}}
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
            write_jsonl(records, output / "best_val_predictions.jsonl")
        write_json(output / "training_history.json", history)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    print(f"Best checkpoint: {output / 'best.pt'}", flush=True)
    write_json(output / "completed.json", {"best_checkpoint": "best.pt", "best_val_CIDEr-D": best})


def load_model(args):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != "siglip_captioner_v1":
        raise ValueError("Use a checkpoint produced by strong_captioner.py")
    config = copy.deepcopy(state["config"])
    if args.model_path:
        config["encoder"]["model_path"] = args.model_path
    device = get_device(config, args.device)
    vocab = Vocabulary(state["vocab"])
    model = StrongCaptioner(config, vocab, device)
    model.load_weights(state)
    model.eval()
    return model, vocab, config, device


def evaluate(args):
    model, vocab, config, device = load_model(args)
    if args.image_root:
        config["data"]["image_root"] = args.image_root
    manifest = args.manifest or config["data"][f"{args.split}_manifest"]
    dataset = CaptionData(manifest, config, vocab, unique=True)
    loader = make_loader(dataset, config, device, batch_size=args.batch_size)
    beam = args.beam_size or config["decoding"]["beam_size"]
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "evaluation" / f"{args.split}_beam{beam}.json"
    started = time.perf_counter()
    records = generate_records(model, loader, config, device, vocab, beam)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    predictions = output.with_name(output.stem + "_predictions.jsonl")
    write_jsonl(records, predictions)
    metrics = evaluate_caption_records(records)
    metrics.update({"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
                    "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                    "preprocessing": config["preprocessing"], "decoding": {**config["decoding"], "beam_size": beam},
                    "encoder": config["encoder"], "model": config["model"], "lora": config["lora"],
                    "prediction_path": str(predictions), "inference_seconds": seconds,
                    "seconds_per_image": seconds / len(records), "metric_metadata": metric_metadata()})
    write_json(output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


@torch.no_grad()
def predict(args):
    model, vocab, config, device = load_model(args)
    with Image.open(args.image) as image:
        pixels = ImageTransform(config)(image).unsqueeze(0).to(device)
    with autocast(config, device):
        sequence = model.generate(pixels, vocab, args.beam_size)[0]
    result = {"image": args.image, "prediction": vocab.decode(sequence.tolist()),
              "checkpoint": args.checkpoint, "beam_size": args.beam_size or config["decoding"]["beam_size"]}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", default="config/task1_captioning/ablation.yaml")
    training.add_argument("--experiment", default="siglip_swiglu")
    training.add_argument("--seed", type=int)
    training.add_argument("--output-dir")
    training.add_argument("--batch-size", type=int)
    training.add_argument("--accumulation", type=int)
    source = training.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--init-checkpoint")
    for name, callback in (("evaluate", evaluate), ("predict", predict)):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--beam-size", type=int)
        command.add_argument("--output")
        if name == "evaluate":
            command.add_argument("--split", choices=["val", "test"], default="val")
            command.add_argument("--manifest")
            command.add_argument("--image-root")
            command.add_argument("--batch-size", type=int)
        else:
            command.add_argument("--image", required=True)
        command.set_defaults(callback=callback)
    training.set_defaults(callback=train)
    listing = commands.add_parser("list-experiments")
    listing.add_argument("--config", default="config/task1_captioning/ablation.yaml")
    listing.set_defaults(callback=lambda args: print("\n".join(load_yaml(args.config)["experiments"])))
    for command in (training, *[commands.choices[name] for name in ("evaluate", "predict")]):
        command.add_argument("--device")
        command.add_argument("--model-path", help="Optional local copy of the same SigLIP checkpoint")
    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
