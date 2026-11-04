"""以图片为单位构造 SCST 批次，复用任务一的模型结构。"""

import json
from pathlib import Path

import torch
from PIL import Image

from experiments.task1_captioning.datasets import image_transform
from experiments.task1_captioning.models import build_model
from experiments.task1_captioning.tokenizer import Vocabulary
from utils.jsonl import read_jsonl


def manifest(config, split):
    rows = list(read_jsonl(Path(config["data_dir"]) / f"{split}.jsonl"))
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError(f"{split} needs nonempty rows with unique image IDs")
    for row in rows:
        if not row["captions"] or any(
            not isinstance(c, str) or not c.strip() for c in row["captions"]
        ):
            raise ValueError(f"Empty reference captions for {row['image_id']}")
    return rows


def image_root(config):
    if config["image_root"]:
        return Path(config["image_root"])
    info = Path(config["data_dir"]) / "dataset_info.json"
    return Path(json.loads(info.read_text(encoding="utf-8"))["image_root"])


def disjoint(train_rows, other_rows):
    if {r["image_id"] for r in train_rows} & {r["image_id"] for r in other_rows}:
        raise ValueError("Training and evaluation image IDs overlap")

    if {r["image_path"] for r in train_rows} & {r["image_path"] for r in other_rows}:
        raise ValueError("Training and evaluation image paths overlap")


def load_model(path, config, device):
    """恢复任务一 ViT-Transformer，冻结视觉编码器，仅微调解码器。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["model_name"] != "vit_transformer":
        raise ValueError("This SCST experiment starts from the Task 1 best ViT-Transformer")
    model_config = dict(state["config"])
    if config["vit_path"]:
        model_config["vit_path"] = config["vit_path"]
    model_config["freeze_encoder"] = True
    vocab = Vocabulary(state["vocab"])
    model = build_model(state["model_name"], len(vocab.token_to_id), model_config)
    model.load_state_dict(state["model"])
    model.encoder.requires_grad_(False)
    model.to(device).eval()
    return model, vocab, state, model_config


def resolve_device(config):
    if config["device"] == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config["device"]


class ImageDataset:
    """每张图片只出现一次，多条参考描述统一用于奖励计算。"""

    def __init__(self, rows, root, image_size):
        self.rows, self.root = rows, root
        self.transform = image_transform(image_size, train=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, row["image_id"]


def collate_images(batch):
    images, ids = zip(*batch)
    return torch.stack(images), list(ids)


def make_loader(rows, config, image_size, shuffle=False):
    dataset = ImageDataset(rows, image_root(config), image_size)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        collate_fn=collate_images,
    )
