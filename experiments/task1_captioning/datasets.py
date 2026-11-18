"""读取图像描述数据，并构造训练和验证批次。"""

import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from utils.jsonl import read_jsonl


def resolve_preprocessing(config, model_name):
    """新训练读取模型声明；旧检查点推理不调用此函数。"""
    if config.get("preprocessing") is not None:
        return dict(config["preprocessing"])
    if model_name == "vit_transformer" and config.get("vit_path"):
        path = Path(config["vit_path"]) / "preprocessor_config.json"
        settings = json.loads(path.read_text(encoding="utf-8"))
        return {"mean": settings["image_mean"], "std": settings["image_std"],
                "resize_mode": "stretch", "interpolation": "bilinear"}
    return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "resize_mode": "stretch", "interpolation": "bilinear"}


class CaptionTransform:
    """按描述约束增强，验证与推理只执行固定预处理。"""

    def __init__(self, size=224, train=False, config=None):
        config = config or {}
        self.size, self.train = size, train
        self.preprocess = config.get("preprocessing", {})
        self.augment = config.get("augmentation", {})
        self.mean = self.preprocess.get("mean", [0.485, 0.456, 0.406])
        self.std = self.preprocess.get("std", [0.229, 0.224, 0.225])
        self.fill = tuple(round(value * 255) for value in self.mean)
        self.tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])

    def __call__(self, image, caption=""):
        if self.preprocess.get("exif_transpose", False):
            image = ImageOps.exif_transpose(image)
        words = set(re.findall(r"[a-z]+", caption.lower()))
        directional = bool(words.intersection(self.augment.get("direction_words", [])))
        colored = bool(words.intersection(self.augment.get("color_words", [])))
        if self.train:
            flip = self.augment.get("horizontal_flip", 0.5)
            if not directional and torch.rand(()).item() < flip:
                image = ImageOps.mirror(image)
            rotation = float(self.augment.get("rotation", 0))
            if rotation and not directional:
                angle = float(torch.empty(()).uniform_(-rotation, rotation))
                image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=self.fill)
            if not colored:
                jitter = transforms.ColorJitter(**{
                    key: self.augment.get(key, 0)
                    for key in ("brightness", "contrast", "saturation", "hue")
                })
                image = jitter(image)
        interpolation = self.preprocess.get("interpolation", "bilinear")
        resample = Image.Resampling.BICUBIC if interpolation == "bicubic" else Image.Resampling.BILINEAR
        if self.preprocess.get("resize_mode", "stretch") == "letterbox":
            image = ImageOps.pad(image, (self.size, self.size), method=resample, color=self.fill)
        else:
            image = image.resize((self.size, self.size), resample)
        pixels = self.tensor(image)
        erasing = self.augment.get("random_erasing", 0)
        if self.train and erasing:
            pixels = transforms.RandomErasing(
                p=erasing, scale=self.augment["erasing_scale"], ratio=self.augment["erasing_ratio"], value=0,
            )(pixels)
        return pixels


def image_transform(size: int = 224, train: bool = False, config=None):
    # 无新增配置时沿用旧版归一化和缩放，保证历史检查点评测口径。
    return CaptionTransform(size, train, config)


class CaptionDataset:
    """将图片的多条参考描述展开为训练样本。"""

    def __init__(
        self,
        manifest: str | Path,
        image_root: str | Path,
        vocab,
        max_length: int = 64,
        train: bool = False,
        image_size: int = 224,
        config: dict | None = None,
        unique_images: bool = False,
    ):
        self.rows = list(read_jsonl(manifest))
        self.image_root = Path(image_root)
        self.vocab = vocab
        self.max_length = max_length
        self.transform = image_transform(image_size, train=train, config=config)
        self.references = {row["image_id"]: row["captions"] for row in self.rows}
        self.items = [
            (row, caption)
            for row in self.rows
            for caption in (row["captions"][:1] if unique_images else row["captions"])
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        row, caption = self.items[index]
        with Image.open(self.image_root / row["image_path"]) as source:
            image = source.convert("RGB")
        return (
            self.transform(image, caption),
            torch.tensor(self.vocab.encode(caption, self.max_length), dtype=torch.long),
            row["image_id"],
        )


def collate_caption(batch: list[Any], pad_id: int):
    """按批次最长序列补齐词元，保留对应的图片编号。"""
    images, sequences, ids = zip(*batch)
    max_len = max(sequence.numel() for sequence in sequences)
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        padded[i, : sequence.numel()] = sequence
    return torch.stack(images), padded, ids
