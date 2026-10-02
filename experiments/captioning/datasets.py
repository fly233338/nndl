from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from utils.jsonl import read_jsonl


def image_transform(size: int = 224, train: bool = False):
    ops = [transforms.Resize((size, size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transforms.Compose(ops)


class CaptionDataset:
    def __init__(self, manifest: str | Path, image_root: str | Path, vocab, max_length: int = 64, train: bool = False):
        self.rows = list(read_jsonl(manifest))
        self.image_root = Path(image_root)
        self.vocab = vocab
        self.max_length = max_length
        self.transform = image_transform(train=train)
        self.items = [(row, caption) for row in self.rows for caption in row.get("captions", [""])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        row, caption = self.items[index]
        image = Image.open(self.image_root / row["image_path"]).convert("RGB")
        return self.transform(image), torch.tensor(self.vocab.encode(caption, self.max_length), dtype=torch.long), row["image_id"]


def collate_caption(batch: list[Any], pad_id: int):
    images, sequences, ids = zip(*batch)
    max_len = max(sequence.numel() for sequence in sequences)
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        padded[i, :sequence.numel()] = sequence
    return torch.stack(images), padded, ids
