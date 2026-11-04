"""读取独立实验共用的数据清单和图片根目录。"""

import json
from pathlib import Path

from utils.jsonl import read_jsonl


def manifest_rows(config, split):
    """读取非空数据划分，并要求图片编号唯一。"""
    rows = list(read_jsonl(Path(config["data_dir"]) / f"{split}.jsonl"))
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError(f"{split} manifest must be nonempty with unique image IDs")
    return rows


def image_root(config):
    value = config["image_root"]
    if not value:
        value = json.loads(
            (Path(config["data_dir"]) / "dataset_info.json").read_text(encoding="utf-8"),
        )["image_root"]
    return Path(value)
