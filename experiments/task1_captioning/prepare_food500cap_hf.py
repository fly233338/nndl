"""将 Hugging Face Food500Cap Parquet 转换为项目数据清单。"""

import argparse
import hashlib
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from utils.jsonl import write_jsonl


def _image_payload(value: Any) -> tuple[bytes | None, str | None]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        payload = value.get("bytes")
        path = value.get("path")
        return (bytes(payload) if payload is not None else None, str(path) if path else None)

    if isinstance(value, (bytes, bytearray)):
        return bytes(value), None
    return None, None


def _read_split(files: list[Path], split: str, image_dir: Path) -> list[dict[str, Any]]:
    """按原始图片内容合并参考描述，沿用首条记录的编号并保存 RGB 图片。"""
    grouped: dict[str, dict[str, Any]] = {}
    for parquet_path in files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=128, columns=["image", "caption"]):
            for row in batch.to_pylist():
                payload, source_path = _image_payload(row["image"])
                if payload is None:
                    raise ValueError(f"{parquet_path}: image bytes are missing")
                image_sha256 = hashlib.sha256(payload).hexdigest()
                if image_sha256 not in grouped:
                    # 保留原有编号规则，避免改变官方测试图片与已有预测的对应关系。
                    source_key = source_path or hashlib.sha1(payload).hexdigest()
                    suffix = Path(source_path).suffix.lower() if source_path else ".jpg"
                    relative_path = Path(split) / f"{len(grouped):08d}{suffix}"
                    target = image_dir / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(BytesIO(payload)).convert("RGB").save(target)
                    grouped[image_sha256] = {
                        "image_id": hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16],
                        "image_path": relative_path.as_posix(),
                        "image_sha256": image_sha256,
                        "captions": [],
                    }
                grouped[image_sha256]["captions"].append(str(row["caption"]).strip())
    return list(grouped.values())


def prepare(dataset_dir: str, output_dir: str, val_ratio: float = 0.1, seed: int = 2026) -> None:
    """保留官方测试集，排除训练中的相同图片后再划分训练集与验证集。"""
    source = Path(dataset_dir)
    train_files = sorted(source.glob("data/train-*.parquet")) or sorted(
        source.glob("train-*.parquet"),
    )
    test_files = sorted(source.glob("data/test-*.parquet")) or sorted(source.glob("test-*.parquet"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Expected train-*.parquet and test-*.parquet under {source}")

    output = Path(output_dir)
    image_dir = output / "images"
    samples = _read_split(train_files, "train", image_dir)
    test = _read_split(test_files, "test", image_dir)
    train_unique_images = len(samples)
    test_hashes = {row["image_sha256"] for row in test}

    # 只比较图片内容，不读取测试描述来筛选训练样本。
    samples = [row for row in samples if row["image_sha256"] not in test_hashes]
    removed_train_images = train_unique_images - len(samples)
    random.Random(seed).shuffle(samples)
    cut = int(len(samples) * (1 - val_ratio))
    train, val = samples[:cut], samples[cut:]
    train_ids = {row["image_id"] for row in train}
    val_ids = {row["image_id"] for row in val}
    test_ids = {row["image_id"] for row in test}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("image split overlap detected")

    train_hashes = {row["image_sha256"] for row in train}
    val_hashes = {row["image_sha256"] for row in val}
    if train_hashes & val_hashes or train_hashes & test_hashes or val_hashes & test_hashes:
        raise ValueError("image content overlap detected")

    if any(not caption for row in samples + test for caption in row["captions"]):
        raise ValueError("empty caption detected")

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, output / "train.jsonl")
    write_jsonl(val, output / "val.jsonl")
    write_jsonl(test, output / "test.jsonl")
    train_source_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in train_files)
    (output / "dataset_info.json").write_text(
        json.dumps(
            {
                "num_train": len(train),
                "num_val": len(val),
                "num_test": len(test),
                "image_root": str(image_dir),
                "deduplication": {
                    "method": "sha256_of_original_image_bytes",
                    "train_source_rows": train_source_rows,
                    "train_unique_images_before_test_filter": train_unique_images,
                    "train_duplicate_rows_merged": train_source_rows - train_unique_images,
                    "train_images_overlapping_test_removed": removed_train_images,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"removed {removed_train_images} training images duplicated in the official test set")
    print(f"train={len(train)} val={len(val)} test={len(test)} images={image_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed/food500cap")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    prepare(args.dataset_dir, args.output_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
