"""Convert the Hugging Face Food500Cap Parquet files to project manifests."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
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
    grouped: dict[str, dict[str, Any]] = {}
    for parquet_path in files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=128, columns=["image", "caption"]):
            for row in batch.to_pylist():
                payload, source_path = _image_payload(row["image"])
                if payload is None:
                    raise ValueError(f"{parquet_path}: image bytes are missing")
                key = source_path or hashlib.sha1(payload).hexdigest()
                if key not in grouped:
                    suffix = Path(source_path).suffix.lower() if source_path else ".jpg"
                    relative_path = Path(split) / f"{len(grouped):08d}{suffix}"
                    target = image_dir / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(BytesIO(payload)).convert("RGB").save(target)
                    grouped[key] = {
                        "image_id": hashlib.sha1(key.encode("utf-8")).hexdigest()[:16],
                        "image_path": relative_path.as_posix(),
                        "captions": [],
                    }
                grouped[key]["captions"].append(str(row["caption"]).strip())
    return list(grouped.values())


def prepare(dataset_dir: str, output_dir: str, val_ratio: float = 0.1, seed: int = 2026) -> None:
    source = Path(dataset_dir)
    train_files = sorted(source.glob("data/train-*.parquet")) or sorted(source.glob("train-*.parquet"))
    test_files = sorted(source.glob("data/test-*.parquet")) or sorted(source.glob("test-*.parquet"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Expected train-*.parquet and test-*.parquet under {source}")

    output = Path(output_dir)
    image_dir = output / "images"
    samples = _read_split(train_files, "train", image_dir)
    random.Random(seed).shuffle(samples)
    cut = int(len(samples) * (1 - val_ratio))
    train, val = samples[:cut], samples[cut:]
    test = _read_split(test_files, "test", image_dir)

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, output / "train.jsonl")
    write_jsonl(val, output / "val.jsonl")
    write_jsonl(test, output / "test.jsonl")
    (output / "dataset_info.json").write_text(
        json.dumps(
            {"num_train": len(train), "num_val": len(val), "num_test": len(test), "image_root": str(image_dir)},
            indent=2,
        ),
        encoding="utf-8",
    )
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
