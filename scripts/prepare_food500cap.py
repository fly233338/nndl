"""Convert Food-500 Cap JSON files into image-level train/val/test manifests."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.jsonl import write_jsonl


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(train_json: str, test_json: str, image_root: str, output_dir: str, val_ratio: float = 0.1, seed: int = 2026) -> None:
    rows = _load_rows(Path(train_json))
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image"])].append(str(row["caption"]).strip())

    image_root_path = Path(image_root)
    samples = [{"image_id": image, "image_path": image, "captions": captions} for image, captions in sorted(grouped.items())]
    random.Random(seed).shuffle(samples)
    cut = int(len(samples) * (1 - val_ratio))
    train, val = samples[:cut], samples[cut:]
    test_rows = _load_rows(Path(test_json))
    test_grouped: dict[str, list[str]] = defaultdict(list)
    for row in test_rows:
        test_grouped[str(row["image"])].append(str(row["caption"]).strip())
    test = [{"image_id": image, "image_path": image, "captions": captions} for image, captions in sorted(test_grouped.items())]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, output / "train.jsonl")
    write_jsonl(val, output / "val.jsonl")
    write_jsonl(test, output / "test.jsonl")
    (output / "dataset_info.json").write_text(
        json.dumps({"num_train": len(train), "num_val": len(val), "num_test": len(test), "image_root": str(image_root_path)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    prepare(args.train_json, args.test_json, args.image_root, args.output_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
