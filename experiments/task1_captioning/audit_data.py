"""用 CPU 审计训练数据划分、文件来源及词表，不加载模型。"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

from utils.files import sha256_file
from utils.jsonl import read_jsonl

from .tokenizer import Vocabulary


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def overlaps(rows, field):
    values = {split: {str(row[field]) for row in records if row.get(field) not in (None, "")}
              for split, records in rows.items()}
    missing = {split: sum(row.get(field) in (None, "") for row in records)
               for split, records in rows.items()}
    pairs = {}
    for left, right in combinations(rows, 2):
        common = sorted(values[left] & values[right])
        pairs[f"{left}/{right}"] = {"count": len(common), "examples": common[:5]}
    return {"complete": not any(missing.values()), "missing_rows": missing, "pairs": pairs}


def pixel_hashes(rows, image_root):
    from PIL import Image

    hashed = {}
    for split, records in rows.items():
        hashed[split] = []
        for index, row in enumerate(records):
            with Image.open(image_root / row["image_path"]) as source:
                image = source.convert("RGB")
                digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()
            hashed[split].append({"pixel_sha256": digest})
            if (index + 1) % 1000 == 0:
                print(f"Pixel audit: {split} {index + 1}/{len(records)}", flush=True)
    return overlaps(hashed, "pixel_sha256")


def audit(args):
    run = Path(args.run_dir)
    config = read_json(run / "run_config.json")
    metadata_path = run / "run_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    manifests = {split: Path(config["data"][f"{split}_manifest"]) for split in ("train", "val", "test")}
    if args.data_dir:
        manifests = {split: Path(args.data_dir) / path.name for split, path in manifests.items()}
    rows = {split: list(read_jsonl(path)) for split, path in manifests.items()}
    issues, unchecked = [], []
    report = {"run_dir": str(run), "counts": {split: len(records) for split, records in rows.items()}}
    for split, records in rows.items():
        if not records:
            issues.append(f"{split}: empty manifest")
    report["manifest_identity"] = {}
    provenance = metadata.get("provenance", {})
    for split in ("train", "val"):
        actual = sha256_file(manifests[split])
        expected = provenance.get(f"{split}_sha256")
        report["manifest_identity"][split] = {
            "current_sha256": actual, "training_record_sha256": expected,
            "matches": actual == expected if expected else None,
        }
        if not expected:
            unchecked.append(f"{split}: training manifest hash unavailable")
        elif actual != expected:
            issues.append(f"{split}: manifest differs from the training record")

    for field in ("image_id", "image_sha256"):
        result = overlaps(rows, field)
        report[field] = result
        if not result["complete"]:
            unchecked.append(f"{field}: missing values; zero observed overlaps cannot prove separation")
        for pair, item in result["pairs"].items():
            if item["count"]:
                issues.append(f"{field}: {pair} has {item['count']} overlaps")

    saved_vocab = run / "vocab.json"
    if saved_vocab.exists():
        vocabulary = read_json(saved_vocab)
        tokens = {split: {word for row in records for caption in row["captions"]
                          for word in Vocabulary.tokenize(caption)} for split, records in rows.items()}
        extra = set(vocabulary) - tokens["train"] - set(Vocabulary.specials)
        missing = tokens["train"] - set(vocabulary)
        report["vocabulary"] = {
            "size": len(vocabulary), "extra_tokens_not_in_train": len(extra),
            "extra_examples": sorted(extra)[:20], "train_tokens_not_in_vocab": len(missing),
            "extra_also_in_val": len(extra & tokens["val"]),
            "extra_also_in_test": len(extra & tokens["test"]),
        }
        if extra:
            issues.append("Vocabulary contains tokens absent from train; investigate its source (not proof of label leakage)")
        if missing:
            issues.append("Vocabulary omits training tokens; check whether it used frequency filtering or a different train split")
    else:
        unchecked.append("Saved run vocabulary unavailable")

    metrics_path = Path(args.metrics) if args.metrics else run / "evaluation" / "test_beam3.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        expected = metrics.get("manifest_sha256")
        actual = sha256_file(manifests["test"])
        report["test_evaluation"] = {
            "metrics_path": str(metrics_path), "current_sha256": actual, "evaluation_sha256": expected,
            "matches": actual == expected if expected else None,
            "count_matches": metrics.get("num_samples") == len(rows["test"]),
        }
        if not expected:
            unchecked.append("Test metrics do not record a manifest hash")
        elif actual != expected:
            issues.append("Test manifest differs from the evaluated manifest")
        if metrics.get("num_samples") != len(rows["test"]):
            issues.append("Test sample count differs from the evaluated count")
    else:
        unchecked.append("Test metrics unavailable; cannot link this audit to the reported test score")

    info = manifests["train"].parent / "dataset_info.json"
    if info.exists():
        report["preparation_deduplication"] = read_json(info).get("deduplication")
    if args.pixel_hashes:
        root = Path(args.image_root or config["data"]["image_root"])
        report["decoded_rgb"] = pixel_hashes(rows, root)
        for pair, item in report["decoded_rgb"]["pairs"].items():
            if item["count"]:
                issues.append(f"Decoded RGB: {pair} has {item['count']} identical pixel hashes")
    report["status"] = "needs_review" if issues else "incomplete" if unchecked else "passed_checked_items"
    report["issues"], report["unchecked"] = issues, unchecked
    report["limits"] = [
        "image_sha256 compares declared original-file hashes; --pixel-hashes additionally reads actual images",
        "No perceptual near-duplicate or pretrained-corpus overlap audit",
        "Cannot determine whether humans selected models or decoding settings using test scores",
        "Does not load checkpoints; run configuration, vocabulary and metadata must belong to the reported run",
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        target = Path(args.output)
        inputs = [*manifests.values(), run / "run_config.json", metadata_path, saved_vocab, metrics_path, info]
        if target.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("Report output must not overwrite an audit input file")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if issues else 2 if unchecked else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory with run_config.json, run_metadata.json and vocab.json")
    parser.add_argument("--data-dir", help="Override manifest directory after moving the dataset")
    parser.add_argument("--metrics", help="Defaults to evaluation/test_beam3.json inside the run")
    parser.add_argument("--pixel-hashes", action="store_true", help="Also read images and compare decoded RGB pixels")
    parser.add_argument("--image-root", help="Override image root for --pixel-hashes")
    parser.add_argument("--output", help="Optional JSON report path; input data and model files are never modified")
    raise SystemExit(audit(parser.parse_args()))


if __name__ == "__main__":
    main()
