"""分别缓存独立估重所需的视觉特征和营养校准所需的语义观察。"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from tqdm import tqdm

from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import read_jsonl

from .data import image_root, manifest_rows


def nutrition_signature(config):
    """根据实际模型、提示词和营养资源生成语义缓存版本。"""
    api = config["api"]
    payload = {
        "api": {**api, "model": os.environ.get(api.get("model_env", "")) or api["model"]},
        "endpoint": os.environ.get(api["base_url_env"], ""),
        "prompt": config["agent"]["system_prompt"],
        "table": sha256_file(config["resources"]["table_path"]),
        "mapping": sha256_file(config["resources"]["mapping_path"]),
        "schema_version": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_nutrition_cache(config):
    """校验语义缓存的版本、图片指纹和训练清单覆盖情况。"""
    path = Path(config["cache"]["nutrition_train_path"])
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if metadata["feature_signature"] != nutrition_signature(config):
        raise ValueError(
            "Stale nutrition observations; rebuild after changing LLM/prompt/table/mapping",
        )
    rows = list(read_jsonl(path))
    indexed = {row["image_id"]: row for row in rows}
    manifest = manifest_rows(config, "train")
    if len(indexed) != len(rows) or set(indexed) != {r["image_id"] for r in manifest}:
        raise ValueError("Nutrition cache must cover the training manifest exactly with unique IDs")
    root = image_root(config)
    result = []
    for row in manifest:
        entry = indexed[row["image_id"]]
        if entry["image_sha256"] != sha256_file(root / row["image_path"]):
            raise ValueError(f"Stale nutrition image: {row['image_id']}")
        result.append(
            {**entry, "label": row["label"], "group": str(row[config["nutrition"]["group_field"]])},
        )
    return result


def prepare_cache(config, split, kind="visual"):
    """分开生成视觉或语义缓存；语义缓存仅允许使用训练集。"""
    if kind == "nutrition" and split != "train":
        raise ValueError("Nutrition priors use the legal training split only")

    if kind == "visual":
        from .weight import feature_signature, visual_features

        path = Path(config["cache"][f"{split}_path"])
        signature = feature_signature(config)
    else:
        from .harness import run_agent

        path = Path(config["cache"]["nutrition_train_path"])
        signature = nutrition_signature(config)
    metadata_path = path.with_suffix(".meta.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["feature_signature"] != signature or metadata["kind"] != kind:
            raise ValueError("Cache inputs changed; choose a fresh cache path")
        records = list(read_jsonl(path))
        cached = {r["image_id"]: r for r in records}
        if len(cached) != len(records):
            raise ValueError("Duplicate image IDs in cache")
    rows = manifest_rows(config, split)
    if set(cached) - {r["image_id"] for r in rows}:
        raise ValueError("Cache contains IDs outside the manifest; use a fresh cache")
    metadata_path.write_text(
        json.dumps({"kind": kind, "split": split, "feature_signature": signature}, indent=2),
        encoding="utf-8",
    )
    root = image_root(config)
    with path.open("a", encoding="utf-8") as stream:
        for row in tqdm(rows, desc=f"{kind} cache {split}"):
            image_path = root / row["image_path"]
            image_hash = sha256_file(image_path)
            if row["image_id"] in cached:
                if cached[row["image_id"]]["image_sha256"] != image_hash:
                    raise ValueError("Image changed under an existing cache ID")
                continue
            record = {"image_id": row["image_id"], "image_sha256": image_hash}
            if kind == "visual":
                record["visual"] = visual_features(image_path, config)
            else:
                result = run_agent(image_path, config, observations_only=True)
                if result["status"] != "success":
                    raise RuntimeError(f"{row['image_id']}: {result['error']}")
                record["observation"] = result["observation"]
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    print(f"Saved {len(rows)} {kind} records to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Defaults to weight.yaml for visual, nutrition_agent.yaml for nutrition",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--kind", choices=["visual", "nutrition"], default="visual")
    args = parser.parse_args()
    config_path = args.config or (
        "config/optional/weight.yaml"
        if args.kind == "visual"
        else "config/optional/nutrition_agent.yaml"
    )
    prepare_cache(load_yaml(config_path), args.split, args.kind)
