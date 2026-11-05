"""仅用训练集统计名称词表，提供候选检索，不自动替换模型识别结果。"""

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from utils.config import load_yaml
from utils.files import sha256_file

from .data import manifest_rows


def prepare_food_names(config):
    """每张训练图片对同一名称计数一次，不保存图片级答案或营养标签。"""
    rows = manifest_rows(config, "train")
    counts = Counter()
    for row in rows:
        names = {item["name"].strip().lower() for item in row["label"]["items"]}
        if "" in names:
            raise ValueError(f"Empty training food name: {row['image_id']}")
        counts.update(names)

    if not counts:
        raise ValueError("The training manifest contains no food names")

    vocabulary = {
        "source_split": "train",
        "training_manifest_sha256": sha256_file(Path(config["data_dir"]) / "train.jsonl"),
        "training_images": len(rows),
        "unique_names": len(counts),
        "names": [
            {"name": name, "training_image_count": count}
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
    path = Path(config["food_names"]["vocabulary_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "training_images": len(rows),
                "unique_names": len(counts),
                "vocabulary": str(path),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return vocabulary


class FoodNameLookup:
    """用固定训练词表检索近似名称；相似度只用于召回，是否同义由模型判断。"""

    def __init__(self, config):
        self.settings = config["food_names"]
        vocabulary = json.loads(
            Path(self.settings["vocabulary_path"]).read_text(encoding="utf-8"),
        )
        if vocabulary["source_split"] != "train":
            raise ValueError("Food-name vocabulary must come from the training split")
        self.names = vocabulary["names"]
        self.aliases = load_yaml(config["resources"]["mapping_path"])["aliases"]

    def lookup(self, name, category=None):
        """返回有限候选，允许空结果；训练频次不是当前图片存在该食物的证据。"""
        queries = [name.strip().lower()]
        if category:
            queries.append(category.strip().lower())
        queries = {self.aliases.get(query, query).strip().lower() for query in queries}
        candidates = []
        for entry in self.names:
            candidate_name = entry["name"]
            candidate_words = set(re.findall(r"\w+", candidate_name))
            scores = []
            for query in queries:
                query_words = set(re.findall(r"\w+", query))
                overlap = (
                    2
                    * len(query_words & candidate_words)
                    / max(
                        1,
                        len(query_words) + len(candidate_words),
                    )
                )
                scores.append((overlap + SequenceMatcher(None, query, candidate_name).ratio()) / 2)
            score = max(scores)
            if score >= self.settings["min_match_score"]:
                candidates.append({**entry, "match_score": round(score, 4)})
        candidates.sort(
            key=lambda entry: (
                -entry["match_score"],
                -entry["training_image_count"],
                entry["name"],
            ),
        )
        return candidates[: self.settings["top_k"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/nutrition_agent_names.yaml")
    parser.add_argument("--query", help="Query an existing vocabulary instead of rebuilding it")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.query is None:
        prepare_food_names(config)
    else:
        print(json.dumps(FoodNameLookup(config).lookup(args.query), ensure_ascii=False, indent=2))
