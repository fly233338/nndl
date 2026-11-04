"""用固定训练语料统计计算 CIDEr-D 奖励，生成结果在训练时实时评分。"""

import argparse
import json
import math
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import read_jsonl

CACHE_VERSION = 1


def ptb_tokenize(captions):
    """批量分词并保持每张图片的参考描述分组。"""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    payload = {i: [{"caption": caption} for caption in group] for i, group in enumerate(captions)}
    result = PTBTokenizer().tokenize(payload)
    if any(len(result.get(i, [])) != len(group) for i, group in enumerate(captions)):
        raise RuntimeError("PTB tokenization failed; check Java and pycocoevalcap")
    return [result[i] for i in range(len(captions))]


def prepare(config):
    """仅缓存训练参考文本和文档频率，不缓存模型生成结果或奖励。"""
    from pycocoevalcap.cider.cider_scorer import precook

    source = Path(config["data_dir"]) / "train.jsonl"
    rows = list(read_jsonl(source))
    if len(rows) < 2 or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("CIDEr-D document frequencies require >=2 unique training images")

    if any(
        not r["captions"] or any(not isinstance(c, str) or not c.strip() for c in r["captions"])
        for r in rows
    ):
        raise ValueError("Every training image needs nonempty references")
    references = ptb_tokenize([r["captions"] for r in rows])
    df = Counter()
    n = config["reward"]["n"]
    if n != 4 or config["reward"]["sigma"] <= 0:
        raise ValueError("CIDEr-D requires n=4 and a positive sigma")
    for group in references:
        df.update({gram for caption in group for gram in precook(caption, n)})
    if not df:
        raise ValueError("No training ngrams; check reference captions and tokenizer")
    data = {
        "version": CACHE_VERSION,
        "train_manifest_sha256": sha256_file(source),
        "pycocoevalcap": version("pycocoevalcap"),
        "num_documents": len(rows),
        "n": n,
        "document_frequency": [[list(gram), count] for gram, count in sorted(df.items())],
        "references": [
            {"image_id": row["image_id"], "captions": group} for row, group in zip(rows, references)
        ],
    }
    target = Path(config["reward_cache"])
    if target.resolve() == source.resolve():
        raise ValueError("Reward cache must not overwrite the training manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Saved fixed training DF: {len(rows)} images, {len(df)} ngrams -> {target}")


class CiderDReward:
    """固定训练语料的 IDF，对当前策略生成的描述实时计算奖励。"""

    def __init__(self, config):
        data = json.loads(Path(config["reward_cache"]).read_text(encoding="utf-8"))
        if data["version"] != CACHE_VERSION or data["train_manifest_sha256"] != sha256_file(
            Path(config["data_dir"]) / "train.jsonl",
        ):
            raise ValueError(
                "Stale reward cache; run caption_rl.reward on the current training split",
            )

        if data["n"] != config["reward"]["n"] or data["pycocoevalcap"] != version("pycocoevalcap"):
            raise ValueError("Reward tokenization/ngram configuration changed; rebuild cache")
        self.n, self.sigma = data["n"], config["reward"]["sigma"]
        if self.n != 4 or self.sigma <= 0:
            raise ValueError("CIDEr-D requires n=4 and a positive sigma")
        self.log_documents = math.log(data["num_documents"])
        self.df = {tuple(gram): count for gram, count in data["document_frequency"]}
        self.references = {
            r["image_id"]: [self.vector(c) for c in r["captions"]] for r in data["references"]
        }

    def vector(self, caption):
        """构造各阶 n-gram 的 TF-IDF 权重、范数及长度。"""
        from pycocoevalcap.cider.cider_scorer import precook

        counts = precook(caption, self.n)
        weights = {
            gram: count * (self.log_documents - math.log(max(1, self.df.get(gram, 0))))
            for gram, count in counts.items()
        }
        norms = []
        for order in range(1, self.n + 1):
            squared_norm = sum(
                value * value for gram, value in weights.items() if len(gram) == order
            )
            norms.append(math.sqrt(squared_norm))
        # 与 pycocoevalcap 一致，使用二元组数量作为长度。
        length = sum(count for gram, count in counts.items() if len(gram) == 2)
        return weights, norms, length

    def score(self, image_id, caption):
        hypothesis, hyp_norm, hyp_length = self.vector(caption)
        references = self.references[image_id]
        total = 0.0
        for reference, ref_norm, ref_length in references:
            penalty = math.exp(-((hyp_length - ref_length) ** 2) / (2 * self.sigma**2))
            for order in range(1, self.n + 1):
                if hyp_norm[order - 1] and ref_norm[order - 1]:
                    dot = sum(
                        min(value, reference.get(gram, 0.0)) * reference.get(gram, 0.0)
                        for gram, value in hypothesis.items()
                        if len(gram) == order
                    )
                    total += penalty * dot / (hyp_norm[order - 1] * ref_norm[order - 1])
        return 10.0 * total / (self.n * len(references))

    def rewards(self, ids, sampled, greedy):
        """同时计算当前采样描述和贪心基线的奖励，按输入顺序返回。"""
        tokenized = ptb_tokenize([[text] for text in sampled + greedy])
        scores = np.asarray(
            [self.score(image_id, group[0]) for image_id, group in zip(ids + ids, tokenized)],
            dtype=np.float32,
        )
        return scores[: len(ids)], scores[len(ids) :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/caption_rl.yaml")
    args = parser.parse_args()
    prepare(load_yaml(args.config))
