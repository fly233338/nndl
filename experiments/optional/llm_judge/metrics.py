"""使用整份语料的统计量导出逐图 CIDEr-D，供相关性分析使用。"""

import argparse
from pathlib import Path

from utils.jsonl import read_jsonl, write_jsonl

from .analyze import indexed


def export_scores(predictions, output):
    """用同一语料级评分器导出逐图分数，避免逐张重算文档频率。"""
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    if Path(predictions).resolve() == Path(output).resolve():
        raise ValueError("Output must not overwrite caption predictions")
    rows = list(read_jsonl(predictions))
    indexed(rows)
    if not rows:
        raise ValueError("No caption predictions")
    references = {
        i: [{"caption": text} for text in row["references"]] for i, row in enumerate(rows)
    }
    candidates = {i: [{"caption": row["prediction"]}] for i, row in enumerate(rows)}
    tokenizer = PTBTokenizer()
    mean, scores = Cider().compute_score(
        tokenizer.tokenize(references),
        tokenizer.tokenize(candidates),
    )
    write_jsonl(
        (
            {"image_id": row["image_id"], "metrics": {"CIDEr-D": float(score)}}
            for row, score in zip(rows, scores)
        ),
        output,
    )
    print(f"Saved {len(rows)} per-image scores; corpus CIDEr-D={mean:.6f} (raw scorer scale)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_scores(args.predictions, args.output)
