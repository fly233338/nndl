"""结合 RGB 图片和参考描述，对任务一预测逐图评分。"""

import argparse
import base64
import json
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import read_jsonl

from .analyze import indexed, summary
from .client import SCORE_FIELDS, check_api, connect, parse_scores, request_scores


def image_content(path, max_size):
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((max_size, max_size))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}


def evaluate(
    config_path,
    predictions_path,
    manifest=None,
    image_root=None,
    output=None,
    limit=None,
):
    """逐图调用 Judge，保留失败记录以统计完整清单的评分覆盖率。"""
    config = load_yaml(config_path)
    manifest = Path(manifest or config["manifest"])
    root = Path(image_root or config["image_root"])
    target = Path(
        output or Path(config["output_dir"]) / Path(predictions_path).stem / "judgments.jsonl",
    )
    summary_path = target.with_name(target.stem + "_summary.json")
    config_output = target.with_name(target.stem + "_config.json")
    if {p.resolve() for p in (target, summary_path, config_output)} & {
        manifest.resolve(),
        Path(predictions_path).resolve(),
        Path(config_path).resolve(),
    }:
        raise ValueError("Outputs must not overwrite inputs")

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest_rows = list(read_jsonl(manifest))
    cohort = indexed(manifest_rows)
    predictions = indexed(read_jsonl(predictions_path))
    if not cohort or set(predictions) - set(cohort):
        raise ValueError("Manifest must be nonempty and contain every prediction image ID")
    rows = manifest_rows[:limit] if limit else manifest_rows
    client, model = connect(config)
    records = []
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_text(
            json.dumps(
                {
                    "config": config,
                    "resolved_model": model,
                    "limit": limit,
                    "manifest_sha256": sha256_file(manifest),
                    "predictions_sha256": sha256_file(predictions_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with target.open("w", encoding="utf-8") as stream:
            for row in tqdm(rows, desc="LLM judge", unit="img"):
                started = time.perf_counter()
                record = {
                    "image_id": row["image_id"],
                    "model": model,
                    "status": "failed",
                    "raw_output": "",
                    "usage": None,
                }
                try:
                    prediction = predictions.get(row["image_id"], {}).get("prediction")
                    if not isinstance(prediction, str) or not prediction.strip():
                        raise ValueError("Missing or empty caption prediction")
                    payload = {"candidate_caption": prediction}
                    if config["judge"]["include_references"]:
                        references = row["captions"]
                        if not isinstance(references, list) or any(
                            not isinstance(r, str) for r in references
                        ):
                            raise ValueError("Manifest captions must be a list of references")
                        payload["reference_captions"] = references
                    image_path = root / row["image_path"]
                    content = [
                        image_content(image_path, config["judge"]["image_max_size"]),
                        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    ]
                    record["prediction"] = prediction
                    record["image_sha256"] = sha256_file(image_path)
                    response = request_scores(client, model, config, content)
                    record["raw_output"] = response.choices[0].message.content or ""
                    record["finish_reason"] = response.choices[0].finish_reason
                    record["usage"] = response.usage.model_dump() if response.usage else None
                    scores = parse_scores(record["raw_output"])
                    record.update(
                        status="success",
                        scores={k: scores[k] for k in SCORE_FIELDS},
                        overall=sum(scores[k] for k in SCORE_FIELDS) / len(SCORE_FIELDS),
                        reason=scores["reason"],
                    )
                # 逐图错误需要保留，才能统计完整清单的覆盖率。
                except Exception as exc:  # noqa: BLE001
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["seconds"] = time.perf_counter() - started
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        result = {**summary(records), "model": model, "judgments": str(target)}
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/llm_judge.yaml")
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--predictions")
    parser.add_argument("--manifest")
    parser.add_argument("--image-root")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.check_api:
        check_api(args.config)
    else:
        if not args.predictions:
            parser.error("--predictions is required unless --check-api is used")
        result = evaluate(
            args.config,
            args.predictions,
            args.manifest,
            args.image_root,
            args.output,
            args.limit,
        )
        raise SystemExit(1 if result["failures"] else 0)
