"""执行单图或清单推理，真实标签仅用于结果保存和后续评测。"""

import argparse
import json
import os
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from tqdm import tqdm

from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import read_jsonl

from .harness import MODES, run_agent


def infer(config_path, mode="agent", image=None, manifest=None, output=None, limit=None):
    """执行单图或批量推理，分别保存预测、工具轨迹和配置指纹。"""
    if (image is None) == (manifest is None):
        raise ValueError("Supply exactly one of image or manifest")

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    if mode not in MODES:
        raise ValueError(f"Unknown inference mode: {mode}")
    config = load_yaml(config_path)
    if not os.environ.get(config["api"]["base_url_env"]):
        raise ValueError(
            f"Set {config['api']['base_url_env']} to your Qwen OpenAI-compatible endpoint",
        )
    required = []
    if mode != "direct":
        required.extend(
            config["resources"][key] for key in ("table_path", "metadata_path", "mapping_path")
        )
        if config["nutrition"]["use_calibration"]:
            required.append(config["resources"]["calibration_path"])
    if mode == "agent":
        required.append(config["weight"]["artifact_path"])
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepare the nutrition database, calibration and weight regressor first. Missing: "
            + ", ".join(missing),
        )
    output = Path(output) if output else Path(config["output_dir"]) / mode / "predictions.jsonl"
    trace_path = output.with_name(output.stem + "_traces.jsonl")
    config_output = output.with_name(output.stem + "_config.json")
    if Path(image or manifest).resolve() in {
        path.resolve() for path in (output, trace_path, config_output)
    }:
        raise ValueError("Output files must not overwrite the input image or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "config": config,
        "mode": mode,
        "limit": limit,
        "input": str(image or manifest),
        "input_sha256": sha256_file(image or manifest),
    }
    snapshot["resource_sha256"] = {str(path): sha256_file(path) for path in required}
    snapshot["resolved_model"] = (
        os.environ.get(config["api"].get("model_env", "")) or config["api"]["model"]
    )
    if mode == "agent":
        from .weight import predictor

        snapshot["weight_feature_signature"] = predictor(config)["feature_signature"]
    endpoint = urlsplit(os.environ.get(config["api"]["base_url_env"], ""))
    host = endpoint.hostname or ""
    if endpoint.port:
        host += f":{endpoint.port}"
    snapshot["api_endpoint"] = (
        urlunsplit((endpoint.scheme, host, endpoint.path, "", "")) if host else None
    )
    config_output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    if image is not None:
        rows = [{"image_id": Path(image).stem, "image_path": str(Path(image).resolve())}]
        image_root = Path()
    else:
        rows = read_jsonl(manifest)
        image_root_value = config["image_root"]
        if not image_root_value:
            info = Path(config["data_dir"]) / "dataset_info.json"
            image_root_value = json.loads(info.read_text(encoding="utf-8"))["image_root"]
        image_root = Path(image_root_value)
    if limit is not None:
        rows = islice(rows, limit)
    samples = successes = 0
    with (
        output.open("w", encoding="utf-8") as predictions,
        trace_path.open("w", encoding="utf-8") as traces,
    ):
        for row in tqdm(rows, desc=mode, unit="img"):
            result = run_agent(str(image_root / row["image_path"]), config, mode)
            trace = result.pop("trace")
            record = {"image_id": row["image_id"], **result}
            if "label" in row:
                record["label"] = row["label"]
            predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            traces.write(
                json.dumps(
                    {"image_id": row["image_id"], **trace},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
            )
            predictions.flush()
            traces.flush()
            samples += 1
            successes += result["status"] == "success"
    summary = {
        "samples": samples,
        "successes": successes,
        "failures": samples - successes,
        "predictions": str(output),
        "traces": str(trace_path),
        "config": str(config_output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/nutrition_agent.yaml")
    parser.add_argument("--mode", choices=MODES, default="agent")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    infer(args.config, args.mode, args.image, args.manifest, args.output, args.limit)
