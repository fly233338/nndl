"""在相同图片、解码设置和指标下比较 CE 与 SCST。"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import write_jsonl
from utils.metrics import evaluate_caption_records, metric_metadata

from .data import disjoint, load_model, make_loader, manifest, resolve_device


@torch.no_grad()
def caption_records(model, vocab, loader, rows, device, max_length, beam_size):
    """按指定解码方式生成描述，保留每张图片的全部参考。"""
    model.eval()
    references = {r["image_id"]: r["captions"] for r in rows}
    records = []
    for images, ids in tqdm(loader, desc="decode", leave=False):
        features = model.encoder(images.to(device))
        sequences = model.decoder.generate(
            features,
            vocab.token_to_id["<bos>"],
            vocab.token_to_id["<eos>"],
            max_length,
            beam_size,
        )
        for image_id, sequence in zip(ids, sequences):
            records.append(
                {
                    "image_id": image_id,
                    "references": references[image_id],
                    "prediction": vocab.decode(sequence.tolist()),
                },
            )
    return records


def compare(config_path, split="test", scst_checkpoint=None):
    """重新评测配对的 CE 和 SCST 检查点，确保数据与解码设置一致。"""
    config = load_yaml(config_path)
    if split not in ("val", "test"):
        raise ValueError("Compare on val or frozen test split only")
    source = Path(config["checkpoint"])
    tuned = Path(scst_checkpoint or Path(config["output_dir"]) / "best.pt")
    state = torch.load(tuned, map_location="cpu", weights_only=False)
    if state.get("training_method") != "scst_cider_d":
        raise ValueError("SCST comparison requires a trained SCST checkpoint")

    if state["ce_checkpoint_sha256"] != sha256_file(source):
        raise ValueError("CE checkpoint must be the exact checkpoint used to initialize SCST")
    train_path = Path(config["data_dir"]) / "train.jsonl"
    val_path = Path(config["data_dir"]) / "val.jsonl"
    if state["train_manifest_sha256"] != sha256_file(train_path) or state[
        "val_manifest_sha256"
    ] != sha256_file(val_path):
        raise ValueError("Training/validation manifests changed after SCST model selection")
    rows = manifest(config, split)
    disjoint(manifest(config, "train"), rows)
    if split == "test":
        disjoint(manifest(config, "val"), rows)
    device = resolve_device(config)
    root = Path(config["output_dir"]) / "comparison" / split
    results = {}
    baseline_spec = None
    for name, checkpoint in (("ce", source), ("scst", tuned)):
        model, vocab, _, model_config = load_model(checkpoint, config, device)
        spec = {
            "vocab": vocab.token_to_id,
            "image_size": model_config.get("image_size", 224),
            "max_length": model_config.get("max_length", 64),
        }
        if baseline_spec is None:
            baseline_spec = spec
        elif spec != baseline_spec:
            raise ValueError("CE/SCST vocabulary, image size and maximum length must match")
        loader = make_loader(rows, config, spec["image_size"])
        results[name] = {}
        for decode in config["evaluation"]["decodes"]:
            if decode not in ("greedy", "beam"):
                raise ValueError("Evaluation decode must be greedy or beam")
            beam_size = 1 if decode == "greedy" else config["beam_size"]
            output = root / name / decode
            records = caption_records(
                model,
                vocab,
                loader,
                rows,
                device,
                spec["max_length"],
                beam_size,
            )
            write_jsonl(records, output / "predictions.jsonl")
            metrics = evaluate_caption_records(records, metrics=config["evaluation"]["metrics"])
            metrics.update(
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256_file(checkpoint),
                decode=decode,
                beam_size=beam_size,
                image_size=spec["image_size"],
                max_length=spec["max_length"],
                manifest_sha256=sha256_file(Path(config["data_dir"]) / f"{split}.jsonl"),
                metric_metadata=metric_metadata(),
            )
            (output / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            results[name][decode] = metrics
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    deltas = {
        decode: {
            metric: results["scst"][decode][metric] - results["ce"][decode][metric]
            for metric in config["evaluation"]["metrics"]
        }
        for decode in config["evaluation"]["decodes"]
    }
    report = {
        "split": split,
        "samples": len(rows),
        "scst_selected_epoch": state["epoch"],
        "results": results,
        "delta_scst_minus_ce": deltas,
        "note": (
            "Both checkpoints reevaluated on identical images and decoding; "
            "positive delta means improvement."
        ),
    }
    (root / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"comparison": str(root / "comparison.json"), "delta_scst_minus_ce": deltas},
            indent=2,
        ),
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/caption_rl.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--scst-checkpoint")
    args = parser.parse_args()
    compare(args.config, args.split, args.scst_checkpoint)
