"""仅从 RGB 图片提取冻结 DINOv2 特征，由 Ridge 预测整餐重量。"""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from utils.config import load_yaml
from utils.files import sha256_file
from utils.jsonl import read_jsonl, write_jsonl

from .data import image_root, manifest_rows

FEATURE_VERSION = "dinov2-small-cls-patch-mean-v2"
FEATURE_DIM = 768


@lru_cache(maxsize=1)
def encoder(path, device):
    """仅加载本地 DINOv2-Small，保持视觉编码器冻结并关闭训练模式。"""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
    if model.config.model_type != "dinov2" or model.config.hidden_size != 384:
        raise ValueError("Use DINOv2-Small without register tokens (384 dimensions per token)")
    model.requires_grad_(False)
    return AutoImageProcessor.from_pretrained(path, local_files_only=True), model


def visual_features(image_path, config):
    """拼接 CLS 和块词元均值，返回 768 维纯视觉特征。"""
    import torch
    from PIL import Image

    processor, model = encoder(config["weight"]["encoder_path"], config["weight"]["device"])
    with Image.open(image_path) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(
            next(model.parameters()).device,
        )
    with torch.inference_mode():
        tokens = model(**inputs).last_hidden_state
        features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
    return features[0].cpu().float().tolist()


@lru_cache(maxsize=2)
def encoder_fingerprint(path):
    root = Path(path)
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}
    )
    if not files:
        raise FileNotFoundError(f"Place DINOv2-Small model and processor at {root}")
    return {str(p.relative_to(root)): sha256_file(p) for p in files}


def feature_signature(config):
    """由视觉模型、处理器和特征定义决定缓存版本，与 LLM 无关。"""
    payload = {
        "encoder": encoder_fingerprint(config["weight"]["encoder_path"]),
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=2)
def load_predictor(path):
    import joblib

    state = joblib.load(path)
    if state.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Incompatible weight artifact; train the independent v2 Ridge predictor")
    return state


def predictor(config):
    """加载独立重量模型，并确认视觉特征版本一致。"""
    state = load_predictor(config["weight"]["artifact_path"])
    if state["feature_signature"] != feature_signature(config):
        raise ValueError("Visual encoder/processor changed; rebuild visual cache and retrain Ridge")
    return state


def predict_features(state, visual):
    values = np.asarray(visual, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or not np.isfinite(values).all():
        raise ValueError("Expected finite 768-dimensional visual feature vectors")
    predictions = state["model"].predict(values)
    if not np.isfinite(predictions).all():
        raise ValueError("Weight predictor returned non-finite values")
    return np.maximum(state["min_weight_g"], predictions)


def estimate_weight(image_path, config):
    """仅接收图片和运行配置，返回整餐总重量，不读取语义或营养标签。"""
    state = predictor(config)
    mass = predict_features(state, [visual_features(image_path, config)])[0]
    return {"total_weight_g": float(mass)}


def load_cache(config, split):
    """将视觉缓存与指定划分对齐，检查特征、图片指纹和真实重量。"""
    path = Path(config["cache"][f"{split}_path"])
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if metadata["feature_signature"] != feature_signature(config):
        raise ValueError("Stale visual cache: rebuild after changing the encoder/processor")
    manifest = manifest_rows(config, split)
    cached = list(read_jsonl(path))
    by_id = {r["image_id"]: r for r in cached}
    if len(by_id) != len(cached) or set(by_id) != {r["image_id"] for r in manifest}:
        raise ValueError(f"{split} visual cache must cover the manifest exactly with unique IDs")
    root = image_root(config)
    result = []
    for row in manifest:
        entry = by_id[row["image_id"]]
        if entry["image_sha256"] != sha256_file(root / row["image_path"]):
            raise ValueError(f"Stale image features: {row['image_id']}")
        target = row["label"]["total_weight_g"]
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not np.isfinite(target)
            or target < 0
        ):
            raise ValueError(f"Invalid weight label: {row['image_id']}")

        if len(entry["visual"]) != FEATURE_DIM or not np.isfinite(entry["visual"]).all():
            raise ValueError(
                "Rebuild the visual cache with 768-dimensional CLS + patch mean features",
            )
        result.append(
            {
                **entry,
                "target_weight_g": float(target),
                "group": str(row[config["weight"]["group_field"]]),
            },
        )
    return result


def ridge_pipeline(alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="cholesky"))


def train(config):
    """用分组交叉验证选择 Ridge 正则强度，再用完整训练集拟合。"""
    import joblib
    from sklearn.model_selection import GroupKFold

    rows = load_cache(config, "train")
    settings = config["weight"]
    groups = [r["group"] for r in rows]
    seen = {}
    for row in rows:
        key = row["image_sha256"]
        if key in seen and seen[key] != row["group"]:
            raise ValueError("Duplicate image appears in different CV groups")
        seen[key] = row["group"]
    development = manifest_rows(config, "val")
    root = image_root(config)
    if set(groups) & {str(r[settings["group_field"]]) for r in development}:
        raise ValueError("Training and validation dish groups overlap")

    if set(seen) & {sha256_file(root / r["image_path"]) for r in development}:
        raise ValueError("Training and validation images overlap")
    visual = np.asarray([r["visual"] for r in rows], dtype=np.float64)
    target = np.asarray([r["target_weight_g"] for r in rows])
    folds = list(GroupKFold(n_splits=settings["cv_splits"]).split(visual, groups=groups))
    baseline = np.empty(len(rows))
    for train_index, holdout_index in folds:
        baseline[holdout_index] = target[train_index].mean()
    trials, best, best_oof = [], None, None
    for alpha in settings["ridge_alphas"]:
        if alpha <= 0:
            raise ValueError("Ridge alpha must be positive")
        oof = np.empty(len(rows))
        for train_index, holdout_index in folds:
            # 标准化器只在当前训练折拟合，避免验证折信息泄漏。
            model = ridge_pipeline(alpha)
            model.fit(visual[train_index], target[train_index])
            oof[holdout_index] = np.maximum(
                settings["min_weight_g"],
                model.predict(visual[holdout_index]),
            )
        trial = {"alpha": alpha, "mse": float(np.mean((oof - target) ** 2))}
        trials.append(trial)
        print(json.dumps(trial), flush=True)
        if best is None or trial["mse"] < best["mse"]:
            best, best_oof = trial, oof.copy()
    if best is None:
        raise ValueError("Configure at least one Ridge alpha")
    model = ridge_pipeline(best["alpha"])
    model.fit(visual, target)
    state = {
        "model": model,
        "feature_signature": feature_signature(config),
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "min_weight_g": settings["min_weight_g"],
        "weight_config": settings,
        "train_mean_weight_g": float(target.mean()),
        "training_manifest_sha256": sha256_file(Path(config["data_dir"]) / "train.jsonl"),
    }
    output = Path(settings["artifact_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, output)
    load_predictor.cache_clear()
    report = {
        "samples": len(rows),
        "groups": len(set(groups)),
        "feature_dim": FEATURE_DIM,
        "mean_baseline_mse": float(np.mean((baseline - target) ** 2)),
        "selected": best,
        "trials": trials,
        "note": "OOF selects alpha; report independent validation performance separately.",
    }
    write_jsonl(
        (
            {
                "image_id": r["image_id"],
                "group": r["group"],
                "target_weight_g": float(y),
                "predicted_weight_g": float(p),
            }
            for r, y, p in zip(rows, target, best_oof)
        ),
        output.with_name("oof_predictions.jsonl"),
    )
    output.with_name("cv_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_weight(config, split="val"):
    """在独立验证或测试划分上评估重量，并报告训练均值基线。"""
    if split not in ("val", "test"):
        raise ValueError("Evaluate only on val or frozen test split")
    training, rows = load_cache(config, "train"), load_cache(config, split)
    if {r["group"] for r in training} & {r["group"] for r in rows} or {
        r["image_sha256"] for r in training
    } & {r["image_sha256"] for r in rows}:
        raise ValueError(f"Training and {split} dishes/images overlap")
    state = predictor(config)
    if state["training_manifest_sha256"] != sha256_file(Path(config["data_dir"]) / "train.jsonl"):
        raise ValueError("Training manifest changed since fitting Ridge")
    predictions = predict_features(state, [r["visual"] for r in rows])
    target = np.asarray([r["target_weight_g"] for r in rows])
    result = {
        "split": split,
        "samples": len(rows),
        "mse": float(np.mean((predictions - target) ** 2)),
        "mae": float(np.mean(np.abs(predictions - target))),
        "mean_baseline_mse": float(np.mean((state["train_mean_weight_g"] - target) ** 2)),
    }
    output = Path(config["weight"]["artifact_path"])
    output.with_name(f"{split}_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    write_jsonl(
        (
            {"image_id": r["image_id"], "target_weight_g": float(y), "predicted_weight_g": float(p)}
            for r, y, p in zip(rows, target, predictions)
        ),
        output.with_name(f"{split}_predictions.jsonl"),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/optional/weight.yaml")
    parser.add_argument("--action", choices=["train", "evaluate", "predict"], required=True)
    parser.add_argument("--image")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.action == "train":
        result = train(config)
    elif args.action == "evaluate":
        result = evaluate_weight(config, args.split)
    else:
        if not args.image:
            parser.error("--image is required for predict")
        result = estimate_weight(args.image, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
