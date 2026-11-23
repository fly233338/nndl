"""FoodLens 统一入口：任务 / 实验组 / 实验，默认只训练与验证。"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from utils.config import load_yaml, _merge


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = {
    "task1": ["config/task1_captioning/baseline.yaml", "config/task1_captioning/ablation.yaml"],
    "task2": ["config/task2_meal_json/ablation.yaml"],
}
MODULES = {
    "caption_baseline": "experiments.task1_captioning.train",
    "caption_ablation": "experiments.task1_captioning.strong_captioner",
    "meal": "experiments.task2_meal_json.strong_meal",
}


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).is_file() else default


def catalog(task, config_path=None):
    entries, groups, aliases = {}, {}, {}
    for filename in [config_path] if config_path else DEFAULT_CONFIGS[task]:
        config = load_yaml(filename)
        backend = config["runner"]["backend"]
        allowed = ("caption_baseline", "caption_ablation") if task == "task1" else ("meal",)
        if backend not in allowed:
            raise ValueError(f"Configuration backend {backend} does not belong to {task}")
        for name, overrides in config["experiments"].items():
            if name in entries:
                raise ValueError(f"Duplicate experiment: {name}")
            entries[name] = {"name": name, "description": config["descriptions"][name],
                             "backend": backend, "config_path": filename,
                             "config": _merge(config, overrides), "groups": []}
        for group, names in config["groups"].items():
            groups.setdefault(group, []).extend(names)
        aliases.update(config.get("aliases", {}))
    for group, names in groups.items():
        for name in names:
            if name not in entries:
                raise ValueError(f"Unknown experiment {name} in group {group}")
            entries[name]["groups"].append(group)
        groups[group] = list(dict.fromkeys(names))
    if any(target not in entries for target in aliases.values()):
        raise ValueError("Alias references an unknown experiment")
    return entries, groups, aliases


def run_directory(entry, seed, output_dir=None):
    return Path(output_dir or entry["config"]["output_dir"]) / entry["name"] / f"seed_{seed}"


def baseline_data(config, args):
    data_dir = Path(config["data_dir"])
    info = read_json(data_dir / "dataset_info.json", {})
    image_root = args.image_root or config.get("image_root") or info.get("image_root") or config.get("runner", {}).get("default_image_root")
    # 显式默认值放在 baseline 配置中；旧检查点可从 dataset_info 解析。
    if not image_root:
        raise ValueError("Baseline image root is unknown; set image_root in YAML or --image-root")
    return str(args.manifest or data_dir / f"{args.split}.jsonl"), str(image_root)


def checkpoint_backend(task, checkpoint, explicit=None):
    config = read_json(Path(checkpoint).parent / "run_config.json", {})
    backend = explicit or config.get("runner", {}).get("backend")
    if backend is None:
        if task == "task2":
            backend = "meal"
        elif "encoder" in config and "model" in config:
            backend = "caption_ablation"
        elif config:
            backend = "caption_baseline"
        else:
            raise ValueError("No run_config.json beside checkpoint; specify --backend caption_baseline or caption_ablation")
    allowed = ("caption_baseline", "caption_ablation") if task == "task1" else ("meal",)
    if backend not in allowed:
        raise ValueError(f"Checkpoint backend {backend} does not belong to {task}")
    return backend, config


def inference_command(args, backend, checkpoint, config):
    if backend == "caption_baseline":
        if args.model_path or args.accumulation or args.device:
            raise ValueError("Legacy baseline inference does not support --model-path/--accumulation/--device")
        module = "experiments.task1_captioning.evaluate" if args.action == "evaluate" else "experiments.task1_captioning.infer"
        command = [sys.executable, "-m", module, "--checkpoint", str(checkpoint)]
        if args.action == "evaluate":
            manifest, image_root = baseline_data(config, args)
            command += ["--manifest", manifest, "--image-root", image_root,
                        "--output", str(Path(checkpoint).parent / "evaluation" / args.split / "metrics.json")]
            if args.batch_size:
                command += ["--batch-size", str(args.batch_size)]
        else:
            if args.batch_size:
                raise ValueError("Baseline single-image prediction has no --batch-size")
            command += ["--image", args.image]
        return command
    command = [sys.executable, "-m", MODULES[backend], args.action, "--checkpoint", str(checkpoint)]
    if args.action == "evaluate":
        command += ["--split", args.split]
        for name in ("manifest", "image_root"):
            if getattr(args, name):
                command += ["--" + name.replace("_", "-"), getattr(args, name)]
        if args.batch_size:
            command += ["--batch-size", str(args.batch_size)]
    else:
        if args.batch_size:
            raise ValueError("Single-image prediction has no --batch-size")
        command += ["--image", args.image]
    for name in ("model_path", "device"):
        if getattr(args, name):
            command += ["--" + name.replace("_", "-"), getattr(args, name)]
    return command


def training_command(args, entry, seed, run):
    backend = entry["backend"]
    if backend == "caption_baseline":
        if args.resume or args.model_path or args.batch_size or args.accumulation or args.device:
            raise ValueError("Legacy baselines do not support unified resume/model-path/batch-size/accumulation/device overrides; edit baseline YAML")
        if entry["config"].get("resume") or entry["config"].get("init_checkpoint"):
            raise ValueError("Unified baseline runs must start fresh; historical initialization uses its original entrypoint")
        return [sys.executable, "-m", MODULES[backend], "--config", entry["config_path"],
                "--model", entry["name"], "--seed", str(seed), "--run-dir", str(run)]
    command = [sys.executable, "-m", MODULES[backend], "train", "--config", entry["config_path"],
               "--experiment", entry["name"], "--seed", str(seed)]
    explicit_resume = args.resume if args.resume and args.resume != "auto" else None
    resume = Path(explicit_resume) if explicit_resume else run / "last.pt"
    if explicit_resume or (args.resume and resume.is_file()):
        command += ["--resume", str(resume)]
    else:
        command += ["--output-dir", str(run.parent.parent)]
        if backend == "meal" and entry["config"]["backend"] == "siglip" and args.caption_checkpoint:
            command += ["--caption-checkpoint", args.caption_checkpoint]
    for name, flag in (("model_path", "--model-path"), ("batch_size", "--batch-size"),
                       ("accumulation", "--accumulation"), ("device", "--device")):
        if getattr(args, name) is not None:
            command += [flag, str(getattr(args, name))]
    return command


def flatten_numbers(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            result.update(flatten_numbers(item, name + "_"))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def summary_record(entry, seed, run):
    config = read_json(run / "run_config.json", entry["config"])
    history = read_json(run / "training_history.json", [])
    metadata = read_json(run / "run_metadata.json", {})
    key = entry["config"]["runner"]["selection_metric"]
    best = {}
    if history:
        if entry["backend"] == "caption_baseline":
            # 旧基线使用 min_delta 更新 best，不能把 history 的绝对最大值冒充已保存模型。
            maximum = float("-inf")
            for record in history:
                if record[key] > maximum + config.get("min_delta", 0):
                    maximum, best = record[key], record
        else:
            best = max(history, key=lambda row: row[key])
    row = {"experiment": entry["name"], "description": entry["description"], "groups": ",".join(entry["groups"]),
           "recipe": entry["config"]["runner"]["recipe"], "backend": entry["backend"], "seed": seed, "run_dir": str(run),
           "status": "complete" if (run / "completed.json").is_file() and (run / "best.pt").is_file() else "incomplete" if run.exists() else "not_run",
           "selection_metric": key, "selection_value": best.get(key), "best_epoch": best.get("epoch"),
           "trainable_params": metadata.get("trainable_params", best.get("trainable_params")),
           "total_params": metadata.get("total_params"),
           "total_epoch_seconds": sum(record["seconds"] for record in history) if history else None,
           "best_val_inference_seconds": best.get("inference_seconds"),
           "peak_memory_allocated": max((record.get("peak_memory_allocated") or 0 for record in history), default=0) or None}
    row.update(flatten_numbers(best.get("val_metrics", {}), "val_"))
    for path in sorted((run / "evaluation").rglob("*.json")):
        relative = path.relative_to(run / "evaluation").with_suffix("")
        prefix = "eval_" + "_".join(relative.parts) + "_"
        row.update(flatten_numbers(read_json(path, {}), prefix))
    return row


def summarize(jobs, task, dry_run=False):
    grouped, seen = {}, set()
    for entry, seed, run in jobs:
        identity = run.resolve()
        if identity in seen:
            continue
        seen.add(identity)
        grouped.setdefault(run.parent.parent, []).append(summary_record(entry, seed, run))
    for root, rows in grouped.items():
        if dry_run:
            print(f"Summary -> {root / 'summary.csv'} ; {root / 'summary.md'} ({len(rows)} unique runs)")
            continue
        root.mkdir(parents=True, exist_ok=True)
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        groups = [["experiment", "groups", "recipe", "seed", "status", "best_epoch", "selection_metric", "selection_value"],
                  ["experiment", "seed", "trainable_params", "total_params", "total_epoch_seconds", "best_val_inference_seconds", "peak_memory_allocated"]]
        metric_columns = [key for key in columns if key.startswith(("val_", "eval_"))]
        groups += [["experiment", "seed"] + metric_columns[i:i + 5] for i in range(0, len(metric_columns), 5)]
        lines = [f"# {task} 实验汇总", "", "按验证集指标选型；不同 recipe 分开解释。未记录数据不填零。"]
        for fields in groups:
            lines += ["", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(key)) if row.get(key) is not None else "未运行/未记录" for key in fields) + " |")
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Summary: {root / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "train", "evaluate", "predict", "summarize"])
    parser.add_argument("task", choices=list(DEFAULT_CONFIGS))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--group")
    selection.add_argument("--exp", nargs="+")
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", nargs="?", const="auto", metavar="CHECKPOINT")
    parser.add_argument("--checkpoint")
    parser.add_argument("--backend", choices=list(MODULES), help="Explicit checkpoint backend when no run_config.json is available")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--image")
    parser.add_argument("--manifest")
    parser.add_argument("--image-root")
    parser.add_argument("--model-path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--device")
    parser.add_argument("--caption-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seeds:
        args.seeds = list(dict.fromkeys(args.seeds))
    # 所有相对路径均以项目根目录为基准，和底层模块保持一致。
    import os
    os.chdir(ROOT)
    try:
        if args.resume and args.action != "train":
            raise ValueError("--resume is only valid for train")
        if "--split" in sys.argv and args.action != "evaluate":
            raise ValueError("--split is only valid for evaluate")
        if any(value is not None and value < 1 for value in (args.batch_size, args.accumulation)):
            raise ValueError("Batch size and accumulation must be positive")
        if args.accumulation and args.action != "train":
            raise ValueError("--accumulation is only valid for train")
        if args.caption_checkpoint and args.action != "train":
            raise ValueError("--caption-checkpoint is only valid for train")
        if args.image and args.action != "predict":
            raise ValueError("--image is only valid for predict")
        if (args.manifest or args.image_root) and args.action != "evaluate":
            raise ValueError("--manifest/--image-root are only valid for evaluate")
        if args.backend and not args.checkpoint:
            raise ValueError("--backend is only used with --checkpoint")
        if args.action in ("list", "summarize") and any(value is not None for value in (args.model_path, args.batch_size, args.device)):
            raise ValueError("Model overrides are not used by list/summarize")
        if args.action == "predict" and (not args.checkpoint or not args.image):
            raise ValueError("predict requires --checkpoint and --image")
        if args.checkpoint:
            if args.action not in ("evaluate", "predict") or args.group or args.exp or args.seeds or args.output_dir:
                raise ValueError("--checkpoint is for single evaluate/predict; do not combine it with groups, experiments, seeds or output roots")
            backend, saved_config = checkpoint_backend(args.task, args.checkpoint, args.backend)
            if backend == "caption_baseline" and not saved_config:
                saved_config = load_yaml(args.config or DEFAULT_CONFIGS["task1"][0])
            command = inference_command(args, backend, args.checkpoint, saved_config)
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                if not Path(args.checkpoint).is_file():
                    raise FileNotFoundError(args.checkpoint)
                subprocess.run(command, check=True, cwd=ROOT)
            return
        entries, groups, aliases = catalog(args.task, args.config)
        if args.group:
            if args.group not in groups:
                raise ValueError(f"Unknown group {args.group}; choose from {list(groups)}")
            names = groups[args.group]
        elif args.exp:
            names = list(dict.fromkeys(aliases.get(name, name) for name in args.exp))
            if any(name not in entries for name in names):
                raise ValueError("Unknown experiment; use list to inspect names")
        else:
            if args.action in ("train", "evaluate"):
                raise ValueError(f"{args.action} requires --group or --exp (or --checkpoint for evaluate)")
            names = list(entries)
        if any(value is not None for value in (args.model_path, args.batch_size, args.accumulation)) and len(names) != 1:
            raise ValueError("--model-path/--batch-size/--accumulation require exactly one experiment")
        if args.caption_checkpoint and not any(entries[name]["config"].get("backend") == "siglip" for name in names):
            raise ValueError("--caption-checkpoint requires selecting siglip_transfer")
        jobs = [(entries[name], seed, run_directory(entries[name], seed, args.output_dir))
                for name in names for seed in (args.seeds or [entries[name]["config"]["seed"]])]
        if args.resume and args.resume != "auto":
            if len(jobs) != 1 or args.output_dir:
                raise ValueError("An explicit resume checkpoint requires one experiment/seed and no --output-dir")
            entry, seed, _ = jobs[0]
            saved_config = read_json(Path(args.resume).parent / "run_config.json", {})
            if saved_config:
                saved_name = aliases.get(saved_config.get("experiment"), saved_config.get("experiment"))
                if saved_name != entry["name"]:
                    raise ValueError("Selected experiment differs from resume run")
                if args.seeds and seed != saved_config["seed"]:
                    raise ValueError("Resume seed differs")
                seed = saved_config["seed"]
            jobs = [(entry, seed, Path(args.resume).resolve().parent)]
        if args.action == "list":
            for name in names:
                entry = entries[name]
                print(f"{name:30} groups={','.join(entry['groups']):20} metric={entry['config']['runner']['selection_metric']}\n  {entry['description']}\n  backend={entry['backend']} output={entry['config']['output_dir']}")
            return
        if args.action == "summarize":
            summarize(jobs, args.task, args.dry_run)
            return
        commands = []
        # 先解析所有参数，确保不支持的旧基线选项不会在跑完部分实验后才报错。
        for entry, seed, run in jobs:
            config = entry["config"]
            if args.action == "train":
                command = training_command(args, entry, seed, run)
                completed = (run / "completed.json").is_file() and (run / "best.pt").is_file()
                if completed:
                    print(f"Skip completed: {run}", flush=True)
                    continue
                explicit = args.resume and args.resume != "auto"
                if not args.dry_run:
                    if explicit and not Path(args.resume).is_file():
                        raise FileNotFoundError(args.resume)
                    if (run / "last.pt").exists() and not args.resume:
                        raise FileExistsError(f"Incomplete run: {run}; supply --resume")
                    if run.exists() and any(run.iterdir()) and not (args.resume and (explicit or (run / "last.pt").is_file())):
                        raise FileExistsError(f"Nonempty run directory: {run}; choose a new output root")
            else:
                checkpoint = run / "best.pt"
                saved = read_json(run / "run_config.json", config)
                command = inference_command(args, entry["backend"], checkpoint, saved)
                if not args.dry_run and not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
            commands.append((entry, seed, run, command))
        for entry, seed, run, command in commands:
            print(f"{args.task}/{entry['name']} seed={seed} -> {run}", flush=True)
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True, cwd=ROOT)
        if not (args.resume and args.resume != "auto"):
            # 汇总该输出根目录中的已登记组，避免后跑的组覆盖共享实验的汇总。
            summary_jobs = [(entry, seed, run_directory(entry, seed, args.output_dir)) for entry in entries.values()
                            for seed in (args.seeds or [entry["config"]["seed"]])]
            summarize(summary_jobs, args.task, args.dry_run)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
