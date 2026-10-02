"""LoRA fine-tuning for the shared Qwen structured-output adapter."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
from utils.config import load_yaml
from utils.jsonl import read_jsonl
from utils.seed import seed_everything
from .prompts import direct_prompt, items_prompt, nutrients_prompt
from .schema import canonical_json


def _task_target(row, task):
    label = row["label"]
    if task == "items":
        return json.dumps({"items": label["items"]}, ensure_ascii=False), items_prompt()
    if task == "nutrients":
        target = {key: label[key] for key in ("total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g")}
        return json.dumps(target, ensure_ascii=False), nutrients_prompt({"items": label["items"]})
    return canonical_json(label), direct_prompt()


def _messages(image, prompt, answer=None):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class QwenDataset(torch.utils.data.Dataset):
    def __init__(self, rows, processor, tasks, max_length, image_size):
        self.rows = rows
        self.processor = processor
        self.tasks = tasks
        self.max_length = max_length
        self.image_size = image_size
        self.samples = [(row, task) for row in rows for task in tasks]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row, task = self.samples[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        answer, prompt = _task_target(row, task)
        prompt_messages = _messages(image, prompt)
        full_messages = _messages(image, prompt, answer)
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_inputs = self.processor(text=prompt_text, images=image, return_tensors="pt", truncation=True, max_length=self.max_length)
        encoded = self.processor(text=full_text, images=image, return_tensors="pt", truncation=True, max_length=self.max_length)
        input_ids = encoded["input_ids"][0]
        prompt_length = min(prompt_inputs["input_ids"].shape[1], input_ids.shape[0])
        if prompt_length >= input_ids.shape[0]:
            raise ValueError("max_length leaves no assistant answer tokens")
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        item = {key: value.squeeze(0) for key, value in encoded.items() if key not in ("input_ids", "attention_mask")}
        item["input_ids"] = input_ids
        item["attention_mask"] = encoded["attention_mask"][0]
        item["labels"] = labels
        return item


def _collate(features, processor):
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    text_features = []
    for item in features:
        text_item = {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
        if "mm_token_type_ids" in item:
            text_item["mm_token_type_ids"] = item["mm_token_type_ids"]
        text_features.append(text_item)
    padded = tokenizer.pad(text_features, padding=True, return_tensors="pt")
    labels = torch.full_like(padded["input_ids"], -100)
    for index, item in enumerate(features):
        labels[index, :item["labels"].numel()] = item["labels"]
    padded["labels"] = labels
    for key in ("pixel_values", "image_grid_thw", "video_grid_thw"):
        values = [item[key] for item in features if key in item]
        if values:
            padded[key] = torch.stack(values)
    return padded


def train(config_path: str, model_name: str | None = None):
    config = load_yaml(config_path)
    seed_everything(config.get("seed", 2026))
    qwen = config["qwen"]
    model_name = model_name or qwen["model_name"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    lora = LoraConfig(r=qwen["lora_r"], lora_alpha=qwen["lora_alpha"], lora_dropout=qwen["lora_dropout"], target_modules=qwen["target_modules"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    if qwen.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    data_dir = Path(config["data_dir"])
    image_root_value = config.get("image_root")
    if not image_root_value:
        info_path = data_dir / "dataset_info.json"
        image_root_value = json.loads(info_path.read_text(encoding="utf-8")).get("image_root", "") if info_path.exists() else ""
    image_root = Path(image_root_value)
    rows = list(read_jsonl(data_dir / "train.jsonl"))
    val_rows = list(read_jsonl(data_dir / "val.jsonl"))
    rows = [{**row, "image_path": str(image_root / row["image_path"])} for row in rows]
    val_rows = [{**row, "image_path": str(image_root / row["image_path"])} for row in val_rows]
    tasks = ["direct", "items", "nutrients"]
    dataset = QwenDataset(rows, processor, tasks, int(config.get("max_length", 256)), int(qwen.get("image_size", config.get("image_size", 448))))
    eval_dataset = QwenDataset(val_rows, processor, tasks, int(config.get("max_length", 256)), int(qwen.get("image_size", config.get("image_size", 448))))
    output = Path(config.get("output_dir", "results/meal_json")) / Path(model_name).name / "shared_adapter"
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=qwen["epochs"],
        per_device_train_batch_size=qwen["batch_size"],
        per_device_eval_batch_size=qwen.get("eval_batch_size", qwen["batch_size"]),
        gradient_accumulation_steps=qwen["gradient_accumulation_steps"],
        learning_rate=qwen["learning_rate"],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=eval_dataset, data_collator=lambda batch: _collate(batch, processor))
    trainer.train()
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    (output / "training_config.json").write_text(json.dumps({**config, "model_name": model_name, "tasks": tasks}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not (output / "adapter_config.json").exists():
        raise RuntimeError("Qwen Adapter was not saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/meal_json.yaml")
    parser.add_argument("--model-name")
    args = parser.parse_args()
    train(args.config, args.model_name)
