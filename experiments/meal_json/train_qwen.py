"""LoRA fine-tuning entry point for Qwen-VL models.

The script deliberately avoids quantization. It uses BF16/FP16 on a 4090 and
supports direct, item-only, and nutrient-only targets for the two-stage setup.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from utils.config import load_yaml
from utils.jsonl import read_jsonl
from utils.seed import seed_everything
from .prompts import direct_prompt, items_prompt, nutrients_prompt
from .schema import canonical_json


def train(config_path: str, mode: str = "direct", model_name: str | None = None):
    config = load_yaml(config_path)
    seed_everything(config.get("seed", 2026))
    qwen = config["qwen"]
    model_name = model_name or qwen["model_name"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, trust_remote_code=True)
    lora = LoraConfig(r=qwen["lora_r"], lora_alpha=qwen["lora_alpha"], lora_dropout=qwen["lora_dropout"], target_modules=qwen["target_modules"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    rows = list(read_jsonl(Path(config["data_dir"]) / "train.jsonl"))

    class Dataset:
        def __len__(self): return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            image = Image.open(Path(config.get("image_root", "")) / row["image_path"]).convert("RGB")
            label = row["label"]
            if mode == "items":
                target = json.dumps({"items": label["items"]}, ensure_ascii=False)
                prompt = items_prompt()
            elif mode == "nutrients":
                target = json.dumps({key: label[key] for key in ("total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g")}, ensure_ascii=False)
                prompt = nutrients_prompt({"items": label["items"]})
            else:
                target = canonical_json(label); prompt = direct_prompt()
            encoded = processor(text=prompt, images=image, return_tensors="pt")
            labels = processor.tokenizer(target, return_tensors="pt", truncation=True, max_length=config.get("max_length", 256)).input_ids[0]
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = labels
            return item
    def collate(features):
        result = {}
        for key in features[0]:
            values = [feature[key] for feature in features]
            if key == "labels":
                result[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=-100)
            elif hasattr(values[0], "shape") and values[0].ndim > 0:
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result
    args = TrainingArguments(output_dir=str(Path(config["output_dir"]) / f"qwen_{mode}"), num_train_epochs=qwen["epochs"], per_device_train_batch_size=qwen["batch_size"], gradient_accumulation_steps=qwen["gradient_accumulation_steps"], learning_rate=qwen["learning_rate"], bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), remove_unused_columns=False, save_strategy="epoch", logging_steps=10, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=Dataset(), data_collator=collate)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/meal_json.yaml")
    parser.add_argument("--mode", choices=["direct", "items", "nutrients"], default="direct")
    parser.add_argument("--model-name")
    args = parser.parse_args()
    train(args.config, args.mode, args.model_name)
