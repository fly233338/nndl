# 实验命令速查（Linux 服务器）

Task 2 的 4 组 ViT、每个 Qwen 模型 9 组调参配置及批量命令见 [Task 2 调参命令](task2_tuning.md)。以下任务二命令使用默认配置输出目录；原来已训练模型仍位于 `results/meal_json/`。

每条命令单独一行。数据、模型和评测依赖已放好；已有结果可跳过训练和推理。

## 0. 公共环境

```bash
cd /public/home/rxa-jj-wzk/projects/fyl/lens
conda activate lens
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
java -version
split=test
```

调参时用 `split=val`，最终结果用 `split=test`。环境安装见 [服务器训练文档](task2_training.md)。

## 1. 任务一：图像描述

### 数据准备（已有处理结果则跳过）

```bash
python -m scripts.prepare_food500cap_hf --dataset-dir data/raw/Food500Cap --output-dir data/processed/food500cap --val-ratio 0.1 --seed 2026
```

### 训练全部五个模型

```bash
python -m experiments.captioning.train --config config/captioning.yaml
```

### 只训练一个模型

```bash
python -m experiments.captioning.train --config config/captioning.yaml --model vit_transformer
```

### 五个模型推理并算分（greedy 和 beam）

```bash
caption_image_root=$(python -c "import json; print(json.load(open('data/processed/food500cap/dataset_info.json'))['image_root'])")
for model in cnn_gru attention_rnn transformer graph_transformer vit_transformer; do for decode in greedy beam; do python -m experiments.captioning.evaluate --checkpoint "results/captioning/$model/best.pt" --manifest "data/processed/food500cap/$split.jsonl" --image-root "$caption_image_root" --decode "$decode" --beam-size 3 --output "results/captioning/$model/$split/$decode/metrics.json" || break 2; done; done
```

### 已有预测，只重新算分（以 ViT beam 为例）

```bash
python -c "import json; from pathlib import Path; from utils.jsonl import read_jsonl; from utils.metrics import evaluate_caption_records, metric_metadata; p=Path('results/captioning/vit_transformer/$split/beam'); r=evaluate_caption_records(list(read_jsonl(p/'predictions_beam.jsonl'))); r['metric_metadata']=metric_metadata(); s=json.dumps(r,ensure_ascii=False,indent=2); (p/'scores.json').write_text(s,encoding='utf-8'); print(s)"
```

## 2. 任务二：膳食 JSON

### 数据准备（保留原始划分和标注）

```bash
python -m scripts.prepare_meal_json --config config/data.yaml
```

### A. ViT-Transformer baseline：训练

Task 1 权重为 `models/best.pt`，ViT 目录为 `models/vit-base-patch16-224`。

```bash
python -m experiments.meal_json.train_baseline --config config/meal_json.yaml
```

### A. baseline：greedy 推理、算分

```bash
python -m experiments.meal_json.infer_baseline --checkpoint results/meal_json/default/baseline_vit_transformer/best.pt --manifest "data/processed/meal_json/$split.jsonl" --image-root models --decode greedy --batch-size 16 --output "results/meal_json/default/baseline_vit_transformer/${split}_greedy.jsonl"
python -m experiments.meal_json.evaluate --predictions "results/meal_json/default/baseline_vit_transformer/${split}_greedy.jsonl" --output "results/meal_json/default/baseline_vit_transformer/${split}_greedy_metrics.json"
```

### A. baseline：beam 推理、算分

```bash
python -m experiments.meal_json.infer_baseline --checkpoint results/meal_json/default/baseline_vit_transformer/best.pt --manifest "data/processed/meal_json/$split.jsonl" --image-root models --decode beam --beam-size 3 --batch-size 16 --output "results/meal_json/default/baseline_vit_transformer/${split}_beam.jsonl"
python -m experiments.meal_json.evaluate --predictions "results/meal_json/default/baseline_vit_transformer/${split}_beam.jsonl" --output "results/meal_json/default/baseline_vit_transformer/${split}_beam_metrics.json"
```

### B/C. Qwen3：训练一次，共用 Adapter

```bash
python -m experiments.meal_json.train_qwen --config config/meal_json.yaml --model-name /public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct
```

### B. Qwen3 直接生成：推理、算分

```bash
python -m experiments.meal_json.infer --adapter-dir results/meal_json/default/Qwen3-VL-4B-Instruct/shared_adapter --base-model-name /public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct --manifest "data/processed/meal_json/$split.jsonl" --image-root models --mode direct --batch-size 8 --output "results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_direct.jsonl"
python -m experiments.meal_json.evaluate --predictions "results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_direct.jsonl" --output "results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_direct_metrics.json"
```

### C. Qwen3 两阶段生成：推理、算分

```bash
python -m experiments.meal_json.infer --adapter-dir results/meal_json/default/Qwen3-VL-4B-Instruct/shared_adapter --base-model-name /public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct --manifest "data/processed/meal_json/$split.jsonl" --image-root models --mode two_stage --batch-size 8 --output "results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_two_stage.jsonl"
python -m experiments.meal_json.evaluate --predictions "results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_two_stage.jsonl" --output "results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_two_stage_metrics.json"
```

### D. Qwen2.5：训练

```bash
python -m experiments.meal_json.train_qwen --config config/meal_json.yaml --model-name /public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct
```

### D. Qwen2.5 直接生成：推理、算分

```bash
python -m experiments.meal_json.infer --adapter-dir results/meal_json/default/Qwen2.5-VL-3B-Instruct/shared_adapter --base-model-name /public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct --manifest "data/processed/meal_json/$split.jsonl" --image-root models --mode direct --batch-size 8 --output "results/meal_json/default/Qwen2.5-VL-3B-Instruct/${split}_direct.jsonl"
python -m experiments.meal_json.evaluate --predictions "results/meal_json/default/Qwen2.5-VL-3B-Instruct/${split}_direct.jsonl" --output "results/meal_json/default/Qwen2.5-VL-3B-Instruct/${split}_direct_metrics.json"
```

### 一条命令依次训练任务二全部模型（已训练则跳过）

```bash
python -m experiments.meal_json.train_baseline --config config/meal_json.yaml && python -m experiments.meal_json.train_qwen --config config/meal_json.yaml --model-name /public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct && python -m experiments.meal_json.train_qwen --config config/meal_json.yaml --model-name /public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct
```

### 已有预测，统一重新算分（baseline beam + 三组 Qwen）

```bash
for predictions in "results/meal_json/default/baseline_vit_transformer/${split}_beam.jsonl" "results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_direct.jsonl" "results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_two_stage.jsonl" "results/meal_json/default/Qwen2.5-VL-3B-Instruct/${split}_direct.jsonl"; do python -m experiments.meal_json.evaluate --predictions "$predictions" --output "${predictions%.jsonl}_metrics.json" || break; done
```

汇总时同时记录 `mse`、`numeric_coverage`、`schema_valid_rate`。

## 3. 任务三：营养 Agent

当前只实现了 JSON 校验和营养字段读取，无独立训练或批量算分入口；需已安装 `langgraph`。

### 从任务二结果中提取一条有效 JSON

```bash
mkdir -p results/nutrition_agent
python -c "import json; from pathlib import Path; from utils.jsonl import read_jsonl; rows=read_jsonl('results/meal_json/default/Qwen3-VL-4B-Instruct/${split}_direct.jsonl'); row=next(r for r in rows if (r.get('parsed') or {}).get('valid')); Path('results/nutrition_agent/meal.json').write_text(json.dumps(row['parsed']['parsed'],ensure_ascii=False,indent=2),encoding='utf-8')"
```

### 运行

```bash
python -m experiments.nutrition_agent.infer --json results/nutrition_agent/meal.json > results/nutrition_agent/result.json
```
