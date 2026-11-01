# Task 2 调参命令（Linux 服务器）

## 1. 公共设置

```bash
cd /public/home/rxa-jj-wzk/projects/fyl/lens
conda activate lens
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
qwen3=/public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct
qwen25=/public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct
split=val
```

Qwen 统一：训练长度 1024，训练/推理图片 448×448，每阶段生成上限 768；batch size 8，梯度累积 1，LoRA r=16、alpha=32，梯度检查点开启。ViT：长度 768，batch size 16，workers 4。

## 2. 配置与输出

配置位于 `config/meal_json_tuning/`，均继承 `config/meal_json.yaml`。

| 配置文件名 | 学习率 | epoch 上限 |
|---|---:|---:|
| `vit_lr1e-5_ep10.yaml` | 1e-5 | 10 |
| `vit_lr1e-5_ep20.yaml` | 1e-5 | 20 |
| `vit_lr5e-5_ep10.yaml` | 5e-5 | 10 |
| `vit_lr5e-5_ep20.yaml` | 5e-5 | 20 |
| `qwen_lr1e-5_ep1.yaml` | 1e-5 | 1 |
| `qwen_lr1e-5_ep2.yaml` | 1e-5 | 2 |
| `qwen_lr1e-5_ep3.yaml` | 1e-5 | 3 |
| `qwen_lr5e-5_ep1.yaml` | 5e-5 | 1 |
| `qwen_lr5e-5_ep2.yaml` | 5e-5 | 2 |
| `qwen_lr5e-5_ep3.yaml` | 5e-5 | 3 |
| `qwen_lr1e-4_ep1.yaml` | 1e-4 | 1 |
| `qwen_lr1e-4_ep2.yaml` | 1e-4 | 2 |
| `qwen_lr1e-4_ep3.yaml` | 1e-4 | 3 |

ViT 4 组，Qwen3 和 Qwen2.5 各 9 组，共 22 次训练。每组从相同初始权重独立训练；最终模型按最低验证损失保存，可能早于最后一个 epoch。

```text
results/meal_json/tuning/vit_lr5e-5_ep20/baseline_vit_transformer/best.pt
results/meal_json/tuning/qwen_lr1e-5_ep2/Qwen3-VL-4B-Instruct/shared_adapter/
results/meal_json/tuning/qwen_lr1e-5_ep2/Qwen2.5-VL-3B-Instruct/shared_adapter/
```

原有结果仍在 `results/meal_json/`；默认配置的新训练写入 `results/meal_json/default/`。以下调参命令均写入 `results/meal_json/tuning/`。重复运行同一组会使用同一输出目录，不自动续训。

## 3. 批量训练（按需选择，每条命令内部串行）

### ViT：4 组

```bash
for config in config/meal_json_tuning/vit_*.yaml; do python -m experiments.meal_json.train_baseline --config "$config" || break; done
```

### Qwen3：9 组

```bash
for config in config/meal_json_tuning/qwen_*.yaml; do python -m experiments.meal_json.train_qwen --config "$config" --model-name "$qwen3" || break; done
```

### Qwen2.5：9 组

```bash
for config in config/meal_json_tuning/qwen_*.yaml; do python -m experiments.meal_json.train_qwen --config "$config" --model-name "$qwen25" || break; done
```

## 4. 批量验证集推理

### ViT：greedy + beam=3

```bash
for config in config/meal_json_tuning/vit_*.yaml; do run=$(basename "$config" .yaml); out="results/meal_json/tuning/$run/baseline_vit_transformer"; for decode in greedy beam; do python -m experiments.meal_json.infer_baseline --checkpoint "$out/best.pt" --manifest "data/processed/meal_json/$split.jsonl" --image-root models --decode "$decode" --beam-size 3 --batch-size 16 --output "$out/${split}_${decode}.jsonl" || break 2; done; done
```

### Qwen3：direct + two_stage（共用 Adapter）

```bash
for config in config/meal_json_tuning/qwen_*.yaml; do run=$(basename "$config" .yaml); out="results/meal_json/tuning/$run/Qwen3-VL-4B-Instruct"; for mode in direct two_stage; do python -m experiments.meal_json.infer --config "$config" --adapter-dir "$out/shared_adapter" --base-model-name "$qwen3" --manifest "data/processed/meal_json/$split.jsonl" --image-root models --mode "$mode" --batch-size 8 --output "$out/${split}_${mode}.jsonl" || break 2; done; done
```

### Qwen2.5：direct

```bash
for config in config/meal_json_tuning/qwen_*.yaml; do run=$(basename "$config" .yaml); out="results/meal_json/tuning/$run/Qwen2.5-VL-3B-Instruct"; python -m experiments.meal_json.infer --config "$config" --adapter-dir "$out/shared_adapter" --base-model-name "$qwen25" --manifest "data/processed/meal_json/$split.jsonl" --image-root models --mode direct --batch-size 8 --output "$out/${split}_direct.jsonl" || break; done
```

## 5. 批量算分（需先准备好 Java 8、CoreNLP）

```bash
for predictions in results/meal_json/tuning/vit_*/baseline_vit_transformer/"${split}_greedy.jsonl" results/meal_json/tuning/vit_*/baseline_vit_transformer/"${split}_beam.jsonl" results/meal_json/tuning/qwen_*/Qwen3-VL-4B-Instruct/"${split}_direct.jsonl" results/meal_json/tuning/qwen_*/Qwen3-VL-4B-Instruct/"${split}_two_stage.jsonl" results/meal_json/tuning/qwen_*/Qwen2.5-VL-3B-Instruct/"${split}_direct.jsonl"; do python -m experiments.meal_json.evaluate --predictions "$predictions" --output "${predictions%.jsonl}_metrics.json" || break; done
```

## 6. 只跑指定一组（修改 run 即可）

### ViT

```bash
run=vit_lr5e-5_ep20
python -m experiments.meal_json.train_baseline --config "config/meal_json_tuning/$run.yaml"
python -m experiments.meal_json.infer_baseline --checkpoint "results/meal_json/tuning/$run/baseline_vit_transformer/best.pt" --manifest "data/processed/meal_json/$split.jsonl" --image-root models --decode greedy --batch-size 16 --output "results/meal_json/tuning/$run/baseline_vit_transformer/${split}_greedy.jsonl"
python -m experiments.meal_json.evaluate --predictions "results/meal_json/tuning/$run/baseline_vit_transformer/${split}_greedy.jsonl" --output "results/meal_json/tuning/$run/baseline_vit_transformer/${split}_greedy_metrics.json"
```

### Qwen3

```bash
run=qwen_lr1e-5_ep2
python -m experiments.meal_json.train_qwen --config "config/meal_json_tuning/$run.yaml" --model-name "$qwen3"
python -m experiments.meal_json.infer --config "config/meal_json_tuning/$run.yaml" --adapter-dir "results/meal_json/tuning/$run/Qwen3-VL-4B-Instruct/shared_adapter" --base-model-name "$qwen3" --manifest "data/processed/meal_json/$split.jsonl" --image-root models --mode direct --batch-size 8 --output "results/meal_json/tuning/$run/Qwen3-VL-4B-Instruct/${split}_direct.jsonl"
python -m experiments.meal_json.evaluate --predictions "results/meal_json/tuning/$run/Qwen3-VL-4B-Instruct/${split}_direct.jsonl" --output "results/meal_json/tuning/$run/Qwen3-VL-4B-Instruct/${split}_direct_metrics.json"
```

Qwen3 两阶段将 `--mode direct` 和输出文件名中的 `direct` 都换成 `two_stage`。Qwen2.5 将上述命令中的 `$qwen3` 换成 `$qwen25`，目录名换成 `Qwen2.5-VL-3B-Instruct`。

## 7. 最终测试（验证集选好参数后，只运行选中的配置）

```bash
split=test
```

设置选中的 `run`，复用第 6 节推理、算分命令，跳过训练。汇总同时记录 `item_micro_f1`、`schema_valid_rate`、各字段 `mse` 和 `numeric_coverage`。

旧 Qwen Adapter 可以直接按新推理设置重新生成结果，无需重训；旧的 256 token / 原图尺寸预测不用于与本轮结果直接比较。
