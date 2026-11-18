# 实验运行命令

以下命令在 Linux 服务器的项目根目录执行。环境安装见 [环境与数据准备](task2_training.md)，RL 和 Judge 见 [选做任务命令](optional_server_commands.md)。已有数据或模型可跳过对应准备和训练步骤。

## 1. 环境与数据

```bash
conda activate lens
export CUDA_VISIBLE_DEVICES=0
```

任务一：本地需有 Food500Cap Parquet、`models/resnet-50` 和 `models/vit-base-patch16-224`。

```bash
python -m experiments.task1_captioning.prepare_food500cap_hf --dataset-dir data/raw/Food500Cap
```

任务二使用课程标注与图片；先在 `config/data.yaml` 确认路径。

```bash
python -m experiments.task2_meal_json.prepare_meal_json
```

## 2. 任务一：图像描述

训练配置中的五个模型；只跑一个时追加 `--model vit_transformer`。

```bash
python -m experiments.task1_captioning.train
```

统一测试 greedy 和 beam，图片根目录应与 `dataset_info.json` 一致。

```bash
for model in cnn_gru attention_rnn transformer graph_transformer vit_transformer; do
  for decode in greedy beam; do
    python -m experiments.task1_captioning.evaluate --checkpoint "results/captioning/$model/best.pt" --manifest data/processed/food500cap/test.jsonl --image-root data/processed/food500cap/images --decode "$decode" --beam-size 3 --output "results/captioning/$model/test/$decode/metrics.json" || break 2
  done
done
```

任务一新增的归一化、增强与微调配置，以及新编码器的后续设计见 [任务一提分方案](task1_improvements.md)。上述五种结构仍作为基线保留。

## 3. 任务二：膳食 JSON

先将两个变量改为服务器实际的完整模型目录。baseline 还需要 `models/best.pt` 和 `models/vit-base-patch16-224`。

```bash
qwen3=models/Qwen3-VL-4B-Instruct
qwen25=models/Qwen2.5-VL-3B-Instruct
python -m experiments.task2_meal_json.train_baseline
python -m experiments.task2_meal_json.train_qwen --model-name "$qwen3"
python -m experiments.task2_meal_json.train_qwen --model-name "$qwen25"
```

baseline 推理与评测：

```bash
for decode in greedy beam; do
  out="results/meal_json/default/baseline_vit_transformer/test_$decode"
  python -m experiments.task2_meal_json.infer_baseline --checkpoint results/meal_json/default/baseline_vit_transformer/best.pt --manifest data/processed/meal_json/test.jsonl --image-root models --decode "$decode" --beam-size 3 --output "$out.jsonl" || break
  python -m experiments.task2_meal_json.evaluate --predictions "$out.jsonl" --output "${out}_metrics.json" || break
done
```

Qwen3 的两种推理方式复用同一个 Adapter：

```bash
for mode in direct two_stage; do
  out="results/meal_json/default/Qwen3-VL-4B-Instruct/test_$mode"
  python -m experiments.task2_meal_json.infer --adapter-dir results/meal_json/default/Qwen3-VL-4B-Instruct/shared_adapter --base-model-name "$qwen3" --manifest data/processed/meal_json/test.jsonl --image-root models --mode "$mode" --output "$out.jsonl" || break
  python -m experiments.task2_meal_json.evaluate --predictions "$out.jsonl" --output "${out}_metrics.json" || break
done
```

Qwen2.5 对照：

```bash
python -m experiments.task2_meal_json.infer --adapter-dir results/meal_json/default/Qwen2.5-VL-3B-Instruct/shared_adapter --base-model-name "$qwen25" --manifest data/processed/meal_json/test.jsonl --image-root models --mode direct --output results/meal_json/default/Qwen2.5-VL-3B-Instruct/test_direct.jsonl
python -m experiments.task2_meal_json.evaluate --predictions results/meal_json/default/Qwen2.5-VL-3B-Instruct/test_direct.jsonl --output results/meal_json/default/Qwen2.5-VL-3B-Instruct/test_direct_metrics.json
```

Qwen zero-shot 对照：不训练、不加载 Adapter，使用原始 Qwen3/Qwen2.5 模型。提示词、图片尺寸、解码及评测流程与上面的 LoRA 实验相同；结果单独保存在配置的 `zero_shot_output_dir`。

```bash
for base_model in "$qwen3" "$qwen25"; do
  model=$(basename "$base_model")
  out="results/meal_json/zero_shot/$model/direct/test_predictions"
  python -m experiments.task2_meal_json.infer --zero-shot --base-model-name "$base_model" --manifest data/processed/meal_json/test.jsonl --image-root models --mode direct --output "$out.jsonl" || break
  python -m experiments.task2_meal_json.evaluate --predictions "$out.jsonl" --output "${out}_metrics.json" || break
done
```

`--zero-shot` 与 `--adapter-dir` 互斥；不指定 `--output` 时按原始模型名称、推理模式和划分自动保存到独立目录。已有任务二汇总未包含这组结果，需实际推理和评测后再补表。

参数消融见 [消融命令](task2_tuning.md)。汇总时保留食物 F1、各字段 MSE、数值覆盖率和 Schema 有效率。
