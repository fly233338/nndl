# 任务二参数消融命令

在 Linux 项目根目录执行。配置位于 `config/task2_meal_json/meal_json_tuning/`：ViT 为学习率 `{1e-5, 5e-5}` × 轮数 `{10, 20}`；Qwen 为学习率 `{1e-5, 5e-5, 1e-4}` × 轮数 `{1, 2, 3}`。各组独立训练，按验证损失保存检查点。

## 1. ViT：训练与评测

```bash
for config in config/task2_meal_json/meal_json_tuning/vit_*.yaml; do
  python -m experiments.task2_meal_json.train_baseline --config "$config" || break
done
for config in config/task2_meal_json/meal_json_tuning/vit_*.yaml; do
  run=$(basename "$config" .yaml)
  out="results/meal_json/tuning/$run/baseline_vit_transformer"
  for decode in greedy beam; do
    python -m experiments.task2_meal_json.infer_baseline --checkpoint "$out/best.pt" --manifest data/processed/meal_json/test.jsonl --image-root models --decode "$decode" --beam-size 3 --output "$out/test_$decode.jsonl" || break 2
    python -m experiments.task2_meal_json.evaluate --predictions "$out/test_$decode.jsonl" --output "$out/test_${decode}_metrics.json" || break 2
  done
done
```

## 2. Qwen：训练与评测

按要运行的模型设置下面三个变量，`base_model` 改为实际模型目录：

| 模型 | `model` | `modes` |
|---|---|---|
| Qwen3 | `Qwen3-VL-4B-Instruct` | `direct two_stage` |
| Qwen2.5 | `Qwen2.5-VL-3B-Instruct` | `direct` |

```bash
model=Qwen3-VL-4B-Instruct
base_model=models/Qwen3-VL-4B-Instruct
modes="direct two_stage"
for config in config/task2_meal_json/meal_json_tuning/qwen_*.yaml; do
  python -m experiments.task2_meal_json.train_qwen --config "$config" --model-name "$base_model" || break
done
for config in config/task2_meal_json/meal_json_tuning/qwen_*.yaml; do
  run=$(basename "$config" .yaml)
  out="results/meal_json/tuning/$run/$model"
  for mode in $modes; do
    python -m experiments.task2_meal_json.infer --config "$config" --adapter-dir "$out/shared_adapter" --base-model-name "$base_model" --manifest data/processed/meal_json/test.jsonl --image-root models --mode "$mode" --output "$out/test_$mode.jsonl" || break 2
    python -m experiments.task2_meal_json.evaluate --predictions "$out/test_$mode.jsonl" --output "$out/test_${mode}_metrics.json" || break 2
  done
done
```

已有模型时跳过训练循环；已有预测时仅运行对应的 `evaluate`。需要继续调参时先用 val，冻结后在 test 统一报告各组结果。汇总食物 F1、Schema 有效率、五项 MSE 和数值覆盖率。
