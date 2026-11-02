# Task 2：完整草稿＋定向复核

复用已有 Adapter，不重新训练。新增 `--mode draft_review`；保留 `direct` 和 `two_stage`。图片仍为 448×448，每阶段最多生成 768 token。

本次 bundle 保持服务器现有的 `experiments.meal_json`、`config/meal_json_tuning/` 路径，下列命令可直接在服务器运行，不指定 CUDA，固定使用 test 集。

## 1. 环境

```bash
cd /public/home/rxa-jj-wzk/projects/fyl/lens
conda activate lens
export HF_HUB_OFFLINE=1
```

## 2. Qwen3：先跑一组（1e-4／2轮）

推理：

```bash
python -m experiments.meal_json.infer --config config/meal_json_tuning/qwen_lr1e-4_ep2.yaml --adapter-dir results/meal_json/tuning/qwen_lr1e-4_ep2/Qwen3-VL-4B-Instruct/shared_adapter --base-model-name /public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct --manifest data/processed/meal_json/test.jsonl --image-root models --mode draft_review --batch-size 8 --output results/meal_json/tuning/qwen_lr1e-4_ep2/Qwen3-VL-4B-Instruct/draft_review/test_predictions.jsonl
```

算分（需已有 Java 8、CoreNLP）：

```bash
python -m experiments.meal_json.evaluate --predictions results/meal_json/tuning/qwen_lr1e-4_ep2/Qwen3-VL-4B-Instruct/draft_review/test_predictions.jsonl --output results/meal_json/tuning/qwen_lr1e-4_ep2/Qwen3-VL-4B-Instruct/draft_review/test_metrics.json
```

## 3. Qwen3：9组依次推理、算分

```bash
for config in config/meal_json_tuning/qwen_*.yaml; do run=$(basename "$config" .yaml); model="results/meal_json/tuning/$run/Qwen3-VL-4B-Instruct"; out="$model/draft_review"; python -m experiments.meal_json.infer --config "$config" --adapter-dir "$model/shared_adapter" --base-model-name /public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct --manifest data/processed/meal_json/test.jsonl --image-root models --mode draft_review --batch-size 8 --output "$out/test_predictions.jsonl" && python -m experiments.meal_json.evaluate --predictions "$out/test_predictions.jsonl" --output "$out/test_metrics.json" || break; done
```

## 4. Qwen2.5：9组依次推理、算分（需要时运行）

```bash
for config in config/meal_json_tuning/qwen_*.yaml; do run=$(basename "$config" .yaml); model="results/meal_json/tuning/$run/Qwen2.5-VL-3B-Instruct"; out="$model/draft_review"; python -m experiments.meal_json.infer --config "$config" --adapter-dir "$model/shared_adapter" --base-model-name /public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct --manifest data/processed/meal_json/test.jsonl --image-root models --mode draft_review --batch-size 8 --output "$out/test_predictions.jsonl" && python -m experiments.meal_json.evaluate --predictions "$out/test_predictions.jsonl" --output "$out/test_metrics.json" || break; done
```

## 5. 已有复核预测，只重新算分

```bash
for predictions in results/meal_json/tuning/qwen_*/Qwen3-VL-4B-Instruct/draft_review/test_predictions.jsonl results/meal_json/tuning/qwen_*/Qwen2.5-VL-3B-Instruct/draft_review/test_predictions.jsonl; do python -m experiments.meal_json.evaluate --predictions "$predictions" --output "${predictions%test_predictions.jsonl}test_metrics.json" || break; done
```

## 6. 输出与行为

```text
results/meal_json/tuning/<配置>/<模型>/draft_review/
  test_predictions.jsonl
  test_metrics.json
```

未指定 `--output` 时，新模式也默认保存到 Adapter 父目录下的 `draft_review/<清单名>_predictions.jsonl`；原有 direct/two_stage 默认输出不变。

- 第一阶段使用 direct 生成完整草稿；第二阶段结合原图、草稿和检查提示返回完整 JSON，可修改食材、份量和营养值。
- 检查提示只使用草稿：Schema错误、重复名称、单项重量大于总重量、营养素质量超过总重量。不读取真实标签，不强制食材重量之和等于总重量，也不强制热量公式精确相等。
- 复核无效或调用失败时，仅在草稿有效的情况下回退；两次都无效时保留失败记录。Schema合法不代表语义一定改善，仍按原评测指标比较。
- 每条预测保留 `stage1_raw`、`stage2_raw`、`draft_validation`、`review_checks`、`review_validation`、`review_fallback` 和 `review_error`。模型调用失败时可能没有 `review_validation`。
- 原先 `test_direct.jsonl` 和 `test_two_stage.jsonl` 不被上述命令覆盖。

查看一组复核中的有效草稿、有效复核、最终有效结果和回退数量：

```bash
python -c "import json; p='results/meal_json/tuning/qwen_lr1e-4_ep2/Qwen3-VL-4B-Instruct/draft_review/test_predictions.jsonl'; rows=[json.loads(s) for s in open(p,encoding='utf-8') if s.strip()]; print({'samples':len(rows),'draft_valid':sum(bool((r.get('draft_validation') or {}).get('valid')) for r in rows),'review_valid':sum(bool((r.get('review_validation') or {}).get('valid')) for r in rows),'final_valid':sum(bool((r.get('parsed') or {}).get('valid')) for r in rows),'fallback_count':sum(bool(r.get('review_fallback')) for r in rows)})"
```
