# 任务二草稿复核命令

复用已经训练好的共享 Adapter，无需重新训练。在 Linux 项目根目录执行；下面以默认 Qwen3 结果为例，模型目录按实际位置修改。

```bash
base_model=models/Qwen3-VL-4B-Instruct
adapter=results/meal_json/default/Qwen3-VL-4B-Instruct/shared_adapter
out=results/meal_json/default/Qwen3-VL-4B-Instruct/test_draft_review
python -m experiments.task2_meal_json.infer --adapter-dir "$adapter" --base-model-name "$base_model" --manifest data/processed/meal_json/test.jsonl --image-root models --mode draft_review --output "$out.jsonl"
python -m experiments.task2_meal_json.evaluate --predictions "$out.jsonl" --output "${out}_metrics.json"
```

消融实验同时将 `--config`、Adapter 和输出目录换成对应组；Qwen2.5 替换模型目录即可。该模式先生成完整草稿，再结合原图和草稿检查提示复核，保留原始输出与失败记录；真实标签不参与复核。
