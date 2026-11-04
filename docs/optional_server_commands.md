# 选做任务运行命令

在 Linux 服务器的项目根目录执行，使用已安装依赖的 `lens` 环境。环境准备见 [环境与数据准备](task2_training.md)。下面复用已有模型服务；接口地址和模型 ID 按实际部署填写。

## 1. RL：CIDEr-D 策略梯度微调

需要任务一数据清单、图片、ViT backbone 和 `results/captioning/vit_transformer/best.pt`。路径与训练参数在 `config/optional/caption_rl.yaml`。

```bash
python -m experiments.optional.caption_rl.reward
python -m experiments.optional.caption_rl.train
python -m experiments.optional.caption_rl.evaluate --split val
```

第一步只缓存训练参考文本和 n-gram 文档频率。训练时由当前策略重新采样，实时计算 CIDEr-D，以 greedy 奖励为基线更新解码器。验证集确定参数后执行最终测试：

```bash
python -m experiments.optional.caption_rl.evaluate --split test
```

结果在 `results/caption_rl/vit_transformer_scst/comparison/`。新训练需使用新的 `output_dir`，当前不支持断点续训。奖励分词和传统文本评测需 Java，完整评测还需 CoreNLP。

## 2. LLM-as-Judge

需要任务一的图片、manifest 和预测描述。配置为 `config/optional/llm_judge.yaml`；服务只有一个模型时可自动识别模型 ID，否则设置 `JUDGE_MODEL`。

```bash
export JUDGE_BASE_URL=http://127.0.0.1:8005/v1
export JUDGE_API_KEY=EMPTY
python -m experiments.optional.llm_judge.evaluate --check-api
python -m experiments.optional.llm_judge.evaluate --predictions results/captioning/vit_transformer/test/beam/predictions_beam.jsonl --limit 1 --output results/llm_judge/vit_transformer/smoke.jsonl
python -m experiments.optional.llm_judge.evaluate --predictions results/captioning/vit_transformer/test/beam/predictions_beam.jsonl --output results/llm_judge/vit_transformer/judgments.jsonl
```

接口检查不验证视觉能力，先确认单图评分成功。启用鉴权时替换密钥；客户端位于其他节点时替换主机地址。其他模型替换输入和输出路径即可。

与传统指标分析一致性：

```bash
python -m experiments.optional.llm_judge.metrics --predictions results/captioning/vit_transformer/test/beam/predictions_beam.jsonl --output results/llm_judge/vit_transformer/automatic_scores.jsonl
python -m experiments.optional.llm_judge.analyze --records results/llm_judge/vit_transformer/judgments.jsonl --automatic-scores results/llm_judge/vit_transformer/automatic_scores.jsonl --output results/llm_judge/vit_transformer/agreement.json
```

有真实人工评分时，第二条追加 `--human-ratings path/to/ratings.jsonl`。每行格式为 `{"image_id":"...","correctness":4,"coverage":3,"groundedness":4,"fluency":5}`，须对应同一模型的同一句描述。

## 3. 任务三：营养 Agent

实现说明见 [任务三方案](task3_nutrition_agent.md)。先确认 `config/data.yaml` 中课程标注与图片路径正确；重量配置为 `config/optional/weight.yaml`，Agent 配置为 `config/optional/nutrition_agent.yaml`。

### 准备数据和独立重量模型

已有处理数据或本地 DINOv2 模型可跳过对应命令。服务器无法联网时，在联网机器下载后复制完整模型目录。

```bash
python -m experiments.task2_meal_json.prepare_meal_json
hf download facebook/dinov2-small --local-dir models/dinov2-small
python -m experiments.optional.nutrition_agent.prepare_cache --kind visual --split train
python -m experiments.optional.nutrition_agent.prepare_cache --kind visual --split val
python -m experiments.optional.nutrition_agent.weight --action train
python -m experiments.optional.nutrition_agent.weight --action evaluate --split val
```

重量工具不需要 LLM。多视角数据须用同一菜品的共享分组字段，避免训练和评测重叠。

### 准备营养库并连接模型

已有 `usda.jsonl` 和 metadata 可跳过营养库准备。模型服务必须支持 RGB 输入和 tool calling；设置准确的服务模型 ID。

```bash
python -m experiments.optional.nutrition_agent.prepare_resources --download
export OPENAI_BASE_URL=http://127.0.0.1:8005/v1
export OPENAI_API_KEY=EMPTY
export AGENT_MODEL="实际部署的模型ID"
python -m experiments.optional.nutrition_agent.prepare_cache --kind nutrition --split train
python -m experiments.optional.nutrition_agent.calibrate_nutrition
```

最后两步是默认开启的训练集营养校准。只使用原始 USDA 时，在 Agent 配置设置 `nutrition.use_calibration: false` 并跳过这两步。离线服务器可将固定版本 USDA 压缩包放到配置路径，再省略 `--download`。

### 单图检查与开发集对照

```bash
python -m experiments.optional.nutrition_agent.infer --manifest data/processed/meal_json/val.jsonl --mode agent --limit 1 --output results/nutrition_agent/smoke/predictions.jsonl
```

确认输出 `status` 为 `success` 后，运行完整对照：

```bash
split=val
for mode in direct agent; do
  python -m experiments.optional.nutrition_agent.infer --manifest "data/processed/meal_json/$split.jsonl" --mode "$mode" --output "results/nutrition_agent/$split/$mode/predictions.jsonl" || break
done
python -m experiments.optional.nutrition_agent.evaluate --predictions "results/nutrition_agent/$split/direct/predictions.jsonl" "results/nutrition_agent/$split/agent/predictions.jsonl" --manifest "data/processed/meal_json/$split.jsonl" --output "results/nutrition_agent/$split/comparison.json"
```

冻结重量模型、营养库、校准、映射、Prompt 和配置后，将 `split=val` 改为 `split=test` 重复上面整段，完成最终测试。需要单独报告重量工具的测试表现时运行：

```bash
python -m experiments.optional.nutrition_agent.prepare_cache --kind visual --split test
python -m experiments.optional.nutrition_agent.weight --action evaluate --split test
```

完整对照结果在 `results/nutrition_agent/{val,test}/comparison.json`，独立重量结果在 `results/nutrition_agent/weight/`。同时报告 F1、五项 MSE/MAE、数值覆盖率和失败率。
