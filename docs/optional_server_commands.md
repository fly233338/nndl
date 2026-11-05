# 选做任务运行命令

在 Linux 服务器的项目根目录执行，使用已安装依赖的 `lens` 环境。环境准备见 [环境与数据准备](task2_training.md)。下面复用已有模型服务；接口地址和模型 ID 按实际部署填写。

## 1. RL：CIDEr-D 策略梯度微调

需要任务一数据清单、图片、ViT backbone 和 `models/best.pt`。路径与训练参数在 `config/optional/caption_rl.yaml`。

加载时自动兼容新旧 Transformers 的 ViT 参数名，并严格检查全部权重。成功后打印 `Loaded complete checkpoint`；参数缺失或形状不符仍会停止，不会跳过未加载的权重。

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

默认关闭营养校准，直接使用 USDA，不需要语义缓存或类别统计。重量配置为 `config/optional/weight.yaml`，Agent 配置为 `config/optional/nutrition_agent.yaml`；实现说明见 [任务三方案](task3_nutrition_agent.md)。

先放好 `models/dinov2-small/`、课程数据 `data/processed/meal_json/{train,val,test}.jsonl` 及图片，以及 `data/processed/nutrition_agent/usda.jsonl`、`usda_metadata.json`。已有任务二处理数据可直接复用。未生成清单时运行 `python -m experiments.task2_meal_json.prepare_meal_json`；只有原始 USDA ZIP 时运行 `python -m experiments.optional.nutrition_agent.prepare_resources`。

### 准备数据和独立重量模型

已有匹配当前 DINOv2 的 `results/nutrition_agent/weight/ridge.joblib` 时可跳过训练；关闭营养校准不能替代重量模型训练。

```bash
python -m experiments.optional.nutrition_agent.prepare_cache --kind visual --split train
python -m experiments.optional.nutrition_agent.prepare_cache --kind visual --split val
python -m experiments.optional.nutrition_agent.weight --action train
python -m experiments.optional.nutrition_agent.weight --action evaluate --split val
```

重量工具不需要 LLM。多视角数据须用同一菜品的共享分组字段，避免训练和评测重叠。

### 连接模型

模型服务必须支持 RGB 输入和 tool calling。下面复用 Judge 的地址、密钥和模型选择；服务只有一个模型时自动获取 ID，有多个时先设置 `JUDGE_MODEL`。

```bash
export OPENAI_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8005/v1}"
export OPENAI_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export AGENT_MODEL="$(python -c 'from utils.config import load_yaml; from experiments.optional.llm_judge.client import connect; c,m=connect(load_yaml("config/optional/llm_judge.yaml")); print(m); c.close()')"
```

可选的营养校准不属于本流程；需要时按 [任务三方案](task3_nutrition_agent.md) 单独开启。

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

冻结重量模型、营养库、映射、Prompt 和配置后，将 `split=val` 改为 `split=test` 重复上面整段，完成最终测试。需要单独报告重量工具的测试表现时运行：

```bash
python -m experiments.optional.nutrition_agent.prepare_cache --kind visual --split test
python -m experiments.optional.nutrition_agent.weight --action evaluate --split test
```

完整对照结果在 `results/nutrition_agent/{val,test}/comparison.json`，独立重量结果在 `results/nutrition_agent/weight/`。同时报告 F1、五项 MSE/MAE、数值覆盖率和失败率。

训练集名称候选规范化是独立可选实验，不需要微调模型，且不改变原 Agent 输出目录或评分规则，见 [训练集名称查表](task3_food_names.md)。
