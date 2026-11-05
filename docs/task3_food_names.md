# 任务三：训练集名称查表

先尝试训练词表辅助规范化，不需要微调 Qwen，也不需要重训独立重量模型。配置为 `config/optional/nutrition_agent_names.yaml`，继承原 Agent 配置，使用同一个模型服务和营养库，默认仍关闭营养校准。

## 1. 仅从训练集统计词表

在服务器项目根目录执行，复用任务二的 `data/processed/meal_json/train.jsonl`：

```bash
python -m experiments.optional.nutrition_agent.food_names
python -m experiments.optional.nutrition_agent.food_names --query "broccoli"
```

输出 `data/processed/nutrition_agent/train_food_names.json`，包含所有训练名称、每个名称出现的图片数及训练清单指纹。同一图片内重复名称只计一次，不保存图片级答案或营养标签。脚本不接收 val/test 划分参数，也不读取它们。

## 2. 使用候选名称生成预测

沿用原来的 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 和 `AGENT_MODEL`，先检查一张开发集图片：

```bash
python -m experiments.optional.nutrition_agent.infer --config config/optional/nutrition_agent_names.yaml --manifest data/processed/meal_json/val.jsonl --mode agent --limit 1 --output results/nutrition_agent/name_normalized/smoke/predictions.jsonl
```

原 `lookup_nutrition` 工具会同时返回训练名称候选，正常流程不增加模型请求轮次。若初次营养查表失败但有名称候选，模型可利用原有的一次复核纠正名称后重新查表。Qwen 在 `calculate` 中选择 `canonical_name`；只有名称语义一致时才选择，否则返回 null，保留观察到的原始名称。未知食物不会被强制归入训练类别。

候选检索使用观察名称、类别及已有中英文别名，相似度不是语义置信度，也不是自动评分规则。原始观察名称、候选和模型选择保留在调用 trace 中；查营养库的逻辑、独立重量模型和营养计算公式不变。

## 3. 在开发集对比

```bash
split=val
python -m experiments.optional.nutrition_agent.infer --config config/optional/nutrition_agent_names.yaml --manifest "data/processed/meal_json/$split.jsonl" --mode agent --output "results/nutrition_agent/name_normalized/$split/agent/predictions.jsonl"
python -m experiments.optional.nutrition_agent.evaluate --predictions "results/nutrition_agent/$split/agent/predictions.jsonl" "results/nutrition_agent/name_normalized/$split/agent/predictions.jsonl" --manifest "data/processed/meal_json/$split.jsonl" --output "results/nutrition_agent/name_normalized/$split/comparison.json"
```

第一份预测是已有的原 Agent 输出，无需覆盖或重新生成。继续使用同一套精确名称 F1 和数值指标，不修改任务二或历史评分。

先在 val 核查候选选择、F1、失败率和数值误差，再冻结词表、配置和提示词后测 test。名称规范化不能自动补回漏识别的食物，也不能代替混合菜与原料粒度对齐。不要根据 test 标签新增别名或调整候选规则；已有 test 分数保留为原实验记录。
