# FoodLens 选做三 Agent 实施方案 v2

**独立重量预测模型 + 营养数据库 + 简单 Agent Loop**。

主 Agent 默认 Qwen3.8-27B，负责识别菜品、食材、烹饪方式、组织工具调用和分配食物项重量。Nutrition DB 按名称和食品状态提供营养密度；Weight Estimator 仅从原始 RGB 图片预测整餐重量。最终热量和三大营养素由程序按克计算。

## 独立 Weight Estimator

```text
原始 RGB 图片
    ↓
DINOv2-Small（冻结，通用自监督视觉编码器）
    ↓
CLS token（384 维）拼接所有 patch token 均值（384 维）
    ↓
768 维视觉向量
    ↓
StandardScaler → Ridge
    ↓
整餐总重量（g）
```

768 维是两个 384 维向量的拼接。这里使用不带 register token 的 `facebook/dinov2-small`，提取最后一层归一化后的 `last_hidden_state`；均值只取 patch tokens，不含 CLS。可参照 [DINOv2 模型卡](https://huggingface.co/facebook/dinov2-small) 和 [Transformers 的 CLS + patch mean 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py)。不需要食物专用预训练权重，不更新视觉编码器。

监督标签直接使用课程允许的约 1,000 张训练图片的真实总重量。Ridge 拟合克数目标，通过平方误差加 L2 正则训练；以分组交叉验证的重量 MSE 选择 `alpha`，不使用 LLM 估计值或残差目标，不做 PCA。标准化器只在每个训练折内部拟合，避免验证折信息泄漏。最终在完整训练集拟合一次，保存标准化器与 Ridge 的 sklearn Pipeline。

接口为 `estimate_weight(image_path, config)`，返回 `{"total_weight_g": ...}`。配置只提供模型路径和运行设置；模型输入只有 RGB 图像。总重限制为训练产物中固定的 `min_weight_g` 下界，默认 1 g。模型文件只保存视觉/回归器配置与训练来源，不包含 LLM 配置、菜品属性或营养库。更换多模态 LLM、Prompt 或营养数据库时，视觉缓存与重量模型保持可用；改变 DINOv2、其 processor 或特征定义才需重建视觉缓存和重训。

## 模块

| 文件 | 职责 |
|---|---|
| `weight.py` | 768 维特征、Ridge 分组交叉验证、独立估重和开发集评估 |
| `prepare_cache.py` | 分开生成纯视觉缓存和营养校准所需的语义缓存 |
| `data.py` | 读取实验 manifest 与图片根目录 |
| `prepare_resources.py` | 整理 USDA/FNDDS 每 100 g 营养和标准份量 |
| `tools.py` | 名称/状态匹配、营养计算、单项重量归一及局部裁剪 |
| `calibrate_nutrition.py` | 用合法训练集离线生成类别营养密度统计 |
| `harness.py` | 单个有界工具循环，最多一次额外复核 |
| `infer.py` / `evaluate.py` | 单图/批量 Agent 推理、items F1 和五项独立 MSE |

独立重量配置位于 `config/optional/weight.yaml`；Agent 配置 `config/optional/nutrition_agent.yaml` 继承它；名称和食品状态映射位于 `config/optional/nutrition_mapping.yaml`。旧 CatBoost 残差模型、旧联合特征缓存和 `prior` 对照模式不再使用；新产物采用独立 v2 缓存路径及 `weight/ridge.joblib`，不会误加载旧模型。

## 先独立训练重量工具

重量工具的全部步骤均无需 LLM 地址、API 密钥或 Nutrition DB。

使用 `data/processed/meal_json/{train,val}.jsonl`：每行含唯一 `image_id`、`image_path`、`label.total_weight_g`。其他营养标签和菜品名称不参与重量训练。图片根目录由 `image_root` 或 `dataset_info.json` 指定。

`weight.group_field` 默认 `image_id`，在本项目中来自 Nutrition5k 的 `dish_id`。若同盘菜有多视角或相关帧，必须使用同一 `group_id` 并修改配置。训练/开发集不得共享菜品组或重复图像。

在 Python 3.11/3.12 环境安装依赖，将 Hugging Face 的 `facebook/dinov2-small` 权重、config 和 processor 放到配置指定的 `models/dinov2-small`。代码只从本地加载视觉模型，并保持 `eval()`、`requires_grad_(False)` 和 `inference_mode()`。

```powershell
python -m pip install -r requirements.txt
python -m experiments.optional.nutrition_agent.prepare_cache --kind visual --split train
python -m experiments.optional.nutrition_agent.prepare_cache --kind visual --split val
python -m experiments.optional.nutrition_agent.weight --action train
python -m experiments.optional.nutrition_agent.weight --action evaluate
python -m experiments.optional.nutrition_agent.weight --action predict --image path/to/meal.jpg
```

视觉缓存只含 image_id、图片 SHA-256 和 768 维向量，支持续跑。缓存版本只由视觉编码器、processor 和特征定义决定。训练使用默认 5 折 GroupKFold，搜索配置中的 `ridge_alphas`；每折独立拟合 StandardScaler 与 Ridge。参考 [GroupKFold 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)。

输出 `results/nutrition_agent/weight/ridge.joblib`、`cv_metrics.json`、`oof_predictions.jsonl`、`val_metrics.json`、`val_predictions.jsonl`。报告 OOF 选模 MSE、独立开发集 MSE/MAE，并用仅由训练标签计算的平均重量作为简单基线。OOF 用于选 alpha，不能代替独立开发集最终表现。训练后无需训练图片或 manifest 即可独立单图预测，只需视觉模型、Ridge 产物与配置。

## 营养数据库与类别统计

```powershell
python -m experiments.optional.nutrition_agent.prepare_resources --download
python -m experiments.optional.nutrition_agent.tools --food "米饭" --method boiled --state cooked --uncalibrated
```

使用固定版本 SR Legacy / FNDDS，保留标准份量和完整营养字段；缺失标准份量为 null，缺失营养值不填零。原始压缩包可手动放到 `resources.sources[].path` 再省略 `--download`。参考 [FoodData Central 字段说明](https://fdc.nal.usda.gov/portal-data/external/dataDictionary) 和 [USDA 下载页](https://fdc.nal.usda.gov/download-datasets/)。

混合菜优先匹配成品菜，独立食物分别查表。明确冲突的生熟、烹饪和酱汁状态被排除，未知状态保留近似匹配标识。低密度差异采用最匹配候选，高差异由 LLM 看图选择。没有合适成品菜时显式失败并要求完善映射；本版不做原料重建或另加已计入成品菜的油/酱汁。标准份量仅供 LLM 分配各项重量，不输入重量模型。

当前默认 `nutrition.use_calibration: false`，直接使用原始 USDA，跳过下面的语义缓存和类别统计。类别营养密度统计是可选的独立功能；需要时先将该选项设为 true，再执行下面的命令。它不读取 DINOv2、Ridge 或纯视觉缓存：

```powershell
$env:OPENAI_BASE_URL = "你的 OpenAI 兼容接口地址"
$env:OPENAI_API_KEY = "你的 API 密钥"
python -m experiments.optional.nutrition_agent.prepare_cache --kind nutrition --split train
python -m experiments.optional.nutrition_agent.calibrate_nutrition
```

语义缓存正常每图一次模型请求，只获取菜品/烹饪观察，不要求 LLM 估总重量。主模型 ID 必须与实际支持 RGB 和 tool calling 的服务一致。改变 LLM、Prompt、营养表或映射后应重建此语义缓存；不影响纯视觉缓存。

整餐营养标签不能唯一确定混合餐的各原料密度，因此仅使用真实单项名称与语义类别对齐的单项餐，并按烹饪/生熟/酱汁区分类别。每组只计一次，`nutrition.group_field` 应与真实菜品分组一致。达到最少样本数才按总营养/总质量生成密度统计；其余类别使用原始 USDA 密度。统计可能覆盖零个类别，不能声称已完成有效校准。此步骤只用 train；配置中的混合强度在开发集确定，最终测试前冻结。如只使用原始数据库，可将 `nutrition.use_calibration` 设为 false 并跳过语义缓存和统计。

两个工具也可分别运行。Nutrition DB 的 `--items` 接收 JSON 数组，每项含 `food_id`、`name`、`weight_g` 和可选的 `calibration_key`：

```powershell
python -m experiments.optional.nutrition_agent.tools --items path/to/items_with_weight.json
```

## Agent Loop 与评测

1. LLM 观察整图，调用 `lookup_nutrition` 获取菜品候选、营养密度和标准份量。
2. 调用无参数 `estimate_weight`：宿主绑定原始 RGB 图片，独立预测总重，拒绝语义参数。
3. 有可能改变食品匹配的歧义时，最多额外复核一次：裁剪后重新查表，或直接纠正识别重新查表。总重量保持不变，不对裁剪图重新估重。
4. LLM 调用 `calculate`，按观察顺序选择食品代码并给出正的相对重量；程序归一到预测总重，再按密度计算营养并输出 JSON。

正常三次模型请求，有裁剪复核时最多五次；无多 Agent、无 LangGraph。最终 JSON 包含 `items`、`total_weight_g`、`total_calories_kcal`、`protein_g`、`fat_g`、`carbohydrate_g`。模型自由文本不能覆盖工具的计算结果。

保留 `direct`（LLM 直接生成）与 `agent`（独立重量工具 + 营养库）两组对照。单独重量评估不需要运行主 Agent。

```powershell
python -m experiments.optional.nutrition_agent.infer --image path/to/meal.jpg --mode agent
foreach ($mode in @('direct', 'agent')) {
    python -m experiments.optional.nutrition_agent.infer --manifest data/processed/meal_json/val.jsonl --mode $mode
    if ($LASTEXITCODE -ne 0) { break }
}
python -m experiments.optional.nutrition_agent.evaluate --predictions results/nutrition_agent/direct/predictions.jsonl results/nutrition_agent/agent/predictions.jsonl --manifest data/processed/meal_json/val.jsonl --output results/nutrition_agent/val_comparison.json
```

分别报告 items micro/macro F1 和重量、热量、蛋白质、脂肪、碳水五项 MSE，同时报告 MAE、numeric coverage、schema valid rate、失败率和重量求和误差。失败不补零，完整 manifest 中缺失预测计入覆盖率分母。最终测试前冻结重量模型、营养库、类别统计、映射、Prompt 和配置；不提交数据、缓存或模型文件。

本地已有的 USDA 表含 13,225 条记录；其中 12,958 条具有标准份量，13,224 条具有完整四项营养密度。代码检查和工具调用检查不等于正式实验：在准备好课程训练 manifest、DINOv2-Small 权重和 LLM 服务前，不生成或声称完整实验精度。

精简运行命令和最终测试流程见 [选做任务运行命令](optional_server_commands.md#3-任务三营养-agent)。重量测试使用 `--split test`，保存独立的 `test_metrics.json` 和 `test_predictions.jsonl`。
