# 任务二可配置对照

入口：`python run.py`；配置：`config/task2_meal_json/ablation.yaml`。所有组按真实生成的 val 食材 micro-F1 保存最佳模型，同分保留更早模型。课程仅规定 F1、MSE 等指标，没有指定唯一主指标；micro-F1 是本实验的选型约定。

## 实验与损失

| 预设 | 模型 | 食材辅助头 | 数值辅助头 |
| --- | --- | --- | --- |
| qwen_lora | Qwen3-VL-4B + LoRA | 关 | 关 |
| qwen_lora_item_aux | 同上 | 开 | 关 |
| qwen_lora_item_numeric_aux | 同上 | 开 | 开 |
| siglip_transfer | 任务一 SigLIP + 自建 Decoder 迁移 | 关 | 关 |

三组 Qwen 都只训练完整 JSON 生成，预算和增强一致，不沿用旧入口的 direct/items/nutrients 三任务展开。保留冻结的原视觉主干及 merger；LoRA 只插入语言层 attention 和 MLP，rank=32、alpha=64、dropout=0.05。BF16、梯度检查点、micro-batch=1、累积=16、AdamW lr=5e-5、warmup=5%+cosine、最多 6 轮，连续 3 轮无 micro-F1 改善早停。

辅助头从最后一个有效提示词位置取语言表示，位于真实答案之前，因果掩码禁止读取答案。食材头为 LayerNorm + Linear(512) + GELU + Linear(K)，监督训练集名称集合的多标签 BCE，权重 0.2；数值头结构相同，输出五个 softplus 非负数值，以训练集字段标准差（下限 1）归一化误差后求 MSE，权重 0.1。三项损失按样本平均；辅助头梯度回传到语言 LoRA。食材头的 0.5 阈值只作辅助训练诊断，不用于最终生成筛选。

推理只使用原生成分支，不调用辅助头替换、过滤或修正 JSON。辅助 F1/MSE 与最终生成指标分开记录。

图像保留宽高比，官方 processor 像素预算约 448²；增强仅训练时使用翻转、小角度旋转和轻微亮度/对比度变化，不裁掉食物。文本总长度 2048，生成上限 768；超过长度报错，不静默截断 JSON。默认 worker=0 便于精确恢复随机增强；增加 worker 数后应留意跨进程数据处理成本。

## 路径与迁移

设置 YAML 中本地 Qwen 目录、数据目录及 `image_root`。默认数据沿用课程划分。SigLIP 迁移起点默认为已训练的 `results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt`；也可指定新实验的最佳检查点。

迁移投影、注意力、SwiGLU 和归一化参数，冻结 SigLIP；重新建立训练集字符词表、输入输出共享嵌入，将文字位置扩展至 1024，原位置复制、新位置随机初始化。Decoder lr=5e-5，嵌入/位置 lr=2e-4，有效 batch=16，最多 30 轮、早停 5 轮，默认 greedy。要求源检查点为冻结 SigLIP，无视觉 LoRA；训练恢复及评测只需新检查点和视觉模型，不再依赖原任务一检查点。

## 运行命令

```bash
python run.py list task2
python run.py train task2 --exp qwen_lora
python run.py train task2 --exp qwen_lora_item_aux
python run.py train task2 --exp qwen_lora_item_numeric_aux
python run.py train task2 --exp siglip_transfer --caption-checkpoint results/captioning_siglip/siglip_swiglu/seed_2026/best.pt
python run.py train task2 --group comparison --caption-checkpoint results/captioning_siglip/siglip_swiglu/seed_2026/best.pt
```

`comparison` 比较 SigLIP 迁移与 Qwen；`ablation` 比较三个 Qwen 辅助监督方案，两组共享 qwen_lora。各组顺序执行，默认只训练和验证，不运行 test。可用 `--exp qwen_lora qwen_lora_item_aux` 选择子集。`--seeds 2026 2027 2028` 用于复核；`--output-dir` 切换结果根目录。单组还支持 `--model-path`、`--batch-size`、`--accumulation`。

```bash
python run.py train task2 --exp qwen_lora_item_aux --resume results/meal_json/strong/qwen_lora_item_aux/seed_2026/last.pt
python run.py train task2 --group ablation --resume
python run.py evaluate task2 --checkpoint results/meal_json/strong/qwen_lora_item_aux/seed_2026/best.pt --split val
python run.py evaluate task2 --checkpoint results/meal_json/strong/qwen_lora_item_aux/seed_2026/best.pt --split test
python run.py predict task2 --checkpoint results/meal_json/strong/qwen_lora_item_aux/seed_2026/best.pt --image path/to/food.jpg
python run.py summarize task2
python run.py evaluate task2 --group ablation --split test
```

恢复以原配置、统计量和优化器状态为准，数据哈希必须匹配。已完成组自动跳过；未完成组必须显式恢复；没有 last 的非空目录须更换输出目录。新检查点为项目自定义格式，不使用旧 `infer --adapter-dir` 加载。

## 指标和产物

每轮生成 val 全集，计算名称集合精确匹配的 micro/macro-F1、Precision/Recall、TP/FP/FN、Schema 有效率、重量一致性、五项 MSE/覆盖率。未解析输出按空食材集合参与 F1；MSE 只统计有效数值，覆盖率同时报告。macro 类别为真实与预测名称并集；名称仅转小写及去首尾空白，不改评分口径。

每轮验证不运行 Java 文本评分；独立 `evaluate` 保留完整任务一文本指标，需准备 pycocoevalcap、Java 和 SPICE。文本指标来自食材名称模板，不能用来评价营养数值。

产物位于 `results/meal_json/strong/<experiment>/seed_<seed>/`：best/last、processor、训练集统计量、完整配置、数据哈希、可训练参数及 LoRA 目标、最佳 val 预测与评分、各损失及辅助指标、学习率、耗时和显存。检查点只保存可训练参数和训练状态，冻结主干通过配置引用。套件汇总到 CSV/Markdown，未运行不填零。

旧学习率×轮数网格已删除。原 `train_baseline/train_qwen/infer_baseline/infer` 保留用于历史 ViT/Qwen 和两阶段复现；新实验不通过这些旧训练入口转接。RL 与 Judge 保留独立模块。
