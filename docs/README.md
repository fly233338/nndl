# FoodLens

NN&DL 课程设计：两个必选任务为食物图像描述和膳食 JSON 生成，选做为强化学习优化和 LLM-as-Judge。

## 代码目录

```text
experiments/
  task1_captioning/       # 五种基线、ViT/SigLIP 四组架构消融
  task2_meal_json/        # SigLIP 迁移、Qwen 辅助监督和结构化评测
  optional/
    caption_rl/          # CIDEr-D SCST 微调与 CE 配对评测
    llm_judge/           # 图文评分与一致性分析
config/
  data.yaml              # 课程数据路径
  task1_captioning/      # 任务一参数
  task2_meal_json/       # 任务二参数和消融配置
  optional/             # RL 和 Judge 参数
utils/                   # 配置、文件、JSONL、指标和随机种子
```

## 运行入口

- [环境与数据准备](task2_training.md)
- [必选任务运行命令](experiment_commands.md)
- [任务一首次下载](task1_commands.md)
- [任务一四组架构消融](task1_improvements.md)
- [任务二辅助监督对照](task2_tuning.md)
- [RL 和 Judge 运行命令](optional_server_commands.md)

命令从项目根目录执行，路径和实验参数集中在配置中。各实验保持独立，数据、模型权重、缓存和实验结果不提交到仓库。结果需同时报告预测误差、结构有效率和数值覆盖率。

新训练按视觉模型配置保存归一化参数，历史检查点按原有预处理加载。任务一固定 M Decoder，对比 ViT/SigLIP × GELU/SwiGLU；任务二包含 SigLIP 迁移和 Qwen 三层辅助监督，全部按生成 val micro-F1 选型。辅助头只参与训练。

```bash
python run.py list task1
python run.py train task1 --group ablation
python run.py train task2 --group ablation
python run.py summarize task2
```

默认只训练和验证，测试集须显式 `evaluate --split test`。本次改动仅静态检查，未运行模型训练、推理或 GPU 测试。
