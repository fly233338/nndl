# FoodLens

NN&DL 课程设计：两个必选任务为食物图像描述和膳食 JSON 生成，选做为强化学习优化和 LLM-as-Judge。

## 代码目录

```text
experiments/
  task1_captioning/       # 五种基线、独立 SigLIP 强模型、训练与评测
  task2_meal_json/        # JSON baseline、Qwen LoRA 和结构化评测
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
- [任务一 SigLIP + 扩容 Transformer 实验](task1_improvements.md)
- [任务二参数消融](task2_tuning.md)
- [RL 和 Judge 运行命令](optional_server_commands.md)

命令从项目根目录执行，路径和实验参数集中在配置中。各实验保持独立，数据、模型权重、缓存和实验结果不提交到仓库。结果需同时报告预测误差、结构有效率和数值覆盖率。

新训练按视觉模型配置保存归一化参数，历史检查点按原有预处理加载。SigLIP 2 Giant 的独立实验入口已实现，默认冻结视觉主干，提供 SwiGLU、S/M/L 对照、512 分辨率续训和可选 LoRA；尚未下载权重或运行新实验。
