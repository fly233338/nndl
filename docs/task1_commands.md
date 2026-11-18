# 任务一命令入口

日常数据准备、训练和测试统一见 [实验运行命令](experiment_commands.md#2-任务一图像描述)。

首次下载数据和视觉模型，在联网环境执行：

```bash
hf download advancedcv/Food500Cap --repo-type dataset --local-dir data/raw/Food500Cap --include "data/*.parquet"
hf download microsoft/resnet-50 --local-dir models/resnet-50
hf download google/vit-base-patch16-224 --local-dir models/vit-base-patch16-224
```

离线服务器直接复制上述目录。训练参数在 `config/task1_captioning/captioning.yaml`，环境安装见 [环境与数据准备](task2_training.md)。

独立 SigLIP 2 Giant + 扩容 Transformer 的配置、五组对照、512 分辨率和 LoRA 命令见 [任务一提分方案](task1_improvements.md)。默认入口为 `python -m experiments.task1_captioning.strong_captioner train`，与原五模型入口独立。

原 ViT 对照仍使用 `config/task1_captioning/improvements/`。新训练会读取本地视觉模型的预处理配置，将归一化参数保存到检查点；不要直接修改旧检查点推理时的归一化方式。
