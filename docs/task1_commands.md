# 任务一命令入口

日常数据准备、训练和测试统一见 [实验运行命令](experiment_commands.md#2-任务一图像描述)。

首次下载数据和视觉模型，在联网环境执行：

```bash
hf download advancedcv/Food500Cap --repo-type dataset --local-dir data/raw/Food500Cap --include "data/*.parquet"
hf download microsoft/resnet-50 --local-dir models/resnet-50
hf download google/vit-base-patch16-224 --local-dir models/vit-base-patch16-224
```

离线服务器直接复制上述目录。训练参数在 `config/task1_captioning/captioning.yaml`，环境安装见 [环境与数据准备](task2_training.md)。
