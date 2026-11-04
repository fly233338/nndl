# 环境与数据准备

以下在 Linux 服务器项目根目录执行。已有可运行环境时跳过安装，日常命令统一见 [实验运行命令](experiment_commands.md)。

## 1. 首次安装

下面保留项目原有 B200 环境使用的 PyTorch 2.8 / CUDA 12.8 组合。

```bash
conda create -n lens python=3.11 -y
conda activate lens
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt "transformers>=4.57,<5"
conda install -c conda-forge openjdk=8 -y
```

完整 SPICE 评测还需要任务一使用的 Stanford CoreNLP 资源。

## 2. 数据与模型

课程数据放在下列目录，保留原始 train/val/test 划分和全部标签：

```text
models/annotations/{train,val,test}.jsonl
models/images/{train,val,test}/
models/best.pt
models/vit-base-patch16-224/
```

原始标注含 `dish_id`、`image` 和 `target`；`image` 已带 `images/` 前缀，因此 `config/data.yaml` 的 `meal_json.image_root` 默认为 `models`。baseline 需要任务一的 `models/best.pt`；Qwen 需要完整的本地模型、tokenizer 和 processor 目录。

```bash
python -m experiments.task2_meal_json.prepare_meal_json
```

输出为 `data/processed/meal_json/{train,val,test}.jsonl` 和 `dataset_info.json`。任务二参数集中在 `config/task2_meal_json/meal_json.yaml`；消融命令见 [参数消融](task2_tuning.md)，选做任务见 [选做命令](optional_server_commands.md)。
