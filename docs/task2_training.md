# Task 2：Nutrition5k 服务器训练

以下命令在 Linux 服务器的项目根目录 `/public/home/rxa-jj-wzk/projects/fyl/lens` 执行。数据使用教学云下载的 `annotations.zip` 和 `images.zip`，保留原始 train/val/test 划分和全部标签，不重新随机划分，不修改重量和营养值。

```bash
cd /public/home/rxa-jj-wzk/projects/fyl/lens
```

## 0. 新服务器环境

已有可用的 Task 1 环境可以继续使用。新建环境时，下面以 NVIDIA 驱动支持 CUDA 12.4 为例；驱动不匹配时，按 PyTorch 官方安装说明选择对应的 GPU 安装包。

```bash
conda create -n foodlens_task2 python=3.11 -y
conda activate foodlens_task2
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt "transformers>=4.57,<5"
nvidia-smi
python -c "import torch, transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
```

CUDA 应显示 `True`。完整运行 METEOR、SPICE 评测还需要 Java：

```bash
conda install -c conda-forge openjdk=8 -y
```

## 1. 放置数据与模型

数据和 Task 1 模型统一放在项目的 `models/` 目录。若数据还是压缩包，执行以下命令解压；已经有 `models/annotations/` 和 `models/images/` 则跳过：

```bash
unzip -n -q models/annotations.zip -d models
unzip -n -q models/images.zip -d models
```

已解压到下列结构则跳过解压：

```text
models/
  annotations/
    train.jsonl                 # 2338 条
    val.jsonl                   # 414 条
    test.jsonl                  # 506 条
  images/
    train/
    val/
    test/
```

baseline 还需要服务器上已有 Task 1 的模型：

```text
models/best.pt
models/vit-base-patch16-224/
```

`vit_path` 必须指向 Task 1 使用的同一份 Hugging Face ViT backbone。

Qwen 模型通过命令行 `--model-name` 指定，使用以下服务器本地路径，目录中需要有完整权重、config、tokenizer 和 processor 文件：

```text
/public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct/
/public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct/
```

`--model-name` 会覆盖配置中的模型名称，不需要改 Qwen 配置文件。仅运行 Qwen 不需要 Task 1 的 ViT 和 checkpoint。

内网服务器训练前启用离线模式：

```bash
export HF_HUB_OFFLINE=1
```

## 2. 转换数据

`config/data.yaml` 已设置为：

```yaml
meal_json:
  annotations_dir: models/annotations
  image_root: models
  output_dir: data/processed/meal_json
```

注意 `image_root` 是 `annotations/` 和 `images/` 的共同父目录，也就是项目里的 `models/`，因为原始标注的图片路径已经带有 `images/train/...` 前缀。若数据实际解压在其它子目录，只修改这两个输入路径。

```bash
python -m scripts.prepare_meal_json --config config/data.yaml
```

预期打印：

```text
train: 2338 rows
val: 414 rows
test: 506 rows
```

输出到 `data/processed/meal_json/`，包含三个 JSONL 文件及 `dataset_info.json`。仅将 `dish_id/image/target` 重命名为训练代码需要的 `image_id/image_path/label`，其余信息保留，记录顺序和标签内容不变。

原始食材重量之和可能不等于总重量，因此不再将重量差异视为 Schema 错误；`weight_consistency_rate` 仍单独统计，不能用它过滤或修正标签。

## 3. 开始训练任务二

baseline 配置已写好：`max_length: 768`、`epochs: 10`、`batch_size: 16`、`num_workers: 4`、`learning_rate: 0.0001`、`freeze_encoder: false`。训练和验证均使用 4 个数据加载进程。768 覆盖已检查的训练和验证标签长度，包含 BOS/EOS。

Task 2 有四组实验，但只训练三次：

| 方法 | 训练与推理 |
|---|---|
| A：自研 ViT-Transformer JSON baseline | 从 Task 1 best.pt 迁移后微调 |
| B：Qwen3-VL 直接生成 | 训练 Qwen3 LoRA，direct 推理 |
| C：Qwen3-VL 两阶段生成 | 复用 B 的 Adapter，two_stage 推理 |
| D：Qwen2.5-VL 对照 | 单独训练 Qwen2.5 LoRA，direct 推理 |

只启动 baseline：

```bash
python -m experiments.meal_json.train_baseline --config config/meal_json.yaml
```

需要依次完成任务二全部训练时，选择一张空闲 GPU，运行下面的组合命令；前一项成功后才运行下一项：

```bash
export CUDA_VISIBLE_DEVICES=0
python -m experiments.meal_json.train_baseline --config config/meal_json.yaml && python -m experiments.meal_json.train_qwen --config config/meal_json.yaml --model-name /public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct && python -m experiments.meal_json.train_qwen --config config/meal_json.yaml --model-name /public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct
```

Qwen3 和 Qwen2.5 共用 `config/meal_json.yaml` 中的 `qwen` 参数，当前为：

```yaml
qwen:
  max_length: 1024
  epochs: 2
  gradient_checkpointing: true
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.00002
  image_size: 448
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

上面是现有 `qwen` 配置的参数摘录，不要删掉配置中的 `model_name` 等字段。Qwen 的 1024 是图像 token、提示词和答案的总序列长度，独立于 baseline 的 768；梯度检查点已开启，并在训练时关闭 KV cache。单卡有效 batch size 为 8，支持 BF16 时自动使用 BF16，否则 GPU 上使用 FP16，不量化。

只训练 Qwen3：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.meal_json.train_qwen --config config/meal_json.yaml --model-name /public/home/rxa-jj-wzk/models/Qwen3-VL-Instruct/Qwen3-VL-4B-Instruct
```

只训练 Qwen2.5：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.meal_json.train_qwen --config config/meal_json.yaml --model-name /public/home/rxa-jj-wzk/models/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct
```

正式训练期间不要修改配置文件。baseline 启动会输出加载的兼容参数数量；词表 embedding 和输出层重新初始化。

模型输出：

```text
results/meal_json/baseline_vit_transformer/best.pt
results/meal_json/Qwen3-VL-4B-Instruct/shared_adapter/
results/meal_json/Qwen2.5-VL-3B-Instruct/shared_adapter/
```

baseline 每次启动都会从 Task 1 权重重新微调，不是接着上次 Task 2 checkpoint 训练。

## 4. 四组推理与评测

读取数据准备时保存的图片根目录：

```bash
image_root=$(python -c 'import json; print(json.load(open("data/processed/meal_json/dataset_info.json"))["image_root"])')
```

方法 A：

```bash
python -m experiments.meal_json.infer_baseline \
  --checkpoint results/meal_json/baseline_vit_transformer/best.pt \
  --manifest data/processed/meal_json/test.jsonl \
  --image-root "$image_root" \
  --output results/meal_json/baseline_vit_transformer/test_predictions.jsonl \
  --decode beam --beam-size 3
```

方法 B/C，共用同一个 Qwen3 Adapter：

```bash
for mode in direct two_stage; do
  python -m experiments.meal_json.infer \
    --adapter-dir results/meal_json/Qwen3-VL-4B-Instruct/shared_adapter \
    --manifest data/processed/meal_json/test.jsonl \
    --image-root "$image_root" \
    --mode "$mode" \
    --output "results/meal_json/Qwen3-VL-4B-Instruct/${mode}.jsonl"
done
```

方法 D：

```bash
python -m experiments.meal_json.infer \
  --adapter-dir results/meal_json/Qwen2.5-VL-3B-Instruct/shared_adapter \
  --manifest data/processed/meal_json/test.jsonl \
  --image-root "$image_root" \
  --mode direct \
  --output results/meal_json/Qwen2.5-VL-3B-Instruct/direct.jsonl
```

分别评测四份预测：

```bash
for predictions in \
  results/meal_json/baseline_vit_transformer/test_predictions.jsonl \
  results/meal_json/Qwen3-VL-4B-Instruct/direct.jsonl \
  results/meal_json/Qwen3-VL-4B-Instruct/two_stage.jsonl \
  results/meal_json/Qwen2.5-VL-3B-Instruct/direct.jsonl; do
  python -m experiments.meal_json.evaluate \
    --predictions "$predictions" \
    --output "${predictions%.jsonl}_evaluation.json"
done
```

比较食物 F1、文本指标与数值误差，最终表格同时列出各字段 MSE、numeric coverage 和整体 schema valid rate。重量一致性仅作为额外观察指标。测试集仅用于最终评测，不用于调整训练参数。

## 5. 内网代码同步的固定基点

后续增量 bundle 始终以 `355ed581a8742faba02dffb4a61e406aa1d39708` 为基点，包含从该提交到最新 `main` 的全部更新。该值是打包基点，不是将当前分支回退到旧提交。服务器需要已导入包含该提交的首次完整 bundle。

本地提交修改后，在项目根目录执行 PowerShell 命令：

```powershell
git bundle create ..\foodlens-update.bundle 355ed581a8742faba02dffb4a61e406aa1d39708..main
git bundle verify ..\foodlens-update.bundle
```

将增量包传到服务器项目目录的 `foodlens-update.bundle`，然后执行：

```bash
cd /public/home/rxa-jj-wzk/projects/fyl/lens
git fetch ./foodlens-update.bundle refs/heads/main:refs/remotes/offline/main
git merge --ff-only refs/remotes/offline/main
python -m scripts.prepare_meal_json --config config/data.yaml
```
