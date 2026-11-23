# FoodLens 统一实验入口

从项目根目录运行 `python run.py <操作> <任务>`。日常仅使用这一入口；模型实现仍位于各任务模块中。下方历史入口用于复现旧训练和两阶段结果。

## 配置与实验组

| 任务 | 组 | 实验 |
| --- | --- | --- |
| task1 | baseline | cnn_gru、attention_rnn、transformer、graph_transformer、vit_transformer |
| task1 | ablation | vit_gelu、vit_swiglu、siglip_gelu、siglip_swiglu |
| task2 | comparison | siglip_transfer、qwen_lora |
| task2 | ablation | qwen_lora、qwen_lora_item_aux、qwen_lora_item_numeric_aux |

参数分别放在 `config/task1_captioning/baseline.yaml`、`config/task1_captioning/ablation.yaml` 和 `config/task2_meal_json/ablation.yaml`。配置同时保存实验说明、后端、分组、别名和结果根目录。任务二的 qwen_lora 在两个组中指向同一个目录，不重复训练。

```bash
python run.py list task1
python run.py list task2
python run.py train task1 --group ablation --dry-run
python run.py train task1 --group baseline
python run.py train task1 --group ablation
python run.py train task2 --group comparison
python run.py train task2 --group ablation
python run.py train task2 --exp qwen_lora qwen_lora_item_aux --seeds 2026 2027 2028
```

train 必须选择 `--group` 或 `--exp`，二者互斥。每组顺序执行，失败即停止。默认只训练与验证；模型选择仍为任务一 val CIDEr-D、任务二生成 val micro-F1。原五模型保留自身原训练配方和 min_delta 选型逻辑，不能与新消融混称同配方实验。

## 恢复与参数覆盖

```bash
python run.py train task2 --group ablation --resume
python run.py train task1 --exp siglip_swiglu --batch-size 4 --accumulation 16
python run.py train task2 --exp siglip_transfer --caption-checkpoint results/captioning_siglip/siglip_swiglu/seed_2026/best.pt
python run.py train task2 --exp qwen_lora --resume results/meal_json/strong/qwen_base/seed_2026/last.pt
```

`--resume` 自动恢复所选实验已有 last，其余未开始组正常开始；完成组跳过。也可以显式给出旧目录的 last 路径，只能选择一个实验/种子。旧基线缺少完整随机状态恢复，统一入口提前拒绝 baseline 的 `--resume`。

单实验支持 `--model-path`、`--batch-size`、`--accumulation`，模型路径须指向相同模型的本地副本。原五模型请修改 baseline YAML，不支持上述命令行覆盖。`--config` 指定一份实验配置，`--output-dir` 覆盖结果根目录；训练配置与检查点禁止直接覆盖已有运行。

## 评测、预测与汇总

```bash
python run.py evaluate task1 --group ablation --split val
python run.py evaluate task2 --group comparison --split test
python run.py evaluate task2 --exp qwen_lora_item_aux --batch-size 2
python run.py predict task2 --checkpoint results/meal_json/strong/qwen_lora/seed_2026/best.pt --image path/to/food.jpg
python run.py summarize task1
python run.py summarize task2
```

evaluate 默认 val，按实验定位 best；test 必须显式指定。predict 必须给检查点和单张图片。完整文本指标需要 Java 和 SPICE 资源，详见[环境准备](task2_training.md)。

也可以直接评测历史检查点：

```bash
python run.py evaluate task1 --checkpoint results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt --split test
python run.py evaluate task1 --checkpoint results/captioning/vit_transformer/best.pt --backend caption_baseline --manifest data/processed/food500cap/test.jsonl --image-root data/processed/food500cap/images --split test
```

task1 根据旁边的 run_config.json 区分旧基线与新消融；缺少该文件时显式指定 `--backend caption_baseline` 或 `caption_ablation`。task2 的检查点模式用于新 strong_meal 格式，历史 Adapter 使用下方原入口。`--manifest`、`--image-root` 可覆盖评测数据位置。

新运行目录统一为 `<output_dir>/<experiment>/seed_<seed>/`，组名不进入目录。原基线以前的无 seed 目录以及已训练的旧名字目录均不移动；通过显式检查点评测/恢复，不自动合并进新结果。

summarize 默认汇总该任务所有已登记的新运行，支持按组或实验筛选。每个结果根目录输出 summary.csv/summary.md，标注分组与 recipe，按实际目录去重，缺失值不填零。训练或组评测结束后自动汇总该任务当前种子的已登记实验；测试指标单列，不参与选型。

所有操作支持 `--dry-run`：只打印解析结果和将执行的命令，既不创建结果目录，也不加载模型。`qwen_base/qwen_items/qwen_items_numeric/m_swiglu_aug` 仍接受为实验别名，但新运行使用规范名称。旧配置文件只是指向新配置的兼容文件，参数只维护一份。

任务一架构细节见[四组消融](task1_improvements.md)，任务二损失与迁移细节见[辅助监督](task2_tuning.md)。

## 数据准备

```bash
conda activate lens
export CUDA_VISIBLE_DEVICES=0
```

任务一：本地需有 Food500Cap Parquet、`models/resnet-50` 和 `models/vit-base-patch16-224`。

```bash
python -m experiments.task1_captioning.prepare_food500cap_hf --dataset-dir data/raw/Food500Cap
```

任务二使用课程标注与图片；先在 `config/data.yaml` 确认路径。

```bash
python -m experiments.task2_meal_json.prepare_meal_json
```

## 历史复现：原任务一入口

训练配置中的五个模型；只跑一个时追加 `--model vit_transformer`。

```bash
python -m experiments.task1_captioning.train
```

统一测试 greedy 和 beam，图片根目录应与 `dataset_info.json` 一致。

```bash
for model in cnn_gru attention_rnn transformer graph_transformer vit_transformer; do
  for decode in greedy beam; do
    python -m experiments.task1_captioning.evaluate --checkpoint "results/captioning/$model/best.pt" --manifest data/processed/food500cap/test.jsonl --image-root data/processed/food500cap/images --decode "$decode" --beam-size 3 --output "results/captioning/$model/test/$decode/metrics.json" || break 2
  done
done
```

任务一四组架构对照及归一化说明见 [任务一提分方案](task1_improvements.md)。上述五种结构仍作为基线保留。

## 历史复现：原任务二入口

先将两个变量改为服务器实际的完整模型目录。baseline 还需要 `models/best.pt` 和 `models/vit-base-patch16-224`。

```bash
qwen3=models/Qwen3-VL-4B-Instruct
qwen25=models/Qwen2.5-VL-3B-Instruct
python -m experiments.task2_meal_json.train_baseline
python -m experiments.task2_meal_json.train_qwen --model-name "$qwen3"
python -m experiments.task2_meal_json.train_qwen --model-name "$qwen25"
```

baseline 推理与评测：

```bash
for decode in greedy beam; do
  out="results/meal_json/default/baseline_vit_transformer/test_$decode"
  python -m experiments.task2_meal_json.infer_baseline --checkpoint results/meal_json/default/baseline_vit_transformer/best.pt --manifest data/processed/meal_json/test.jsonl --image-root models --decode "$decode" --beam-size 3 --output "$out.jsonl" || break
  python -m experiments.task2_meal_json.evaluate --predictions "$out.jsonl" --output "${out}_metrics.json" || break
done
```

Qwen3 的两种推理方式复用同一个 Adapter：

```bash
for mode in direct two_stage; do
  out="results/meal_json/default/Qwen3-VL-4B-Instruct/test_$mode"
  python -m experiments.task2_meal_json.infer --adapter-dir results/meal_json/default/Qwen3-VL-4B-Instruct/shared_adapter --base-model-name "$qwen3" --manifest data/processed/meal_json/test.jsonl --image-root models --mode "$mode" --output "$out.jsonl" || break
  python -m experiments.task2_meal_json.evaluate --predictions "$out.jsonl" --output "${out}_metrics.json" || break
done
```

Qwen2.5 对照：

```bash
python -m experiments.task2_meal_json.infer --adapter-dir results/meal_json/default/Qwen2.5-VL-3B-Instruct/shared_adapter --base-model-name "$qwen25" --manifest data/processed/meal_json/test.jsonl --image-root models --mode direct --output results/meal_json/default/Qwen2.5-VL-3B-Instruct/test_direct.jsonl
python -m experiments.task2_meal_json.evaluate --predictions results/meal_json/default/Qwen2.5-VL-3B-Instruct/test_direct.jsonl --output results/meal_json/default/Qwen2.5-VL-3B-Instruct/test_direct_metrics.json
```

Qwen zero-shot 对照：不训练、不加载 Adapter，使用原始 Qwen3/Qwen2.5 模型。提示词、图片尺寸、解码及评测流程与上面的 LoRA 实验相同；结果单独保存在配置的 `zero_shot_output_dir`。

```bash
for base_model in "$qwen3" "$qwen25"; do
  model=$(basename "$base_model")
  out="results/meal_json/zero_shot/$model/direct/test_predictions"
  python -m experiments.task2_meal_json.infer --zero-shot --base-model-name "$base_model" --manifest data/processed/meal_json/test.jsonl --image-root models --mode direct --output "$out.jsonl" || break
  python -m experiments.task2_meal_json.evaluate --predictions "$out.jsonl" --output "${out}_metrics.json" || break
done
```

`--zero-shot` 与 `--adapter-dir` 互斥；不指定 `--output` 时按原始模型名称、推理模式和划分自动保存到独立目录。已有任务二汇总未包含这组结果，需实际推理和评测后再补表。

参数消融见 [消融命令](task2_tuning.md)。汇总时保留食物 F1、各字段 MSE、数值覆盖率和 Schema 有效率。
