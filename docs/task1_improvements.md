# 任务一：SigLIP 2 Giant + 扩容 Transformer

更新：2026-11-17。已按讨论实现独立实验入口，尚未下载新权重、运行训练、推理或测试；当前没有实测提分、显存和耗时结论。

## 1. 当前实现

入口：`experiments/task1_captioning/strong_captioner.py`。

配置：`config/task1_captioning/strong_captioner.yaml`。所有实验在这一份 YAML 中选择预设，不串联原来的五模型训练配置。

```text
RGB 图片，默认 384×384
  → 冻结的 SigLIP 2 Giant /16 视觉分支
  → 全部 576 个 Patch，每个 1536 维
  → Linear + LayerNorm
  → 自建 Pre-Norm Transformer Decoder
      因果自注意力 + 视觉跨注意力 + SwiGLU
  → 描述
```

主干固定为 `google/siglip2-giant-opt-patch16-384`，默认完全冻结，延续原 ViT 基线的训练边界。不会加载文本塔，不删除首个 Patch，也不只保留池化检索向量。归一化 mean/std 均为 0.5，按官方固定尺寸双线性缩放。[模型配置](https://huggingface.co/google/siglip2-giant-opt-patch16-384/blob/main/config.json)、[预处理配置](https://huggingface.co/google/siglip2-giant-opt-patch16-384/blob/main/preprocessor_config.json)

| 规格 | 层数 | 隐藏维度 | 头数 | SwiGLU 中间维度 | 主体参数估算 |
|---|---:|---:|---:|---:|---:|
| S | 4 | 512 | 8 | 1408 | 约 1700 万 |
| M，默认 | 8 | 768 | 12 | 2048 | 约 7500 万 |
| L | 12 | 1024 | 16 | 2816 | 约 2.04 亿 |

估算不包括视觉投影、词嵌入和少量归一化参数。实际参数量写入运行记录。GELU 对照的 S 规格使用 2048 维 FFN；SwiGLU 对照参数量接近，实际差异需报告。[SwiGLU 研究依据](https://arxiv.org/abs/2002.05202)

共同设置：dropout 0.1、可学习文字位置编码、最终 LayerNorm、最大长度 64、输入词嵌入与输出矩阵共享。训练使用 SDPA 的因果及 padding 掩码；生成缓存视觉 K/V 与逐层文字 K/V，支持批量 greedy/beam。

## 2. 数据、增强与训练配方

沿用已经准备好的 `train/val/test.jsonl`，不重新划分数据。已有 `vocab.json` 优先复用；没有时只从训练描述构建，并保存到本次结果目录，不修改数据目录。训练展开全部参考描述，生成评测每张图像只处理一次，保留全部参考。

增强每轮在线执行：

- 水平翻转概率 0.5；描述含方位词时跳过。
- ±5° 旋转，概率 0.3；扩展画布保留边缘，方位敏感样本跳过。
- 亮度/对比度幅度 0.1，概率 0.3；描述含颜色词时跳过。
- 以同图全部参考描述做关键词保护。关键词并不能覆盖所有语义，后续仍应检查增强样例。
- 验证/推理只有固定缩放及归一化；不加入色调变化、强裁剪、MixUp、CutMix 或遮挡。

图文增强需要保持图像和描述对应，不能直接照搬分类增强。[Augment the Pairs](https://openaccess.thecvf.com/content/WACV2024/html/Yi_Augment_the_Pairs_Semantics-Preserving_Image-Caption_Pair_Augmentation_for_Grounding-Based_Vision_WACV_2024_paper.html)

默认训练设置：

- AdamW，学习率 `2e-4`，betas `(0.9, 0.98)`，eps `1e-8`，weight decay `0.01`。
- bias、归一化和嵌入参数不施加权重衰减；冻结视觉参数不进入优化器。
- 前 5% 优化步数 warmup，之后 cosine，最低学习率为峰值的 1%。
- 交叉熵 teacher forcing，标签平滑 0.05，梯度裁剪 1.0。
- BF16，micro-batch 8，梯度累积 8，有效 batch 64；梯度按实际非 padding 目标词元数归一化，包含末尾不完整累积组。
- 最多 60 轮；每轮计算验证损失与 beam 3 的 CIDEr-D。保存实际最高分检查点，早停以 0.001 的显著改善和耐心 10 轮判断。

主干始终 eval，冻结视觉前向使用 no_grad，投影和解码器保留梯度。L 规格默认启用自建解码器的梯度检查点。显存不足时用 `--batch-size 4 --accumulation 16`，再不足则用 `2 × 32`；这些只是可选运行设置，尚未测量显存。

## 3. 实验预设

| 预设 | 结构与增强 | 目的 |
|---|---|---|
| `s_gelu` | S，GELU，不增强 | SigLIP 下的普通解码器起点 |
| `s_swiglu` | S，SwiGLU，不增强 | FFN 升级 |
| `m_swiglu` | M，SwiGLU，不增强 | 扩大容量 |
| `m_swiglu_aug` | M，SwiGLU，增强 | 默认完整方案 |
| `l_swiglu_aug` | L，SwiGLU，增强 | 进一步扩容 |
| `continue_384` | 选定模型，384，继续 10 轮 | 额外训练预算对照 |
| `continue_512` | 同一起点，512，继续 10 轮 | 分辨率收益 |
| `lora_384` | 同一起点，384，LoRA 10 轮 | 视觉领域适配 |

三个续训预设自动继承来源检查点的解码器规格和增强设置，因此也能继续训练选定的 S/L 模型。它们必须指定 `--init-checkpoint`；各自采用新的优化器和学习率日程，不延长原训练的 cosine 日程。

512 组仍用同一个 Giant 检查点，通过位置编码插值得到 1024 个 Patch。384 与 512 两组从同一检查点分别继续 10 轮，新模块学习率均为 `2e-5`，有效 batch 均为 64。

LoRA 默认关闭，仅作为独立实验：在视觉末 8 层的 Q/V 投影注入 `r=16, alpha=32, dropout=0.05` 的适配器，基础视觉权重仍冻结。LoRA 学习率 `1e-4`，解码器 `2e-5`，micro-batch 2、累积 32。启用时视觉路径允许适配器反向传播，基础主干保持 eval，只有 LoRA dropout 切换训练状态。[PEFT 适配器接口](https://huggingface.co/docs/peft/developer_guides/low_level_api)

## 4. 后续运行命令

下面是用户后续在训练机器执行的命令，本轮均未运行。从项目根目录使用模块入口。

### 环境和权重

训练依赖沿用 `requirements.txt`。数据准备及 Java 指标环境见 [实验运行命令](experiment_commands.md)。默认从项目的 `models/siglip2-giant-opt-patch16-384` 加载本地权重，且 `encoder.local_files_only: true`。先通过 ModelScope 下载完整检查点：

```powershell
python -m pip install -U modelscope
modelscope download --model google/siglip2-giant-opt-patch16-384 --local_dir models/siglip2-giant-opt-patch16-384
```

下载后直接运行训练命令即可，无需再传 `--model-path`。换用其他本地存放位置时，可通过该参数覆盖默认路径。本地目录必须是同一个检查点；镜像仓库可能同时包含双塔权重，训练只加载视觉分支。

### 五组监督训练

```powershell
python -m experiments.task1_captioning.strong_captioner train --experiment s_gelu
python -m experiments.task1_captioning.strong_captioner train --experiment s_swiglu
python -m experiments.task1_captioning.strong_captioner train --experiment m_swiglu
python -m experiments.task1_captioning.strong_captioner train --experiment m_swiglu_aug
python -m experiments.task1_captioning.strong_captioner train --experiment l_swiglu_aug
```

结果默认位于 `results/captioning_siglip/<experiment>/seed_2026/`。用 `--seed 2027`、`--seed 2028` 复核较优配置，避免覆盖原结果。

### 高分辨率与 LoRA 配对

以下以选中 M 为例；若选中 S/L，替换来源检查点路径即可。

```powershell
python -m experiments.task1_captioning.strong_captioner train --experiment continue_384 --init-checkpoint results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt
python -m experiments.task1_captioning.strong_captioner train --experiment continue_512 --init-checkpoint results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt
python -m experiments.task1_captioning.strong_captioner train --experiment lora_384 --init-checkpoint results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt
```

上述续训目录按预设名区分；比较不同来源模型时，增加不同的 `--output-dir`，例如 `results/captioning_siglip_from_l`。不要让不同来源占用同一预设/种子目录。

### 验证、测试和单图生成

```powershell
python -m experiments.task1_captioning.strong_captioner evaluate --checkpoint results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt --split val
python -m experiments.task1_captioning.strong_captioner evaluate --checkpoint results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt --split val --beam-size 1
python -m experiments.task1_captioning.strong_captioner evaluate --checkpoint results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt --split test
python -m experiments.task1_captioning.strong_captioner predict --checkpoint results/captioning_siglip/m_swiglu_aug/seed_2026/best.pt --image path/to/food.jpg
```

先在验证集选型和选择 greedy/beam；测试集仅在最终方案确定后统一评价。`evaluate` 支持 `--manifest`、`--image-root`、`--batch-size`、`--output`，不改变检查点中的图像预处理。

### 断点恢复

```powershell
python -m experiments.task1_captioning.strong_captioner train --experiment m_swiglu_aug --resume results/captioning_siglip/m_swiglu_aug/seed_2026/last.pt
```

恢复时必须保持原实验、种子、训练日程和训练/验证清单。若初次使用了 batch、累积或本地模型参数，恢复时继续提供同样参数。改变总轮数、图像尺寸或 LoRA 设置时使用 `--init-checkpoint` 开始新的续训实验。

## 5. 数据泄漏审计

已经训练的模型无需重训。服务器从项目根目录运行以下命令，读取实际运行配置、元数据、保存的词表和三个数据清单，只使用 CPU，不加载模型：

```powershell
python -m experiments.task1_captioning.audit_data --run-dir results/captioning_siglip/m_swiglu_aug/seed_2026 --output results/captioning_siglip/m_swiglu_aug/seed_2026/data_audit.json
```

优先检查：

- `manifest_identity`：当前 train/val 清单哈希必须与训练时的记录相同，否则不能用现在的清单证明过去那次训练没有重叠。
- `image_id`、`image_sha256`：三个集合两两交集都应为 0，且 `complete` 为 true；缺少哈希字段会标记未完成，不会当作已排除重复。
- `vocabulary.extra_tokens_not_in_train`：默认 min_freq=1 下应为 0。额外词也可能来自旧词表或外部词表，出现时需查来源，不能直接判定读过测试答案。
- `test_evaluation.matches` 与 `count_matches`：应为 true，确认检查的测试清单就是这次评分使用的清单。

可进一步读取真实图片，检查不同文件名、相同 RGB 像素的图像。这比清单审计慢，但仍不使用 GPU：

```powershell
python -m experiments.task1_captioning.audit_data --run-dir results/captioning_siglip/m_swiglu_aug/seed_2026 --pixel-hashes --output results/captioning_siglip/m_swiglu_aug/seed_2026/data_audit_pixels.json
```

数据移动后用 `--data-dir`、`--image-root` 指向原清单和图片的新位置；不同评测文件用 `--metrics` 指定。不要重新生成清单后再拿新清单替旧训练作证明。审计只在指定 `--output` 时写报告，不修改数据、词表或权重。

`status` 为 `passed_checked_items` 表示所列检查通过；`needs_review` 表示有问题待查；`incomplete` 表示证据不全。退出码依次为 0、1、2。这些检查不覆盖近似重复图片、SigLIP 预训练数据重叠，或人为看测试分数反复调参；不能据此宣称绝对不存在任何形式的数据泄漏。

## 6. 记录与分析

每次运行保存 `run_config.json`、`run_metadata.json`、`training_history.json`、`vocab.json`、`best.pt`、`last.pt` 及最佳验证描述。记录实际参数量、可训练视觉参数、输入 Token 数、有效 batch、更新步数、模型版本、数据清单哈希、词元截断数量、学习率、每轮耗时与显存峰值。

检查点只保存自建解码器、投影、可选 LoRA、优化器、调度器和随机状态，不重复保存冻结 SigLIP。加载时仍需访问相同版本的基础模型。恢复支持按已完成轮次继续，不承诺跨硬件和环境逐位复现。

评测复用原指标函数，统一 PTB 分词和原始分数尺度，输出 BLEU-1/2/3/4、METEOR、ROUGE-L、CIDEr-D、SPICE，以及逐图预测、清单/检查点哈希、预处理、解码设置和生成耗时。

报告重点回答：

1. SwiGLU 相比 GELU 的收益是多少？S 两组 FFN 实际参数有多大差异？
2. S → M → L 的收益是否随容量增加继续增长，还是验证先饱和？
3. 增强是否减少过拟合，颜色、方位或多食材样本有无受损？
4. 512 的收益能否超过同预算 384 续训，是否集中在小配料？
5. LoRA 是否超过冻结继续训练，是否引入新的错识别和幻觉？
6. CIDEr 提升时，SPICE、真实食材描述与错误案例是否同步改善？较优组多个种子的波动有多大？

历史测试集已经被探索，报告如实说明。既有五模型和归一化修正对照仍保留；原 `improvements/` 配置只服务于原 ViT 入口，不用于新的 SigLIP 实验。不得直接比较不同训练配方下的历史分差并全部归因于编码器。
