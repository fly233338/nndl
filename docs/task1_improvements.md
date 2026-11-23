# 任务一架构消融

入口：`python run.py`；配置：`config/task1_captioning/ablation.yaml`。旧五模型基线保持独立。

## 四组配置

| 预设 | 冻结编码器 | 前馈层 / 中间维度 |
| --- | --- | --- |
| vit_gelu | google/vit-base-patch16-224 | GELU / 3072 |
| vit_swiglu | google/vit-base-patch16-224 | SwiGLU / 2048 |
| siglip_gelu | google/siglip2-giant-opt-patch16-384 | GELU / 3072 |
| siglip_swiglu | google/siglip2-giant-opt-patch16-384 | SwiGLU / 2048 |

所有组使用同一个 8 层、768 维、12 头的 Pre-Norm Decoder，长度 64，dropout 0.1，输入输出权重共享。GELU 与 SwiGLU 的前馈参数量大致相当。均使用 384 输入，ViT 插值位置编码并删除 CLS，SigLIP 保留全部 576 个 patch。训练前读取对应模型官方 mean/std 和插值方式，写入运行配置；旧检查点评测保持原预处理。

增强、AdamW、warmup/cosine、标签平滑、有效 batch 64、最大 60 轮、早停和 beam=3 在各组保持一致。按 val CIDEr-D 严格提高保存 best，同分保留更早模型。这里比较的是不同预训练编码器整体，不将收益单独归因于预训练目标、架构或参数量。

## 准备与运行

本地模型目录在 YAML 配置；SigLIP 可通过 ModelScope 下载：

```bash
modelscope download --model google/siglip2-giant-opt-patch16-384 --local_dir models/siglip2-giant-opt-patch16-384
hf download google/vit-base-patch16-224 --local-dir models/vit-base-patch16-224
python run.py list task1 --group ablation
python run.py train task1 --exp vit_gelu
python run.py train task1 --group ablation
```

训练消融从预训练编码器和新 Decoder 开始，不从已经训练的完整模型删除模块继续训练。当前已完成的 `m_swiglu_aug` 结果保留作历史参考；旧名字映射到 `siglip_swiglu`，不会新建同名重复预设。只有配置、数据哈希、预处理和训练预算一致时才能复用历史结果，套件不自动将其冒充新实验。

输出：`results/captioning_siglip/<experiment>/seed_2026/`。显存不足可保持有效 batch 不变：

```bash
python run.py train task1 --exp siglip_swiglu --batch-size 4 --accumulation 16
python run.py train task1 --exp vit_gelu --resume results/captioning_siglip/vit_gelu/seed_2026/last.pt
python run.py train task1 --group ablation --resume
```

恢复原检查点可使用原目录，恢复时以保存的配置为准。`completed.json` 表示预算结束或早停；套件跳过有完成标记和 best 的组。非空目录但没有可恢复检查点时，需要指定新的输出根目录。

## 评测、预测与汇总

```bash
python run.py evaluate task1 --checkpoint results/captioning_siglip/siglip_swiglu/seed_2026/best.pt --split val
python run.py evaluate task1 --checkpoint results/captioning_siglip/siglip_swiglu/seed_2026/best.pt --split test
python run.py predict task1 --checkpoint results/captioning_siglip/siglip_swiglu/seed_2026/best.pt --image path/to/food.jpg
python run.py summarize task1
python run.py evaluate task1 --group ablation --split test
```

`evaluate` 默认 val；全指标需要 pycocoevalcap、Java 和 SPICE 资源。预测清单和评分分别落盘。套件的 `--exp` 可以指定子集，`--seeds 2026 2027 2028` 可重复实验，`--output-dir` 可切换整套结果根目录。

保留 `best.pt`、`last.pt`、词表、完整配置、数据哈希、最佳验证预测、每轮学习率和显存记录。`summary.csv` 与 `summary.md` 汇总最高 val CIDEr-D；测试评分带单独列前缀，不参与选型。

## 数据口径与历史实验

四组必须共用相同的清单和词表；本次实现不重建或改写已有数据划分。历史 Food500Cap 上游数据存在跨划分重复，报告应说明具体版本和审计结果。

```bash
python -m experiments.task1_captioning.audit_data --run-dir results/captioning_siglip/siglip_swiglu/seed_2026 --output results/captioning_siglip/siglip_swiglu/seed_2026/data_audit.json
```

可追加 `--pixel-hashes` 在 CPU 上检查实际图片像素。该检查不能证明不存在近重复或预训练数据重叠。

旧 ViT 的 `improvements/00_legacy.yaml` 与 `01_normalization.yaml` 保留为归一化对照。S/L 容量、512 续训、视觉 LoRA 续训不在当前实验菜单；旧检查点所需的加载和推理能力仍保留。不同配方的历史分数不作为本轮单因素消融。
