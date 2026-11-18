# 选做任务运行命令

在 Linux 服务器的项目根目录执行，使用已安装依赖的 `lens` 环境。环境准备见 [环境与数据准备](task2_training.md)。下面复用已有模型服务；接口地址和模型 ID 按实际部署填写。

## 1. RL：CIDEr-D 策略梯度微调

需要任务一数据清单、图片、ViT backbone 和 `models/best.pt`。路径与训练参数在 `config/optional/caption_rl.yaml`。

加载时自动兼容新旧 Transformers 的 ViT 参数名，并严格检查全部权重。成功后打印 `Loaded complete checkpoint`；参数缺失或形状不符仍会停止，不会跳过未加载的权重。

```bash
python -m experiments.optional.caption_rl.reward
python -m experiments.optional.caption_rl.train
python -m experiments.optional.caption_rl.evaluate --split val
```

第一步只缓存训练参考文本和 n-gram 文档频率。训练时由当前策略重新采样，实时计算 CIDEr-D，以 greedy 奖励为基线更新解码器。验证集确定参数后执行最终测试：

```bash
python -m experiments.optional.caption_rl.evaluate --split test
```

结果在 `results/caption_rl/vit_transformer_scst/comparison/`。新训练需使用新的 `output_dir`，当前不支持断点续训。奖励分词和传统文本评测需 Java，完整评测还需 CoreNLP。

## 2. LLM-as-Judge

需要任务一的图片、manifest 和预测描述。配置为 `config/optional/llm_judge.yaml`；服务只有一个模型时可自动识别模型 ID，否则设置 `JUDGE_MODEL`。

```bash
export JUDGE_BASE_URL=http://127.0.0.1:8005/v1
export JUDGE_API_KEY=EMPTY
python -m experiments.optional.llm_judge.evaluate --check-api
python -m experiments.optional.llm_judge.evaluate --predictions results/captioning/vit_transformer/test/beam/predictions_beam.jsonl --limit 1 --output results/llm_judge/vit_transformer/smoke.jsonl
python -m experiments.optional.llm_judge.evaluate --predictions results/captioning/vit_transformer/test/beam/predictions_beam.jsonl --output results/llm_judge/vit_transformer/judgments.jsonl
```

接口检查不验证视觉能力，先确认单图评分成功。启用鉴权时替换密钥；客户端位于其他节点时替换主机地址。其他模型替换输入和输出路径即可。

与传统指标分析一致性：

```bash
python -m experiments.optional.llm_judge.metrics --predictions results/captioning/vit_transformer/test/beam/predictions_beam.jsonl --output results/llm_judge/vit_transformer/automatic_scores.jsonl
python -m experiments.optional.llm_judge.analyze --records results/llm_judge/vit_transformer/judgments.jsonl --automatic-scores results/llm_judge/vit_transformer/automatic_scores.jsonl --output results/llm_judge/vit_transformer/agreement.json
```

有真实人工评分时，第二条追加 `--human-ratings path/to/ratings.jsonl`。每行格式为 `{"image_id":"...","correctness":4,"coverage":3,"groundedness":4,"fluency":5}`，须对应同一模型的同一句描述。
