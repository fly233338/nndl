"""提供共用视觉编码器、注意力和自回归解码器。"""

import math

import torch

nn = torch.nn


def load_resnet(pretrained: bool = True, model_path: str | None = None):
    if model_path:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        return model, 2048
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    return nn.Sequential(*list(model.children())[:-2]), 2048


class _FrozenModule(nn.Module):
    """固定视觉模块参数，并在外层切换训练模式时保持其评估状态。"""

    def _set_frozen(self, module: nn.Module, freeze: bool) -> None:
        self._freeze = freeze
        if freeze:
            for parameter in module.parameters():
                parameter.requires_grad = False
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze", False):
            getattr(self, self._frozen_name).eval()
        return self


class ResNetGlobal(_FrozenModule):
    """将 CNN 特征图平均池化为整图特征。"""

    def __init__(self, freeze: bool = True, model_path: str | None = None):
        super().__init__()
        self.features, self.output_dim = load_resnet(model_path=model_path)
        self._frozen_name = "features"
        self.huggingface = bool(model_path)
        self._set_frozen(self.features, freeze)

    def forward(self, images):
        fmap = self.features(images)
        if self.huggingface:
            fmap = fmap.last_hidden_state
        return fmap.mean(dim=(-2, -1))


class ResNetGrid(_FrozenModule):
    """将 CNN 特征图展平为局部网格序列。"""

    def __init__(self, freeze: bool = True, model_path: str | None = None):
        super().__init__()
        self.features, self.output_dim = load_resnet(model_path=model_path)
        self._frozen_name = "features"
        self.huggingface = bool(model_path)
        self._set_frozen(self.features, freeze)

    def forward(self, images):
        fmap = self.features(images)
        if self.huggingface:
            fmap = fmap.last_hidden_state
        return fmap.flatten(2).transpose(1, 2)


class ViTPatches(_FrozenModule):
    """提取 ViT 特征，支持位置插值和末层微调。"""

    def __init__(self, freeze: bool = True, model_path: str | None = None, include_cls=False):
        super().__init__()
        if model_path:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.output_dim = int(self.model.config.hidden_size)
            self.huggingface = True
        else:
            from torchvision.models import ViT_B_16_Weights, vit_b_16

            self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.output_dim = self.model.hidden_dim
            self.huggingface = False
        self._frozen_name = "model"
        self._set_frozen(self.model, freeze)
        self.include_cls = include_cls
        self._trainable_blocks = None

    def _blocks_and_norm(self):
        if not self.huggingface:
            return list(self.model.encoder.layers), self.model.encoder.ln
        if hasattr(self.model, "encoder"):
            return list(self.model.encoder.layer), self.model.layernorm
        return list(self.model.layers), self.model.layernorm

    def set_trainable_blocks(self, count):
        blocks, norm = self._blocks_and_norm()
        if not 0 <= count <= len(blocks):
            raise ValueError(f"trainable_blocks must be between 0 and {len(blocks)}")
        self.model.requires_grad_(False)
        for block in blocks[-count:] if count else []:
            block.requires_grad_(True)
        norm.requires_grad_(count > 0)
        self._trainable_blocks, self._freeze = count, count == 0
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_trainable_blocks", None) is not None:
            self.model.eval()
            blocks, norm = self._blocks_and_norm()
            for block in blocks[-self._trainable_blocks:] if self._trainable_blocks else []:
                block.train(mode)
            norm.train(mode and self._trainable_blocks > 0)
        return self

    def forward(self, images):
        if self.huggingface:
            tokens = self.model(pixel_values=images, interpolate_pos_encoding=True).last_hidden_state
        else:
            x = self.model.conv_proj(images).flatten(2).transpose(1, 2)
            n = x.shape[0]
            x = torch.cat([self.model.class_token.expand(n, -1, -1), x], dim=1)
            position = self.model.encoder.pos_embedding
            if x.size(1) != position.size(1):
                side = int(math.sqrt(position.size(1) - 1))
                patches = position[:, 1:].transpose(1, 2).reshape(1, self.output_dim, side, side)
                patches = torch.nn.functional.interpolate(
                    patches, size=(images.size(-2) // self.model.patch_size, images.size(-1) // self.model.patch_size),
                    mode="bicubic", align_corners=True,
                ).flatten(2).transpose(1, 2)
                position = torch.cat([position[:, :1], patches], dim=1)
            x = self.model.encoder.dropout(x + position)
            tokens = self.model.encoder.ln(self.model.encoder.layers(x))
        return tokens if self.include_cls else tokens[:, 1:]


class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature = nn.Linear(feature_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, features, hidden):
        scores = self.score(
            torch.tanh(self.feature(features) + self.hidden(hidden).unsqueeze(1)),
        ).squeeze(
            -1,
        )
        weights = scores.softmax(dim=1)
        return (features * weights.unsqueeze(-1)).sum(dim=1), weights


def _best_beams(candidates, beam_size):
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:beam_size]


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, feature_dim: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.init = nn.Linear(feature_dim, hidden_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, tokens):
        hidden = self.init(features).unsqueeze(0)
        outputs, _ = self.gru(self.embedding(tokens), hidden)
        return self.output(outputs)

    @torch.no_grad()
    def generate(self, features, bos_id: int, eos_id: int, max_length: int, beam_size: int = 1):
        if beam_size <= 1:
            hidden = self.init(features).unsqueeze(0)
            current = torch.full(
                (features.size(0), 1),
                bos_id,
                device=features.device,
                dtype=torch.long,
            )
            outputs = [current]
            finished = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)
            for _ in range(max_length - 1):
                logits, hidden = self.gru(self.embedding(current[:, -1:]), hidden)
                next_token = self.output(logits[:, -1]).argmax(-1, keepdim=True)
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
                outputs.append(next_token)
                current = torch.cat([current, next_token], dim=1)
                finished |= next_token[:, 0].eq(eos_id)
                if finished.all():
                    break
            return torch.cat(outputs, dim=1)

        results = []
        for sample in features:
            initial_hidden = self.init(sample.unsqueeze(0)).unsqueeze(0)
            beams = [(torch.tensor([[bos_id]], device=features.device), initial_hidden, 0.0, False)]
            for _ in range(max_length - 1):
                candidates = []
                for sequence, hidden, score, finished in beams:
                    if finished:
                        candidates.append((sequence, hidden, score, True))
                        continue
                    logits, next_hidden = self.gru(self.embedding(sequence[:, -1:]), hidden)
                    log_probs = self.output(logits[:, -1]).log_softmax(-1)
                    values, indices = torch.topk(
                        log_probs,
                        min(beam_size, log_probs.size(-1)),
                        dim=-1,
                    )
                    for value, index in zip(values[0], indices[0]):
                        token = index.view(1, 1)
                        candidates.append(
                            (
                                torch.cat([sequence, token], dim=1),
                                next_hidden.clone(),
                                score + float(value),
                                bool(index == eos_id),
                            ),
                        )
                beams = _best_beams(candidates, beam_size)
                if all(item[3] for item in beams):
                    break
            results.append(beams[0][0].squeeze(0))

        width = max(sequence.numel() for sequence in results)
        output = torch.full((len(results), width), eos_id, device=features.device, dtype=torch.long)
        for index, sequence in enumerate(results):
            output[index, : sequence.numel()] = sequence
        return output


def _grid_position(n: int, dim: int, device, dtype):
    """为正方形视觉网格生成二维正弦位置编码。"""
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"CNN grid must be square, got {n} tokens")
    coordinates = torch.arange(side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    half = math.ceil(dim / 4)
    frequencies = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / half),
    )
    parts = [
        torch.sin(xx.reshape(-1, 1) * frequencies),
        torch.cos(xx.reshape(-1, 1) * frequencies),
        torch.sin(yy.reshape(-1, 1) * frequencies),
        torch.cos(yy.reshape(-1, 1) * frequencies),
    ]
    return torch.cat(parts, dim=1)[:, :dim].unsqueeze(0)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        embedding_dim: int,
        layers: int,
        heads: int,
        max_length: int = 128,
        use_visual_encoder: bool = True,
        use_2d_position: bool = False,
        pad_id: int = 0,
        dropout: float = 0.1,
        norm_first: bool = False,
        feedforward_dim: int = 2048,
        activation: str = "relu",
        tie_embeddings: bool = False,
        global_fusion: str = "none",
        decoding: dict | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position = nn.Embedding(max_length, embedding_dim)
        self.input_projection = nn.Linear(feature_dim, embedding_dim)
        self.use_visual_encoder = use_visual_encoder
        self.use_2d_position = use_2d_position
        self.pad_id = pad_id
        self.global_fusion = global_fusion
        self.decoding = decoding or {}
        if global_fusion == "gated":
            self.global_projection = nn.Linear(feature_dim, feature_dim)
            self.global_gate = nn.Linear(feature_dim * 2, feature_dim)
        layer_options = dict(
            dim_feedforward=feedforward_dim, dropout=dropout,
            norm_first=norm_first, activation=activation, batch_first=True,
        )
        if use_visual_encoder:
            encoder_layer = nn.TransformerEncoderLayer(embedding_dim, heads, **layer_options)
            self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        else:
            self.encoder = None
        decoder_layer = nn.TransformerDecoderLayer(embedding_dim, heads, **layer_options)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, layers, norm=nn.LayerNorm(embedding_dim) if norm_first else None,
        )
        self.output = nn.Linear(embedding_dim, vocab_size)
        if tie_embeddings:
            self.output.weight = self.embedding.weight

    def _encode(self, features):
        """投影视觉特征，并按配置添加网格位置编码和视觉编码层。"""
        if self.global_fusion == "gated":
            global_token, patches = features[:, :1], features[:, 1:]
            context = global_token.expand(-1, patches.size(1), -1)
            gate = self.global_gate(torch.cat([patches, context], dim=-1)).sigmoid()
            features = patches + gate * self.global_projection(context)
        projected = self.input_projection(features)
        if self.use_2d_position:
            projected = projected + _grid_position(
                projected.size(1),
                projected.size(2),
                projected.device,
                projected.dtype,
            )
        return self.encoder(projected) if self.encoder is not None else projected

    def _decode(self, memory, tokens):
        """使用因果掩码预测下一词元，屏蔽输入中的填充位置。"""
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        embedded = self.embedding(tokens) + self.position(positions)
        mask = torch.ones(
            (tokens.size(1), tokens.size(1)),
            device=tokens.device,
            dtype=torch.bool,
        ).triu(
            1,
        )
        return self.output(
            self.decoder(
                embedded,
                memory,
                tgt_mask=mask,
                tgt_key_padding_mask=tokens.eq(self.pad_id),
            ),
        )

    def forward(self, features, tokens):
        return self._decode(self._encode(features), tokens)

    def _filter_logits(self, logits, tokens, bos_id, eos_id):
        if self.decoding.get("suppress_special_tokens", False):
            logits[:, [self.pad_id, bos_id]] = -torch.inf
        if tokens.size(1) - 1 < self.decoding.get("min_length", 0):
            logits[:, eos_id] = -torch.inf
        size = int(self.decoding.get("no_repeat_ngram_size", 0))
        if size and tokens.size(1) >= size:
            for row, sequence in enumerate(tokens.tolist()):
                prefix = tuple(sequence[-(size - 1):]) if size > 1 else ()
                banned = [
                    sequence[index + size - 1]
                    for index in range(len(sequence) - size + 1)
                    if tuple(sequence[index:index + size - 1]) == prefix
                ]
                logits[row, banned] = -torch.inf
        return logits

    @torch.no_grad()
    def generate(self, features, bos_id: int, eos_id: int, max_length: int, beam_size: int = 1):
        memory = self._encode(features)
        if beam_size <= 1:
            tokens = torch.full(
                (features.size(0), 1),
                bos_id,
                device=features.device,
                dtype=torch.long,
            )
            finished = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)
            for _ in range(max_length - 1):
                logits = self._filter_logits(self._decode(memory, tokens)[:, -1], tokens, bos_id, eos_id)
                next_token = logits.argmax(-1, keepdim=True)
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
                tokens = torch.cat([tokens, next_token], dim=1)
                finished |= next_token[:, 0].eq(eos_id)
                if finished.all():
                    break
            return tokens

        batch, _, dim = memory.shape
        beam_size = min(beam_size, self.output.out_features)
        memory = memory[:, None].expand(-1, beam_size, -1, -1).reshape(batch * beam_size, -1, dim)
        tokens = torch.full((batch, beam_size, 1), bos_id, device=features.device, dtype=torch.long)
        scores = torch.full((batch, beam_size), -torch.inf, device=features.device)
        scores[:, 0] = 0
        finished = torch.zeros_like(scores, dtype=torch.bool)
        lengths = torch.zeros_like(scores, dtype=torch.long)
        alpha = float(self.decoding.get("length_penalty", 0))
        for _ in range(max_length - 1):
            flat_tokens = tokens.reshape(batch * beam_size, -1)
            logits = self._decode(memory, flat_tokens)[:, -1]
            logits = self._filter_logits(logits, flat_tokens, bos_id, eos_id)
            log_probs = logits.log_softmax(-1).reshape(batch, beam_size, -1)
            log_probs = log_probs.masked_fill(finished[:, :, None], -torch.inf)
            log_probs[:, :, eos_id] = torch.where(finished, 0.0, log_probs[:, :, eos_id])
            candidate_scores = scores[:, :, None] + log_probs
            candidate_lengths = lengths + (~finished).long()
            rank = candidate_scores / ((5 + candidate_lengths[:, :, None]).float() / 6).pow(alpha)
            _, selected = rank.flatten(1).topk(beam_size, dim=-1)
            parents = selected // log_probs.size(-1)
            next_tokens = selected % log_probs.size(-1)
            tokens = tokens.gather(1, parents[:, :, None].expand(-1, -1, tokens.size(-1)))
            tokens = torch.cat([tokens, next_tokens[:, :, None]], dim=-1)
            scores = candidate_scores.flatten(1).gather(1, selected)
            lengths = candidate_lengths.gather(1, parents)
            finished = finished.gather(1, parents) | next_tokens.eq(eos_id)
            if finished.all():
                break
        return tokens[:, 0]
